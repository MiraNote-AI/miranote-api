"""Sign in with Apple and our sessions (design 3.1, 3.2).

    POST /auth/apple    {identity_token, nonce}  -> token pair
    POST /auth/refresh  {refresh_token}          -> new token pair (rotation)
    POST /auth/signout  {refresh_token}          -> 204, revokes that sign-in
    GET  /.well-known/jwks.json                  -> our public signing keys
    GET  /me                                     -> {id}, behind current_user

Tokens never reach a log line: only the reason a token was refused.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

import apple
import config
import tokens
from models import Session, User

log = logging.getLogger("accounts.auth")

router = APIRouter()

# Two refreshes in parallel at app launch both present the same token; the
# one that loses the race within this grace gets a pair too instead of
# tripping reuse detection (design 3.2).
REFRESH_GRACE = dt.timedelta(seconds=10)

users = User.__table__
sessions = Session.__table__


class AppleSignIn(BaseModel):
    identity_token: str
    nonce: str


class RefreshRequest(BaseModel):
    refresh_token: str


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


async def _new_session(
    conn: AsyncConnection, request: Request, user_id: uuid.UUID, family_id: uuid.UUID, now: dt.datetime
) -> Dict[str, Any]:
    """Store a new refresh token and return the pair the client keeps."""
    refresh = tokens.new_refresh_token()
    await conn.execute(
        sessions.insert().values(
            user_id=user_id,
            family_id=family_id,
            token_hash=tokens.hash_refresh_token(refresh),
            created_at=now,
            expires_at=now + tokens.REFRESH_TTL,
        )
    )
    return {
        "access_token": request.app.state.signer.issue(user_id, now),
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": int(tokens.ACCESS_TTL.total_seconds()),
        "user_id": str(user_id),
    }


async def _revoke_family(conn: AsyncConnection, family_id: uuid.UUID, now: dt.datetime) -> None:
    await conn.execute(
        update(sessions)
        .where(sessions.c.family_id == family_id, sessions.c.revoked_at.is_(None))
        .values(revoked_at=now)
    )


@router.post("/auth/apple")
async def sign_in_with_apple(body: AppleSignIn, request: Request):
    try:
        identity = await apple.verify_identity_token(
            body.identity_token, body.nonce, request.app.state.apple_keys, config.APPLE_BUNDLE_ID
        )
    except apple.InvalidIdentityToken as exc:
        log.info("apple token refused", extra={"reason": str(exc)})
        raise _unauthorized("invalid identity token")
    except apple.AppleKeysUnavailable:
        # Our outage, not a bad token: the app should retry, not sign out.
        log.warning("apple token not checked: apple keys unavailable")
        raise HTTPException(
            503,
            detail="sign-in is unavailable, try again",
            headers={"Retry-After": str(apple.REFETCH_SECONDS)},
        )

    now = _now()
    async with request.app.state.engine.begin() as conn:
        stmt = insert(users).values(apple_sub=identity.sub, apple_team_id=config.APPLE_TEAM_ID)
        # DO UPDATE (a no-op) rather than DO NOTHING so RETURNING also gives
        # the id of an existing user, in one statement and race-free.
        stmt = stmt.on_conflict_do_update(
            constraint="uq_users_apple_team_sub", set_={"apple_sub": stmt.excluded.apple_sub}
        ).returning(users.c.id)
        user_id = (await conn.execute(stmt)).scalar_one()
        pair = await _new_session(conn, request, user_id, uuid.uuid4(), now)
    log.info("signed in", extra={"user_id": str(user_id)})
    return pair


@router.post("/auth/refresh")
async def refresh(body: RefreshRequest, request: Request):
    now = _now()
    reused = False
    async with request.app.state.engine.begin() as conn:
        # FOR UPDATE: a parallel refresh of the same token waits here and
        # then sees replaced_at, so it takes the grace path below.
        row = (
            await conn.execute(
                select(sessions)
                .where(sessions.c.token_hash == tokens.hash_refresh_token(body.refresh_token))
                .with_for_update()
            )
        ).one_or_none()
        if row is None or row.revoked_at is not None or row.expires_at <= now:
            raise _unauthorized("invalid refresh token")
        if row.replaced_at is not None and now - row.replaced_at > REFRESH_GRACE:
            # A replaced token used again: someone else holds a copy. End
            # the whole sign-in; the user signs in again (design 3.2).
            await _revoke_family(conn, row.family_id, now)
            reused = True
        else:
            if row.replaced_at is None:
                await conn.execute(
                    update(sessions).where(sessions.c.id == row.id).values(replaced_at=now)
                )
            pair = await _new_session(conn, request, row.user_id, row.family_id, now)
    if reused:
        log.warning("refresh token reuse, sign-in revoked", extra={"user_id": str(row.user_id)})
        raise _unauthorized("invalid refresh token")
    return pair


@router.post("/auth/signout", status_code=204)
async def sign_out(body: RefreshRequest, request: Request):
    """Idempotent: an unknown or already revoked token is still a 204."""
    async with request.app.state.engine.begin() as conn:
        family_id = (
            await conn.execute(
                select(sessions.c.family_id).where(
                    sessions.c.token_hash == tokens.hash_refresh_token(body.refresh_token)
                )
            )
        ).scalar_one_or_none()
        if family_id is not None:
            await _revoke_family(conn, family_id, _now())
    return Response(status_code=204)


@router.get("/.well-known/jwks.json")
async def jwks(request: Request):
    return {"keys": request.app.state.signer.jwks}


async def current_user(request: Request, authorization: Optional[str] = Header(None)) -> uuid.UUID:
    """The signed-in user's id from `Authorization: Bearer <access token>`.
    401 when the token is missing or invalid, or the user is gone or
    being deleted."""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("missing access token")
    try:
        user_id = request.app.state.signer.verify(token.strip())
    except tokens.InvalidAccessToken as exc:
        log.info("access token refused", extra={"reason": str(exc)})
        raise _unauthorized("invalid access token")
    async with request.app.state.engine.connect() as conn:
        found = (
            await conn.execute(
                select(users.c.id).where(users.c.id == user_id, users.c.deletion_requested_at.is_(None))
            )
        ).scalar_one_or_none()
    if found is None:
        raise _unauthorized("invalid access token")
    return user_id


@router.get("/me")
async def me(user_id: uuid.UUID = Depends(current_user)):
    return {"id": str(user_id)}

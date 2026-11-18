"""Sign in with Apple and our sessions (design 3.1, 3.2).

    POST /auth/apple    {identity_token, nonce}  -> token pair
    POST /auth/refresh  {refresh_token}          -> new token pair (rotation)
    POST /auth/signout  {refresh_token}          -> 204, revokes that sign-in
    GET  /.well-known/jwks.json                  -> our public signing keys
    GET  /me                                     -> {id}, behind current_user
    DELETE /me          {authorization_code}     -> deletes the account (design 7.1)

Tokens never reach a log line: only the reason a token was refused.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

import apple
import config
import purge
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


class DeleteAccount(BaseModel):
    # From Sign in with Apple run again just before the request (7.1).
    authorization_code: str


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
        # Signing in again before the purge has run gives a new, empty
        # account: the old one is purged first (design 3.1, 7.1 step 5).
        await purge.purge_deleted_accounts(
            conn, users.c.apple_team_id == config.APPLE_TEAM_ID, users.c.apple_sub == identity.sub
        )
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
        # Marked by DELETE /me. Its revoke can miss the successor of a
        # refresh that held this row's lock at the time; refuse that too.
        marked = (
            await conn.execute(select(users.c.deletion_requested_at).where(users.c.id == row.user_id))
        ).scalar_one_or_none()
        if marked is not None:
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


@router.delete("/me")
async def delete_me(body: DeleteAccount, request: Request, user_id: uuid.UUID = Depends(current_user)):
    """Marks the account and ends every sign-in at once; purge.py deletes
    the rest within 24 hours. Then Apple is asked to revoke the app's
    access. Our data is deleted whatever Apple answers: 200 when Apple
    revoked too, otherwise 502 (Apple refused or was unreachable, or the
    revoke failed on our side, e.g. an unusable key) or 503
    (no Sign in with Apple key), both with "deleted": true. There is no
    later retry: the code is single-use and we keep no Apple token."""
    now = _now()
    async with request.app.state.engine.begin() as conn:
        await conn.execute(update(users).where(users.c.id == user_id).values(deletion_requested_at=now))
        await conn.execute(
            update(sessions)
            .where(sessions.c.user_id == user_id, sessions.c.revoked_at.is_(None))
            .values(revoked_at=now)
        )
    log.info("account deletion requested", extra={"user_id": str(user_id)})
    try:
        await request.app.state.apple_revoker.revoke(body.authorization_code)
    except apple.RevokeNotConfigured:
        log.error("apple revoke skipped: not configured", extra={"user_id": str(user_id)})
        return JSONResponse(
            {"deleted": True, "apple_revoked": False, "detail": "Sign in with Apple revocation is not configured"},
            status_code=503,
        )
    except Exception as exc:  # noqa: BLE001 -- the account is deleted already; say so
        # RevokeFailed, or anything else, such as an unusable SIWA key or an
        # answer from Apple that is not JSON; of those only the exception's
        # type is logged.
        reason = str(exc) if isinstance(exc, apple.RevokeFailed) else type(exc).__name__
        log.warning("apple revoke failed", extra={"user_id": str(user_id), "reason": reason})
        return JSONResponse(
            {"deleted": True, "apple_revoked": False, "detail": "Apple did not revoke the sign-in"},
            status_code=502,
        )
    return {"deleted": True, "apple_revoked": True}

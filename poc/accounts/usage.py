"""The AI usage ledger (design 6.1, 6.2).

    POST /usage/reserve  internal  {user_id, feature, request_id, units, model, cost_usd}
    POST /usage/settle   internal  {request_id, status: ok|failed, cost_usd}
    GET  /me/usage       app       remaining per feature and reset times

The AI services call reserve before a paid model call and settle after it.
Reserve checks the cap and writes a `pending` row in one transaction that
holds the user's row lock, so two parallel requests of one user cannot both
take the last unit. A `failed` settle is not charged. A retried request_id
runs again only when its row failed; a pending or ok one answers 409
duplicate_request without a model call (design 6.1). A feature over its
daily cost budget answers 503 service_paused for everyone (budget.py).

The internal routes take `X-Internal-Secret` (ACCOUNTS_INTERNAL_SECRET) and
are never reachable through the Cloudflare tunnel: the tunnel's ingress
answers 404 for /usage/* (README.md), and a request that still arrives with
cloudflared's Cf-Connecting-IP header is answered 404 here as well.
"""

from __future__ import annotations

import datetime as dt
import hmac
import logging
import uuid
from decimal import Decimal
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update

import budget
import quotas
from auth import current_user
from models import AiUsage, User

log = logging.getLogger("accounts.usage")

router = APIRouter()

ai_usage = AiUsage.__table__
users = User.__table__


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(value: Optional[dt.datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def internal_caller(request: Request, x_internal_secret: Optional[str] = Header(None)) -> None:
    """Only the AI services, on this machine's loopback, may call."""
    if request.headers.get("cf-connecting-ip") is not None:
        raise HTTPException(404)
    if not x_internal_secret or not hmac.compare_digest(
        x_internal_secret.encode(), request.app.state.internal_secret.encode()
    ):
        raise HTTPException(403, detail="internal route")


class ReserveRequest(BaseModel):
    user_id: uuid.UUID
    feature: str
    request_id: uuid.UUID
    units: int = Field(1, ge=1, le=10)
    model: str = Field("", max_length=200)
    # The call's estimated cost, kept on the pending row until settle sets
    # the real one, so the daily budgets (budget.py) see calls in flight.
    cost_usd: Decimal = Field(Decimal(0), ge=0)


class SettleRequest(BaseModel):
    request_id: uuid.UUID
    status: Literal["ok", "failed"]
    cost_usd: Decimal = Field(Decimal(0), ge=0)


@router.post("/usage/reserve", dependencies=[Depends(internal_caller)])
async def reserve(body: ReserveRequest, request: Request):
    if body.feature not in quotas.METERED_FEATURES:
        raise HTTPException(400, detail="unknown feature")
    now = _now()
    async with request.app.state.engine.begin() as conn:
        # The lock that makes check-then-insert safe: every reserve of this
        # user waits here for the one before it to commit.
        user = (
            await conn.execute(
                select(users.c.deletion_requested_at).where(users.c.id == body.user_id).with_for_update()
            )
        ).one_or_none()
        if user is None or user.deletion_requested_at is not None:
            return JSONResponse({"reason": "unknown_user"}, status_code=401)

        existing = (
            await conn.execute(select(ai_usage).where(ai_usage.c.request_id == body.request_id))
        ).one_or_none()
        if existing is not None:
            if existing.user_id != body.user_id or existing.feature != body.feature:
                return JSONResponse({"reason": "request_id_conflict"}, status_code=409)
            if existing.status != "failed":
                # The result was never stored, so running it again would be
                # a free, uncounted call; the user loses this unit (6.1).
                return JSONResponse({"reason": "duplicate_request"}, status_code=409)

        paused = await budget.paused(conn, body.feature, request.app.state.budgets, now)
        if paused is not None:
            return JSONResponse(paused, status_code=503)

        plan, environment = await quotas.entitlement(conn, body.user_id, now)
        cap = quotas.CAPS[(plan, environment)][body.feature]
        used = await quotas.usage(conn, body.user_id, cap, now)
        if used.used + body.units > cap.limit:
            log.info("quota reached", extra={"user_id": str(body.user_id), "feature": body.feature, "plan": plan})
            return JSONResponse(
                {
                    "reason": "quota",
                    "feature": body.feature,
                    "plan": plan,
                    "limit": cap.limit,
                    "resets_at": _iso(used.resets_at),
                },
                status_code=429,
            )

        row = dict(
            units=body.units,
            model=body.model,
            environment=environment,
            status="pending",
            cost_usd=body.cost_usd,
            created_at=now,
        )
        if existing is not None:
            # A failed request retried with the same id: charge it afresh.
            await conn.execute(update(ai_usage).where(ai_usage.c.request_id == body.request_id).values(**row))
        else:
            await conn.execute(
                ai_usage.insert().values(
                    request_id=body.request_id, user_id=body.user_id, feature=body.feature, **row
                )
            )
    return {
        "request_id": str(body.request_id),
        "plan": plan,
        "remaining": cap.limit - used.used - body.units,
    }


@router.post("/usage/settle", dependencies=[Depends(internal_caller)])
async def settle(body: SettleRequest, request: Request):
    """Idempotent: only a pending row changes, so a late or repeated settle
    never turns a charged request into a free one or back."""
    async with request.app.state.engine.begin() as conn:
        changed = (
            await conn.execute(
                update(ai_usage)
                .where(ai_usage.c.request_id == body.request_id, ai_usage.c.status == "pending")
                .values(status=body.status, cost_usd=body.cost_usd if body.status == "ok" else 0)
                .returning(ai_usage.c.status)
            )
        ).scalar_one_or_none()
        if changed is not None:
            return {"status": changed}
        current = (
            await conn.execute(select(ai_usage.c.status).where(ai_usage.c.request_id == body.request_id))
        ).scalar_one_or_none()
    if current is None:
        return JSONResponse({"reason": "unknown_request"}, status_code=404)
    return {"status": current}


@router.get("/me/usage")
async def my_usage(request: Request, user_id: uuid.UUID = Depends(current_user)):
    now = _now()
    async with request.app.state.engine.connect() as conn:
        plan, environment = await quotas.entitlement(conn, user_id, now)
        features = {}
        for feature, cap in quotas.CAPS[(plan, environment)].items():
            used = await quotas.usage(conn, user_id, cap, now)
            features[feature] = {
                "limit": used.limit,
                "used": used.used,
                "remaining": used.remaining,
                "window": used.window,
                "resets_at": _iso(used.resets_at),
            }
    return {"plan": plan, "environment": environment, "features": features}

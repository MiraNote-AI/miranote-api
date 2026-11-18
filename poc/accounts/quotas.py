"""Entitlement and AI quotas (design 2, 5.1, 5.2, 6.2).

Every cap lives in CAPS below; change a number there and restart. A cap
counts the `units` of the ai_usage rows of one or more features in a
window; `pending` and `ok` rows count, `failed` rows do not.

Windows: rolling 24 h and rolling 7 days, or the UTC calendar month.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncConnection

from models import AiUsage, Subscription

ai_usage = AiUsage.__table__
subscriptions = Subscription.__table__

DAY = "rolling_24h"
WEEK = "rolling_7d"
MONTH = "utc_month"
_ROLLING = {DAY: dt.timedelta(hours=24), WEEK: dt.timedelta(days=7)}

FREE = "free"
PLUS = "plus"


@dataclass(frozen=True)
class Cap:
    limit: int
    window: str
    # The features whose rows count against this cap. Plus images share
    # one cap between generation and edits.
    counts: Tuple[str, ...]


_IMAGES = ("image_generate", "image_edit")

# (plan, environment) -> feature -> cap. Starting values from design 2.
# A Sandbox entitlement (App Review, TestFlight: Plus for free) has its own,
# lower caps on every feature so public testers cannot run up a real bill
# (5.2). /mood is metered only for signed-in callers here; signed-out
# /mood waits for App Attest (6.5).
_FREE_CAPS = {
    "chat": Cap(5, DAY, ("chat",)),
    "expand": Cap(10, DAY, ("expand",)),
    "image_generate": Cap(5, MONTH, ("image_generate",)),
    "image_edit": Cap(2, MONTH, ("image_edit",)),
    "photo_push": Cap(1, WEEK, ("photo_push",)),
    "mood": Cap(30, DAY, ("mood",)),
}
_PLUS_CAPS = {
    "chat": Cap(200, DAY, ("chat",)),
    "expand": Cap(300, DAY, ("expand",)),
    "image_generate": Cap(100, MONTH, _IMAGES),
    "image_edit": Cap(100, MONTH, _IMAGES),
    "photo_push": Cap(1, DAY, ("photo_push",)),
    "mood": Cap(30, DAY, ("mood",)),
}
_SANDBOX_PLUS_CAPS = {
    "chat": Cap(30, DAY, ("chat",)),
    "expand": Cap(30, DAY, ("expand",)),
    "image_generate": Cap(10, MONTH, _IMAGES),
    "image_edit": Cap(10, MONTH, _IMAGES),
    "photo_push": Cap(1, WEEK, ("photo_push",)),
    "mood": Cap(30, DAY, ("mood",)),
}
CAPS: Dict[Tuple[str, str], Dict[str, Cap]] = {
    (FREE, "Production"): _FREE_CAPS,
    (PLUS, "Production"): _PLUS_CAPS,
    (PLUS, "Sandbox"): _SANDBOX_PLUS_CAPS,
}

METERED_FEATURES = frozenset(_FREE_CAPS)


@dataclass(frozen=True)
class Usage:
    limit: int
    used: int
    window: str
    resets_at: Optional[dt.datetime]

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)


async def entitlement(conn: AsyncConnection, user_id: uuid.UUID, now: dt.datetime) -> Tuple[str, str]:
    """(plan, environment). Plus = an owned, unrevoked chain whose expiry or
    grace period is in the future, read from timestamps so a stale `status`
    never locks anyone out (5.1). A Production chain wins over a Sandbox
    one. Free users are tagged Production."""
    environments = (
        await conn.execute(
            select(subscriptions.c.environment).where(
                subscriptions.c.user_id == user_id,
                subscriptions.c.revoked_at.is_(None),
                or_(subscriptions.c.expires_at > now, subscriptions.c.grace_period_expires_at > now),
            )
        )
    ).scalars().all()
    if not environments:
        return FREE, "Production"
    return PLUS, "Production" if "Production" in environments else "Sandbox"


def _month_start(now: dt.datetime) -> dt.datetime:
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month_start(now: dt.datetime) -> dt.datetime:
    start = _month_start(now)
    return start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)


async def usage(conn: AsyncConnection, user_id: uuid.UUID, cap: Cap, now: dt.datetime) -> Usage:
    """Units used against `cap` and when the next unit frees up: the start
    of next month, or when the oldest counted row leaves the rolling window
    (None when nothing is counted)."""
    since = _month_start(now) if cap.window == MONTH else now - _ROLLING[cap.window]
    counted = and_(
        ai_usage.c.user_id == user_id,
        ai_usage.c.feature.in_(cap.counts),
        ai_usage.c.status != "failed",
        ai_usage.c.created_at > since,
    )
    used, oldest = (
        await conn.execute(
            select(func.coalesce(func.sum(ai_usage.c.units), 0), func.min(ai_usage.c.created_at)).where(counted)
        )
    ).one()
    if cap.window == MONTH:
        resets_at: Optional[dt.datetime] = _next_month_start(now)
    elif oldest is None:
        resets_at = None
    else:
        leaves_last = oldest
        if used > cap.limit:
            # Over the cap (Plus ran out, or a cap was lowered): a unit frees
            # up only when the newest rows left in the window hold fewer
            # than `limit` units, i.e. when the newest row that brings them
            # to `limit` leaves -- a later row than the oldest.
            newer = func.sum(ai_usage.c.units).over(
                order_by=(ai_usage.c.created_at.desc(), ai_usage.c.request_id.desc())
            )
            ranked = select(ai_usage.c.created_at, newer.label("newer")).where(counted).subquery()
            leaves_last = (
                await conn.execute(
                    select(func.max(ranked.c.created_at)).where(ranked.c.newer >= cap.limit)
                )
            ).scalar_one()
        resets_at = leaves_last + _ROLLING[cap.window]
    return Usage(limit=cap.limit, used=int(used), window=cap.window, resets_at=resets_at)

"""The cost kill switch (design 6.3).

Each feature can have a daily budget in USD, and all features together a
global one. /usage/reserve sums today's cost_usd from ai_usage (ok and
pending rows) and, once a budget is reached, answers 503 for everyone:

    {"reason": "service_paused", "feature", "resets_at", "detail"}

until the next UTC midnight, or until the budget is raised (restart).

    ACCOUNTS_DAILY_BUDGET_USD                 all features together
    ACCOUNTS_DAILY_BUDGET_USD_<FEATURE>       one feature: CHAT, EXPAND,
                                              IMAGE_GENERATE, IMAGE_EDIT,
                                              PHOTO_PUSH, MOOD

Unset or empty means no budget, so nothing changes until one is set. 0
pauses the feature at once.

- The day is the UTC calendar day, not the rolling 24 h of design 6.3: it
  resets at a known time and matches what `python budget.py` prints.
- A pending row counts at the estimate the AI service sent with reserve
  (ai_auth.COST_USD) until settle sets the real cost, so calls in flight
  and calls whose settle never arrives count too. Reserves of different
  users at the same moment can still each see the spend before the others'
  rows commit (the lock is per user), and overshoot by those calls: at
  most one per database connection (15 with SQLAlchemy's default pool).
- Only reserved calls count. Beta-token calls never reserve (ai_auth.py),
  so they are neither counted nor stopped by a budget.

    .venv/bin/python budget.py      today's spend per feature, with budgets
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import sys
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Set, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

import config
import db
import quotas
from models import AiUsage

log = logging.getLogger("accounts.budget")

ai_usage = AiUsage.__table__

TOTAL = "total"
ENV_PREFIX = "ACCOUNTS_DAILY_BUDGET_USD"

# (day, scope) pairs already logged, so a tripped budget logs once a day
# per process, not on every refused request.
_tripped: Set[Tuple[dt.date, str]] = set()


def from_env() -> Dict[str, Decimal]:
    """feature (or TOTAL) -> budget. Read once at startup; a malformed value
    or an unknown feature stops the service rather than running unbudgeted."""
    budgets = {}
    for name, value in os.environ.items():
        if name == ENV_PREFIX:
            scope = TOTAL
        elif name.startswith(ENV_PREFIX + "_"):
            scope = name[len(ENV_PREFIX) + 1:].lower()
        else:
            continue
        if not value.strip():
            continue
        if scope != TOTAL and scope not in quotas.METERED_FEATURES:
            raise RuntimeError("{}: unknown feature {!r}".format(name, scope))
        try:
            amount = Decimal(value.strip())
        except InvalidOperation:
            raise RuntimeError("{} is not a number: {!r}".format(name, value))
        if not amount.is_finite() or amount < 0:
            raise RuntimeError("{} must be 0 or more: {!r}".format(name, value))
        budgets[scope] = amount
    return budgets


def _day_start(now: dt.datetime) -> dt.datetime:
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def spend_today(conn: AsyncConnection, now: dt.datetime) -> Dict[str, Decimal]:
    """feature -> cost_usd of today's (UTC) ok and pending rows, through the
    (feature, created_at) index. Features with no rows are left out."""
    rows = await conn.execute(
        select(ai_usage.c.feature, func.sum(ai_usage.c.cost_usd))
        .where(
            ai_usage.c.feature.in_(sorted(quotas.METERED_FEATURES)),
            ai_usage.c.status.in_(("ok", "pending")),
            ai_usage.c.created_at >= _day_start(now),
        )
        .group_by(ai_usage.c.feature)
    )
    return {feature: spent for feature, spent in rows}


async def paused(
    conn: AsyncConnection, feature: str, budgets: Dict[str, Decimal], now: dt.datetime
) -> Optional[Dict[str, Any]]:
    """The 503 body when `feature` has reached its own budget or the global
    one, else None. No query when neither is set."""
    scopes = [scope for scope in (feature, TOTAL) if scope in budgets]
    if not scopes:
        return None
    spend = await spend_today(conn, now)
    for scope in scopes:
        spent = sum(spend.values(), Decimal(0)) if scope == TOTAL else spend.get(feature, Decimal(0))
        if spent < budgets[scope]:
            continue
        if (now.date(), scope) not in _tripped:
            _tripped.add((now.date(), scope))
            log.warning(
                "daily budget reached",
                extra={"scope": scope, "budget_usd": str(budgets[scope]), "spent_usd": str(spent)},
            )
        return {
            "reason": "service_paused",
            "feature": feature,
            "resets_at": (_day_start(now) + dt.timedelta(days=1)).isoformat(),
            "detail": "this feature is paused for today, try again later",
        }
    return None


async def _main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    budgets = from_env()
    engine = db.create_engine(config.database_url())
    try:
        async with engine.connect() as conn:
            spend = await spend_today(conn, now)
    finally:
        await engine.dispose()
    print("{} (UTC) spend so far".format(now.date().isoformat()))
    print("{:<16}{:>12}{:>12}".format("feature", "spent_usd", "budget_usd"))
    for feature in sorted(quotas.METERED_FEATURES) + [TOTAL]:
        spent = sum(spend.values(), Decimal(0)) if feature == TOTAL else spend.get(feature, Decimal(0))
        budget = budgets.get(feature)
        print("{:<16}{:>12.4f}{:>12}".format(feature, spent, "-" if budget is None else str(budget)))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))

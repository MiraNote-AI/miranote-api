"""The purge and retention jobs (design 7.2). Both are plain DELETEs, safe
to run any number of times, in any order, on any number of hosts.

- Purge: every account marked by DELETE /me is deleted. The foreign keys
  do the rest: its sessions go, and subscriptions and ai_usage keep their
  rows with user_id set to null (pseudonymized). The running service does
  a pass every PURGE_INTERVAL (main.py), well inside the 24 hours the app
  promises; signing in again purges that one account at once (auth.py).
- Retention: rows past the periods in _retention are deleted, owned or
  not.
  app_store_notifications never held an owner: payloads are stored
  without appAccountToken (appstore.py).

    .venv/bin/python purge.py      one run now, prints the counts
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import sys
import uuid
from typing import Any, Dict, List

from sqlalchemy import Interval, delete, func, literal, literal_column
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

import config
import db
from models import AiUsage, AppAttestKey, AppStoreNotification, Session, Subscription, User

log = logging.getLogger("accounts.purge")

PURGE_INTERVAL = dt.timedelta(hours=1)

users = User.__table__


def _older_than(column: Any, now: dt.datetime, interval: str) -> Any:
    # Postgres month arithmetic, so "13 months" means calendar months.
    return column < literal(now) - literal_column("interval '{}'".format(interval), Interval)


def _retention(now: dt.datetime) -> Dict[str, Any]:
    """What each table keeps (README, Data retention)."""
    sessions = Session.__table__
    subscriptions = Subscription.__table__
    notifications = AppStoreNotification.__table__
    ai_usage = AiUsage.__table__
    attest = AppAttestKey.__table__
    # Postgres's greatest() skips a null grace period.
    chain_end = func.greatest(subscriptions.c.expires_at, subscriptions.c.grace_period_expires_at)
    return {
        # An expired refresh token is refused before any other check.
        "sessions": delete(sessions).where(sessions.c.expires_at < now),
        # Refunds, disputes and tax questions about a chain.
        "subscriptions": delete(subscriptions).where(_older_than(chain_end, now, "13 months")),
        "app_store_notifications": delete(notifications).where(
            _older_than(notifications.c.processed_at, now, "1 year")
        ),
        # Cost reports; no quota window is longer than a month.
        "ai_usage": delete(ai_usage).where(_older_than(ai_usage.c.created_at, now, "13 months")),
        "app_attest_keys": delete(attest).where(_older_than(attest.c.last_used_at, now, "90 days")),
    }


async def purge_deleted_accounts(conn: AsyncConnection, *where: Any) -> List[uuid.UUID]:
    """Delete the accounts marked for deletion (narrowed by `where`);
    returns their ids."""
    purged = (
        await conn.execute(
            delete(users).where(users.c.deletion_requested_at.is_not(None), *where).returning(users.c.id)
        )
    ).scalars().all()
    for user_id in purged:
        log.info("account purged", extra={"user_id": str(user_id)})
    return list(purged)


async def run_once(engine: AsyncEngine, now: dt.datetime) -> Dict[str, int]:
    """One purge and retention pass in one transaction; rows per table."""
    async with engine.begin() as conn:
        counts = {"users": len(await purge_deleted_accounts(conn))}
        for table, stmt in _retention(now).items():
            counts[table] = (await conn.execute(stmt)).rowcount
    log.info("purge run", extra=counts)
    return counts


async def run_forever(engine: AsyncEngine) -> None:
    """main.py runs this for the service's lifetime, every PURGE_INTERVAL.
    A failed run is logged and retried at the next one."""
    while True:
        await asyncio.sleep(PURGE_INTERVAL.total_seconds())
        try:
            await run_once(engine, dt.datetime.now(dt.timezone.utc))
        except Exception as exc:  # noqa: BLE001 -- the next run retries
            log.warning("purge run failed", extra={"error": type(exc).__name__})


async def _main() -> int:
    engine = db.create_engine(config.database_url())
    try:
        counts = await run_once(engine, dt.datetime.now(dt.timezone.utc))
    finally:
        await engine.dispose()
    for table, count in counts.items():
        print("{}: {} row(s) deleted".format(table, count))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))

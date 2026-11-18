"""Manual repair: recompute subscription rows from Apple (design 5.5).

Run after downtime, or when a row looks wrong. For each chain it calls the
App Store Server API's Get All Subscription Statuses and applies every
returned snapshot exactly as a notification would: the row is recomputed,
and an unowned row gets its owner only from a live appAccountToken.
No scheduled job in 1.0.

    .venv/bin/python backfill.py                  every chain in subscriptions
    .venv/bin/python backfill.py 2000000123 ...   these chains (Production
                                                  unless the row says Sandbox,
                                                  or --sandbox)

Needs DATABASE_URL, Apple's root certificates and the In-App Purchase key
(config.py).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

import appstore
import config
import db
from subscriptions import adopt_by_token, apply_snapshot, subscriptions


async def backfill(
    engine: AsyncEngine, store: appstore.AppStore, ids: List[str], default_env: str
) -> Tuple[int, List[str]]:
    """Returns the number of snapshots applied and the chains that failed.
    A failing chain (unknown to Apple, refused data, Apple unreachable)
    does not stop the others; rerun with its id."""
    if not store.verifiers or not store.clients:
        raise appstore.NotConfigured()
    async with engine.connect() as conn:
        known = dict(
            (await conn.execute(select(subscriptions.c.original_transaction_id, subscriptions.c.environment))).all()
        )
    chains = {otid: known.get(otid, default_env) for otid in ids} if ids else known
    applied = 0
    failed: List[str] = []
    for otid, environment in chains.items():
        try:
            response = await store.clients[environment].get_all_subscription_statuses(otid)
            snapshots = [
                await store.snapshot(last.signedTransactionInfo, last.signedRenewalInfo)
                for group in response.data or []
                for last in group.lastTransactions or []
            ]
            now = dt.datetime.now(dt.timezone.utc)
            async with engine.begin() as conn:
                for snap in snapshots:
                    await apply_snapshot(conn, snap, now)
                    await adopt_by_token(conn, snap, now, "backfill")
        except Exception as exc:  # noqa: BLE001 -- reported below; the chain is unchanged
            failed.append(otid)
            print("{}: failed ({})".format(otid, type(exc).__name__), file=sys.stderr)
            continue
        applied += len(snapshots)
        print("{}: {} snapshot(s)".format(otid, len(snapshots)))
    return applied, failed


async def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="*", help="original transaction ids (default: every known chain)")
    parser.add_argument("--sandbox", action="store_true", help="look up unknown ids in Sandbox")
    args = parser.parse_args(argv)
    store = appstore.from_config()
    engine = db.create_engine(config.database_url())
    try:
        applied, failed = await backfill(engine, store, args.ids, "Sandbox" if args.sandbox else "Production")
    except appstore.NotConfigured:
        print("needs APPSTORE_ROOT_CERTS, APPSTORE_APP_APPLE_ID and the APPSTORE_KEY_* settings", file=sys.stderr)
        return 2
    finally:
        await engine.dispose()
        await store.aclose()
    print("applied {} snapshot(s)".format(applied))
    if failed:
        print("not recomputed: {}".format(" ".join(failed)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))

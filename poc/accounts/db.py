"""The async SQLAlchemy engine.

Written for a transaction-mode pooler (Neon's pooled URL, PgBouncer) from
day one (design 8.1): no statement caches, unique prepared-statement names,
no session-level SET, no LISTEN/NOTIFY, no advisory locks.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Tuple

from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

log = logging.getLogger("accounts.db")

_POSTGRES_SCHEMES = ("postgresql", "postgres", "postgresql+asyncpg")

# libpq query options that asyncpg does not take as keyword arguments. Neon's
# connection strings carry both. sslmode becomes asyncpg's `ssl` (it accepts
# the same mode names); asyncpg has no channel binding, so that one is dropped.
_LIBPQ_ONLY = ("sslmode", "channel_binding")


def engine_args(raw_url: str) -> Tuple[URL, Dict[str, Any]]:
    """Turn a libpq-style DATABASE_URL into an asyncpg URL and connect_args.

    Errors never quote the URL: it carries the password.
    """
    url = make_url(raw_url)
    if url.drivername not in _POSTGRES_SCHEMES:
        raise ValueError("DATABASE_URL must be a postgresql:// URL")
    connect_args: Dict[str, Any] = {
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
        # A pooler may hand the next transaction to another server
        # connection, where asyncpg's numbered names can already be taken.
        "prepared_statement_name_func": lambda: "__asyncpg_{}__".format(uuid.uuid4()),
    }
    sslmode = url.query.get("sslmode")
    if sslmode:
        connect_args["ssl"] = sslmode
    url = url.set(drivername="postgresql+asyncpg").difference_update_query(_LIBPQ_ONLY)
    return url, connect_args


def create_engine(raw_url: str, **kw: Any) -> AsyncEngine:
    url, connect_args = engine_args(raw_url)
    # hide_parameters: an error's message, which uvicorn logs on a 500, would
    # otherwise end with the bound values (refresh-token hashes, Apple subs).
    return create_async_engine(
        url, connect_args=connect_args, pool_pre_ping=True, hide_parameters=True, **kw
    )


async def ping(engine: AsyncEngine, timeout: float) -> bool:
    """True when the database answers `SELECT 1` within `timeout` seconds."""

    async def _select_one() -> None:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    try:
        await asyncio.wait_for(_select_one(), timeout)
        return True
    except Exception as exc:  # noqa: BLE001 -- any failure means "not healthy"
        # The type only: driver messages can echo the host and user.
        log.warning("db ping failed", extra={"error": type(exc).__name__})
        return False

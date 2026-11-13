"""Shared fixtures. Every test here runs against a real Postgres -- never
SQLite (design 8.1) -- and never the network beyond it.

The database comes from ACCOUNTS_TEST_DATABASE_URL, deliberately not
DATABASE_URL: these tests drop and recreate every table, so a shell that has
sourced the service's own .env must not point them at a real database.
Locally (README.md "Tests"): a throwaway Postgres 17 in Docker or Homebrew.
In CI: the postgres:17 service container of .github/workflows/accounts.yml.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List

import pytest

POC_DIR = Path(__file__).resolve().parent.parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

TEST_URL = os.environ.get("ACCOUNTS_TEST_DATABASE_URL", "").strip()
if not TEST_URL:
    raise pytest.UsageError(
        "ACCOUNTS_TEST_DATABASE_URL is not set -- the accounts tests need a "
        "throwaway Postgres (see poc/accounts/README.md, Tests)"
    )
os.environ["DATABASE_URL"] = TEST_URL


def new_ec_key_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    return (
        ec.generate_private_key(ec.SECP256R1())
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )


# A throwaway signing key for this run; tests that rotate set their own.
os.environ["ACCOUNTS_SIGNING_KEY"] = new_ec_key_pem()
os.environ.pop("ACCOUNTS_PREVIOUS_PUBLIC_KEY", None)

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

import db  # noqa: E402

TABLES = ("users", "sessions", "subscriptions", "app_store_notifications", "ai_usage", "app_attest_keys")


def alembic_config() -> Config:
    return Config(str(POC_DIR / "alembic.ini"))


def run_sql(sql: str, **params: Any) -> List[Any]:
    """Run one statement in its own transaction; returns the rows, if any."""

    async def _run() -> List[Any]:
        engine = db.create_engine(TEST_URL, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                result = await conn.execute(text(sql), params)
                return list(result) if result.returns_rows else []
        finally:
            await engine.dispose()

    return asyncio.run(_run())


@pytest.fixture(scope="session")
def migrated():
    """The schema at head for the whole run, starting from an empty database."""
    cfg = alembic_config()
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield cfg


@pytest.fixture
def clean_db(migrated):
    """Empty tables for each test that writes rows."""
    run_sql("TRUNCATE {} CASCADE".format(", ".join(TABLES)))

"""Alembic up and down against real Postgres, and models.py kept in step
with the migrations."""

from __future__ import annotations

import asyncio

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy.pool import NullPool

import db
import models
from tests.conftest import TABLES, TEST_URL, run_sql


def _tables():
    rows = run_sql(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )
    return {r[0] for r in rows}


def test_upgrade_creates_exactly_the_tier1_tables(migrated):
    assert _tables() == set(TABLES) | {"alembic_version"}
    assert run_sql("SELECT version_num FROM alembic_version") == [("0001",)]


def test_downgrade_to_base_drops_everything_and_upgrade_restores_it(migrated):
    command.downgrade(migrated, "base")
    assert _tables() == {"alembic_version"}
    assert run_sql("SELECT version_num FROM alembic_version") == []

    command.upgrade(migrated, "head")
    assert _tables() == set(TABLES) | {"alembic_version"}


def test_every_id_is_uuid_and_every_time_is_timestamptz(migrated):
    rows = run_sql(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name <> 'alembic_version'"
    )
    for table, column, data_type in rows:
        if column.endswith("_at") or column == "signed_date" or column == "last_signed_date":
            assert data_type == "timestamp with time zone", (table, column)
        if column in ("id", "user_id", "family_id", "request_id"):
            assert data_type == "uuid", (table, column)


def test_models_match_the_migrations(migrated):
    async def _diff():
        engine = db.create_engine(TEST_URL, poolclass=NullPool)
        try:
            async with engine.connect() as conn:
                return await conn.run_sync(
                    lambda sync_conn: compare_metadata(
                        MigrationContext.configure(sync_conn), models.Base.metadata
                    )
                )
        finally:
            await engine.dispose()

    assert asyncio.run(_diff()) == []

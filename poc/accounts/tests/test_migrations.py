"""Alembic up and down against real Postgres, and models.py kept in step
with the migrations."""

from __future__ import annotations

import asyncio

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
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
    assert run_sql("SELECT version_num FROM alembic_version") == [("0002",)]


def test_downgrade_to_base_drops_everything_and_upgrade_restores_it(migrated):
    command.downgrade(migrated, "base")
    assert _tables() == {"alembic_version"}
    assert run_sql("SELECT version_num FROM alembic_version") == []

    command.upgrade(migrated, "head")
    assert _tables() == set(TABLES) | {"alembic_version"}


def _plan(sql):
    """EXPLAIN with sequential scans priced out: the tables are empty, so the
    plan shows whether an index fits the query, not what the planner would
    pick on real data. (On empty tables the per-user and per-device windows
    tie between indexes, so they are not pinned here.)"""

    async def _run():
        engine = db.create_engine(TEST_URL, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SET LOCAL enable_seqscan = off"))
                rows = await conn.execute(text("EXPLAIN " + sql))
                return "\n".join(r[0] for r in rows)
        finally:
            await engine.dispose()

    return asyncio.run(_run())


# Queries that run across every user, and the index each one needs: the
# cost kill switch (design 6.3) and revoking one sign-in (3.2).
_INDEXED_QUERIES = [
    (
        "cost kill switch",
        "SELECT sum(cost_usd) FROM ai_usage WHERE feature = 'image_generate' "
        "AND status <> 'failed' AND created_at > now() - interval '24 hours'",
        "ix_ai_usage_feature_created_at",
    ),
    (
        "revoke a sign-in",
        "UPDATE sessions SET revoked_at = now() "
        "WHERE family_id = '00000000-0000-0000-0000-000000000002' AND revoked_at IS NULL",
        "ix_sessions_family_id",
    ),
]


def test_the_cross_user_queries_have_an_index(migrated):
    for name, sql, index in _INDEXED_QUERIES:
        plan = _plan(sql)
        assert index in plan and "Index Cond" in plan, (name, plan)


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


# compare_metadata above does not compare CHECK constraints, server defaults
# or a foreign key's ON DELETE, so models.py could, say, cascade ai_usage on
# user deletion (the purge in 7.2 must keep those rows) and still pass. Here
# models.py builds its own copy of the schema in a scratch schema, and the
# two are compared as Postgres reports them.
_MODELS_SCHEMA = "models_probe"


def _catalog(schema):
    """Columns (exact type, NOT NULL, default), constraints and indexes of the
    tables in `schema`, with the schema name taken out so that two schemas
    compare equal."""
    columns = run_sql(
        "SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull, "
        "pg_get_expr(d.adbin, d.adrelid) FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace "
        "LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum "
        "WHERE n.nspname = :s AND c.relkind = 'r' AND c.relname <> 'alembic_version' "
        "AND a.attnum > 0 AND NOT a.attisdropped",
        s=schema,
    )
    constraints = run_sql(
        "SELECT t.relname, c.conname, pg_get_constraintdef(c.oid) FROM pg_constraint c "
        "JOIN pg_class t ON t.oid = c.conrelid JOIN pg_namespace n ON n.oid = t.relnamespace "
        "WHERE n.nspname = :s AND t.relname <> 'alembic_version'",
        s=schema,
    )
    indexes = run_sql(
        "SELECT tablename, indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = :s AND tablename <> 'alembic_version'",
        s=schema,
    )

    def unqualified(rows):
        return {tuple(str(v).replace(schema + ".", "") for v in row) for row in rows}

    return unqualified(columns), unqualified(constraints), unqualified(indexes)


def test_models_build_the_same_schema_as_the_migrations(migrated):
    async def _create_models_schema():
        engine = db.create_engine(TEST_URL, poolclass=NullPool)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("DROP SCHEMA IF EXISTS {} CASCADE".format(_MODELS_SCHEMA)))
                await conn.execute(text("CREATE SCHEMA {}".format(_MODELS_SCHEMA)))
                await conn.run_sync(
                    lambda sync_conn: models.Base.metadata.create_all(
                        sync_conn.execution_options(schema_translate_map={None: _MODELS_SCHEMA})
                    )
                )
        finally:
            await engine.dispose()

    asyncio.run(_create_models_schema())
    try:
        from_migrations = _catalog("public")
        from_models = _catalog(_MODELS_SCHEMA)
    finally:
        run_sql("DROP SCHEMA {} CASCADE".format(_MODELS_SCHEMA))

    for what, migrated_part, models_part in zip(
        ("columns", "constraints", "indexes"), from_migrations, from_models
    ):
        assert migrated_part, what
        assert sorted(migrated_part - models_part) == [], "only in the migrations: " + what
        assert sorted(models_part - migrated_part) == [], "only in models.py: " + what

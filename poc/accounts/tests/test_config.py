"""DATABASE_URL handling: required, and Neon's libpq URL made asyncpg-ready."""

from __future__ import annotations

import pytest

import config
import db

NEON_STYLE = (
    "postgresql://user:pw@ep-x-pooler.us-east-1.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)


def test_database_url_is_required(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        config.database_url()


def test_neon_url_becomes_asyncpg_with_ssl_and_no_statement_cache():
    url, connect_args = db.engine_args(NEON_STYLE)
    assert url.drivername == "postgresql+asyncpg"
    assert dict(url.query) == {}
    assert connect_args["ssl"] == "require"
    assert connect_args["statement_cache_size"] == 0
    assert connect_args["prepared_statement_cache_size"] == 0
    name = connect_args["prepared_statement_name_func"]
    assert name() != name()


def test_plain_local_url_has_no_ssl():
    url, connect_args = db.engine_args("postgresql://u:p@localhost:5432/x")
    assert url.drivername == "postgresql+asyncpg"
    assert "ssl" not in connect_args


def test_non_postgres_url_is_refused_without_echoing_it():
    with pytest.raises(ValueError) as err:
        db.engine_args("mysql://user:hunter2@host/db")
    assert "hunter2" not in str(err.value)

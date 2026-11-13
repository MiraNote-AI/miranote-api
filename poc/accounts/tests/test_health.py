"""/health against a real database and an unreachable one, and the request
log's promise: no bodies, no tokens."""

from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

import beta_auth
import main


@pytest.fixture
def client(migrated, monkeypatch):
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    beta_auth._limiter._hits.clear()  # the gate's rate limit is per process
    with TestClient(main.app) as c:
        yield c


def test_health_ok_with_the_database_up(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "db": "ok"}


def test_health_503_when_the_database_does_not_answer(migrated, monkeypatch):
    # Nothing listens on port 1; the service still starts (Neon may be
    # waking up) and /health tells the truth.
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    with TestClient(main.app) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "error", "db": "unreachable"}


def test_other_routes_need_the_beta_token(client):
    assert client.get("/nope").status_code == 401
    r = client.get("/nope", headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 404


def test_request_log_has_no_body_token_or_query(client):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(main.JsonFormatter())
    logging.getLogger("accounts").addHandler(handler)
    try:
        client.post(
            "/nope?refresh_token=secret-in-query",
            headers={"Authorization": "Bearer test-token"},
            json={"identity_token": "secret-in-body"},
        )
    finally:
        logging.getLogger("accounts").removeHandler(handler)

    out = buf.getvalue()
    assert "secret-in" not in out and "test-token" not in out
    line = json.loads(out.strip().splitlines()[-1])
    assert line["msg"] == "request"
    assert (line["method"], line["path"], line["status"]) == ("POST", "/nope", 404)

"""The beta gate is attached to this service, and a rejected request never
reaches the model (acceptance A5)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def _anonymous(authenticated):
    return TestClient(authenticated.app)


def test_compose_needs_a_token(client, fake_model):
    r = _anonymous(client).post("/collections/compose", data={"meta": "{}"})
    assert r.status_code == 401
    assert fake_model.calls == []


def test_wrong_token_is_rejected(client, fake_model):
    r = TestClient(client.app, headers={"Authorization": "Bearer nope"}).post(
        "/collections/compose", data={"meta": "{}"}
    )
    assert r.status_code == 401
    assert fake_model.calls == []


def test_jobs_route_needs_a_token(client):
    assert _anonymous(client).get("/collections/jobs/abc").status_code == 401


def test_health_needs_no_token(client):
    r = _anonymous(client).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm"] == "ok"
    assert set(body) >= {"model", "call_a_mode", "call_b_mode"}


def test_cors_preflight_is_not_rejected(client):
    r = _anonymous(client).options(
        "/collections/compose",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code == 200


def test_batch_poll_needs_token(client):
    assert _anonymous(client).get("/collections/jobs?ids=a,b").status_code == 401

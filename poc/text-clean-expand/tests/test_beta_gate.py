"""The beta gate is actually attached to this service.

tests/test_beta_auth.py proves the gate behaves correctly. These prove this
particular app is wired to it -- the failure mode being a service that quietly
ships without one while the tunnel in front of it is public.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _anonymous(authenticated):
    """A client with no Authorization header, over the same stubbed app."""
    return TestClient(authenticated.app)


def test_an_endpoint_needs_a_token(client):
    authenticated, _ = client
    r = _anonymous(authenticated).post("/polish", json={"text": "hello"})
    assert r.status_code == 401


def test_health_needs_no_token(client):
    authenticated, _ = client
    assert _anonymous(authenticated).get("/health").status_code == 200


def test_mounted_static_files_need_a_token(client):
    """This service mounts a UI, and mounts do not inherit route dependencies."""
    authenticated, _ = client
    assert _anonymous(authenticated).get("/static/index.html").status_code == 401


def test_cors_preflight_is_not_rejected(client):
    authenticated, _ = client
    r = _anonymous(authenticated).options(
        "/polish",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200

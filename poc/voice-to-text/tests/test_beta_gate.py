"""The beta gate is actually attached to this service.

This service mounts StaticFiles at "/" with html=True, which catches every
path no route matched. That mount is the reason the gate is middleware rather
than a route dependency: a dependency does not reach it.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _anonymous(authenticated):
    return TestClient(authenticated.app)


def test_an_endpoint_needs_a_token(voice_client):
    authenticated, _, _ = voice_client
    r = _anonymous(authenticated).post("/transcribe")
    assert r.status_code == 401


def test_health_needs_no_token(voice_client):
    authenticated, _, _ = voice_client
    assert _anonymous(authenticated).get("/health").status_code == 200


def test_the_catch_all_mount_needs_a_token(voice_client):
    authenticated, _, _ = voice_client
    anonymous = _anonymous(authenticated)
    for path in ("/", "/index.html", "/anything-not-a-route"):
        assert anonymous.get(path).status_code == 401, path


def test_cors_preflight_is_not_rejected(voice_client):
    authenticated, _, _ = voice_client
    r = _anonymous(authenticated).options(
        "/transcribe",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200

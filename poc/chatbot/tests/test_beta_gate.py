"""The beta gate is actually attached to this service."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BETA_TOKENS", "test-token")

from poc.chatbot import main  # noqa: E402


@pytest.fixture
def anonymous():
    return TestClient(main.app)


def test_an_endpoint_needs_a_token(anonymous):
    assert anonymous.post("/chat", json={"message": "hi"}).status_code == 401


def test_health_needs_no_token(anonymous):
    assert anonymous.get("/health").status_code == 200


def test_a_valid_token_gets_past_the_gate(anonymous):
    """Not a 401, whatever the endpoint then decides to do with the request."""
    r = anonymous.post(
        "/chat",
        json={"message": "hi"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code != 401


def test_cors_preflight_is_not_rejected(anonymous):
    r = anonymous.options(
        "/chat",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.status_code == 200

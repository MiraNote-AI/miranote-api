"""Signed-in requests: our access token, images metered (api #95 step 3).

/generate is `image_generate`; /stylize and the model-backed /border mode
are `image_edit`; /border's local "outline" mode is not metered. The gate
itself is tested in tests/test_ai_auth.py at the repo root. Every model
call is stubbed.
"""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import main
from fake_accounts import FakeAccounts

os.environ["BETA_TOKENS"] = "test-token"

BETA = {"Authorization": "Bearer test-token"}


class FakeGenerate:
    def __init__(self):
        self.calls = 0
        self.fail = 0  # a status to raise, or 0

    async def __call__(self, req, timings):
        self.calls += 1
        if self.fail:
            raise HTTPException(status_code=self.fail, detail="provider failed")
        return {"images": []}


@pytest.fixture
def svc(monkeypatch):
    import beta_auth

    beta_auth._limiter._hits.clear()
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    generate = FakeGenerate()
    monkeypatch.setattr(main, "_generate", generate)
    monkeypatch.setattr(main.border, "outline_cutout", lambda raw, color, width: b"png")
    monkeypatch.setattr(main.border, "ai_outline_border", lambda *args: b"png")
    accounts = FakeAccounts(caps={"image_generate": 1, "image_edit": 1})
    accounts.install(monkeypatch)
    return TestClient(main.app), generate, accounts


def generate(http, headers):
    return http.post("/generate", json={"command": "sticker", "prompt": "a cat"}, headers=headers)


def border(http, headers, mode):
    return http.post("/border", params={"mode": mode, "prompt": "a thin white line"}, files={"file": ("a.png", b"img")}, headers=headers)


def test_a_valid_user_token_is_served_and_charged(svc):
    http, gen, accounts = svc
    assert generate(http, accounts.headers()).status_code == 200
    assert accounts.statuses("image_generate") == ["ok"]


def test_an_expired_token_is_401(svc):
    http, gen, accounts = svc
    r = generate(http, accounts.headers(expires_in=-5))
    assert r.status_code == 401 and r.json()["reason"] == "token_expired"
    assert gen.calls == 0


def test_over_the_cap_is_429_without_a_model_call(svc):
    http, gen, accounts = svc
    assert generate(http, accounts.headers()).status_code == 200
    r = generate(http, accounts.headers())
    assert r.status_code == 429
    assert r.json()["reason"] == "quota" and r.json()["feature"] == "image_generate" and r.json()["resets_at"]
    assert gen.calls == 1


def test_a_failed_provider_call_is_not_charged(svc):
    http, gen, accounts = svc
    gen.fail = 502
    assert generate(http, accounts.headers()).status_code == 502
    assert accounts.statuses("image_generate") == ["failed"]
    gen.fail = 0
    assert generate(http, accounts.headers()).status_code == 200


def test_a_generate_past_its_deadline_is_not_charged(svc):
    """The 50 s bound answers 504 inside the route; the gate settles it failed."""
    http, gen, accounts = svc
    gen.fail = 504
    assert generate(http, accounts.headers()).status_code == 504
    assert accounts.statuses("image_generate") == ["failed"]


def test_ai_border_is_an_edit_and_outline_is_free(svc):
    http, gen, accounts = svc
    assert border(http, accounts.headers(), "outline").status_code == 200
    assert accounts.rows == {}
    assert border(http, accounts.headers(), "ai_outline").status_code == 200
    assert accounts.statuses("image_edit") == ["ok"]
    r = border(http, accounts.headers(), "ai_outline")
    assert r.status_code == 429 and r.json()["feature"] == "image_edit"


def test_stylize_is_an_edit_and_a_rejected_image_is_not_charged(svc):
    http, gen, accounts = svc
    r = http.post("/stylize", files={"file": ("a.png", b"not an image")}, headers=accounts.headers())
    assert 400 <= r.status_code < 500
    assert accounts.statuses("image_edit") == ["failed"]


def test_accounts_down_is_503(svc):
    http, gen, accounts = svc
    accounts.reachable = False
    r = generate(http, accounts.headers())
    assert r.status_code == 503 and r.json()["reason"] == "accounts_unavailable"
    assert gen.calls == 0


def test_beta_flag_off_rejects_the_beta_token(svc, monkeypatch):
    http, gen, _ = svc
    assert generate(http, BETA).status_code == 200
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    r = generate(http, BETA)
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"
    assert gen.calls == 1

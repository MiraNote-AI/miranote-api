"""Signed-in requests: our access token (api #95 step 3). Transcription has
no cap in design 2, so nothing here is metered: no reserve, no 429 quota.
The gate itself is tested in tests/test_ai_auth.py at the repo root.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from fake_accounts import FakeAccounts

BETA = {"Authorization": "Bearer test-token"}


@pytest.fixture
def svc(voice_client, monkeypatch):
    authenticated, _, _ = voice_client
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    accounts = FakeAccounts()
    accounts.install(monkeypatch)
    return TestClient(authenticated.app), accounts


def emotion(http, headers):
    files = {"file": ("clip.wav", io.BytesIO(b"\x00\x01" * 2048), "audio/wav")}
    return http.post("/emotion", files=files, headers=headers)


def test_a_valid_user_token_is_served_unmetered(svc):
    http, accounts = svc
    assert emotion(http, accounts.headers()).status_code == 200
    assert accounts.rows == {}


def test_an_expired_token_is_401(svc):
    http, accounts = svc
    r = emotion(http, accounts.headers(expires_in=-5))
    assert r.status_code == 401 and r.json()["reason"] == "token_expired"


def test_the_catch_all_mount_takes_a_user_token(svc):
    http, accounts = svc
    assert http.get("/anything-not-a-route", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert http.get("/anything-not-a-route", headers=accounts.headers()).status_code != 401


def test_beta_flag_off_rejects_the_beta_token(svc, monkeypatch):
    http, _ = svc
    assert emotion(http, BETA).status_code == 200
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    r = emotion(http, BETA)
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"

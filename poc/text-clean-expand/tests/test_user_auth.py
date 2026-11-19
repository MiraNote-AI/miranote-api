"""Signed-in requests: our access token, metered per feature (api #95 step 3).

The gate itself is tested in tests/test_ai_auth.py at the repo root; these
prove this service is wired to it with the right features.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fake_accounts import FakeAccounts

BETA = {"Authorization": "Bearer test-token"}


@pytest.fixture
def svc(client, monkeypatch):
    beta_client, fake_llm = client
    accounts = FakeAccounts(caps={"expand": 1, "mood": 1})
    accounts.install(monkeypatch)
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    return TestClient(beta_client.app), fake_llm, accounts


def test_a_valid_user_token_is_served_and_charged(svc):
    http, fake_llm, accounts = svc
    fake_llm.reply_with("Polished.")
    r = http.post("/polish", json={"text": "polish me"}, headers=accounts.headers())
    assert r.status_code == 200 and r.json()["polished"] == "Polished."
    assert accounts.statuses("expand") == ["ok"]


def test_an_expired_token_is_401_and_never_reaches_the_model(svc):
    http, fake_llm, accounts = svc
    r = http.post("/polish", json={"text": "x"}, headers=accounts.headers(expires_in=-5))
    assert r.status_code == 401 and r.json()["reason"] == "token_expired"
    assert fake_llm.chat.completions.calls == []


def test_expand_and_polish_share_one_cap_and_over_it_is_429(svc):
    http, fake_llm, accounts = svc
    fake_llm.reply_with("Polished.")
    assert http.post("/polish", json={"text": "x"}, headers=accounts.headers()).status_code == 200
    r = http.post("/expand", json={"text": "x"}, headers=accounts.headers())
    assert r.status_code == 429
    assert r.json()["reason"] == "quota" and r.json()["feature"] == "expand" and r.json()["resets_at"]
    assert len(fake_llm.chat.completions.calls) == 1


def test_rewrite_is_metered_on_the_expand_cap(svc):
    """/rewrite spends the same free cap as /expand and /polish, so the app's
    quota 429 (the paywall) covers it too."""
    http, fake_llm, accounts = svc
    body = {"text": "x", "instruction": "warmer"}
    fake_llm.reply_with("Warmer.")
    assert http.post("/rewrite", json=body, headers=accounts.headers()).status_code == 200
    assert accounts.statuses("expand") == ["ok"]
    r = http.post("/rewrite", json=body, headers=accounts.headers())
    assert r.status_code == 429
    assert r.json()["reason"] == "quota" and r.json()["feature"] == "expand"
    assert len(fake_llm.chat.completions.calls) == 1


def test_rewrite_without_a_token_is_401(svc):
    http, fake_llm, _ = svc
    r = http.post("/rewrite", json={"text": "x", "instruction": "warmer"})
    assert r.status_code == 401 and r.json()["reason"] == "missing_token"
    assert fake_llm.chat.completions.calls == []


def test_a_failed_provider_call_is_not_charged(svc):
    http, fake_llm, accounts = svc
    # Nothing scripted: the fake model raises, the route answers 502.
    assert http.post("/polish", json={"text": "x"}, headers=accounts.headers()).status_code == 502
    assert accounts.statuses("expand") == ["failed"]
    fake_llm.reply_with("Polished.")
    assert http.post("/polish", json={"text": "x"}, headers=accounts.headers()).status_code == 200


def test_unmetered_routes_take_a_user_token(svc):
    http, fake_llm, accounts = svc
    fake_llm.reply_with("Clean.")
    assert http.post("/clean", json={"text": "x"}, headers=accounts.headers()).status_code == 200
    assert accounts.rows == {}


def test_accounts_down_is_503_for_a_metered_route(svc):
    http, fake_llm, accounts = svc
    accounts.reachable = False
    r = http.post("/polish", json={"text": "x"}, headers=accounts.headers())
    assert r.status_code == 503 and r.json()["reason"] == "accounts_unavailable"
    assert fake_llm.chat.completions.calls == []


def test_beta_flag_off_rejects_the_beta_token(svc, monkeypatch):
    http, fake_llm, _ = svc
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    r = http.post("/polish", json={"text": "x"}, headers=BETA)
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"
    assert fake_llm.chat.completions.calls == []


def test_mood_keeps_the_beta_token_with_the_flag_off(svc, monkeypatch):
    """Until App Attest (design 6.5): signed-out /mood rides the beta token."""
    http, fake_llm, accounts = svc
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    fake_llm.reply_with("calm")
    assert http.post("/mood", json={"text": "a page"}, headers=BETA).status_code == 200
    assert accounts.rows == {}


def test_signed_in_mood_is_metered(svc):
    http, fake_llm, accounts = svc
    fake_llm.reply_with("calm")
    assert http.post("/mood", json={"text": "a page"}, headers=accounts.headers()).status_code == 200
    r = http.post("/mood", json={"text": "a page"}, headers=accounts.headers())
    assert r.status_code == 429 and r.json()["feature"] == "mood"

"""Signed-in requests: our access token, /chat metered as `chat` (api #95
step 3). The gate itself is tested in tests/test_ai_auth.py at the repo root.

run_turn is replaced, so no test here reaches a model.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BETA_TOKENS", "test-token")

from fake_accounts import FakeAccounts  # noqa: E402
from poc.chatbot import main  # noqa: E402
from poc.chatbot.chat_loop import ChatTurnResult  # noqa: E402

BETA = {"Authorization": "Bearer test-token"}


class FakeTurn:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return ChatTurnResult(session_id="s1", reply="Hi there.")


@pytest.fixture
def svc(monkeypatch):
    import beta_auth

    beta_auth._limiter._hits.clear()
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    turn = FakeTurn()
    monkeypatch.setattr(main, "run_turn", turn)
    accounts = FakeAccounts(caps={"chat": 1})
    accounts.install(monkeypatch)
    return TestClient(main.app), turn, accounts


def chat(http, headers):
    return http.post("/chat", json={"message": "hi"}, headers=headers)


def test_a_valid_user_token_is_served_and_charged(svc):
    http, turn, accounts = svc
    r = chat(http, accounts.headers())
    assert r.status_code == 200 and r.json()["reply"] == "Hi there."
    assert accounts.statuses("chat") == ["ok"]


def test_an_expired_token_is_401(svc):
    http, turn, accounts = svc
    r = chat(http, accounts.headers(expires_in=-5))
    assert r.status_code == 401 and r.json()["reason"] == "token_expired"
    assert turn.calls == 0


def test_over_the_cap_is_429_without_a_model_call(svc):
    http, turn, accounts = svc
    assert chat(http, accounts.headers()).status_code == 200
    r = chat(http, accounts.headers())
    assert r.status_code == 429
    assert r.json()["reason"] == "quota" and r.json()["feature"] == "chat" and r.json()["resets_at"]
    assert turn.calls == 1


def test_a_failed_provider_call_is_not_charged(svc):
    http, turn, accounts = svc
    turn.fail = True
    assert chat(http, accounts.headers()).status_code == 502
    assert accounts.statuses("chat") == ["failed"]
    turn.fail = False
    assert chat(http, accounts.headers()).status_code == 200


def test_accounts_down_is_503(svc):
    http, turn, accounts = svc
    accounts.reachable = False
    r = chat(http, accounts.headers())
    assert r.status_code == 503 and r.json()["reason"] == "accounts_unavailable"
    assert turn.calls == 0


def test_the_beta_token_still_works_by_default_unmetered(svc):
    http, turn, accounts = svc
    assert chat(http, BETA).status_code == 200
    assert accounts.rows == {}


def test_beta_flag_off_rejects_the_beta_token(svc, monkeypatch):
    http, turn, _ = svc
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    r = chat(http, BETA)
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"
    assert turn.calls == 0

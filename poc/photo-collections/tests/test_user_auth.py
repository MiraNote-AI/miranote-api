"""Signed-in requests: our access token, compose metered as `photo_push`
(api #95 step 3). Compose answers 202 and runs the model in a job, so the
job settles: charged when it produced a push, not when it failed or was
rejected. The gate itself is tested in tests/test_ai_auth.py at the repo
root.
"""

from __future__ import annotations

import json

import pytest

from fake_accounts import FakeAccounts
from tests.test_api import _files, _poll
from tests.test_compose import _a_json, _b_json, _meta

BETA = {"Authorization": "Bearer test-token"}


@pytest.fixture
def accounts(monkeypatch):
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    fake = FakeAccounts(caps={"photo_push": 1})
    fake.install(monkeypatch)
    return fake


def submit(client, headers):
    return client.post("/collections/compose", data={"meta": json.dumps(_meta())}, files=_files(),
                       headers=headers)


def test_a_valid_user_token_composes_and_is_charged(client, fake_model, accounts):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    r = submit(client, accounts.headers())
    assert r.status_code == 202
    assert _poll(client, r.json()["job_id"]).json()["status"] == "ok"
    assert accounts.statuses("photo_push") == ["ok"]


def test_an_expired_token_is_401(client, fake_model, accounts):
    r = submit(client, accounts.headers(expires_in=-5))
    assert r.status_code == 401 and r.json()["reason"] == "token_expired"
    assert fake_model.calls == []


def test_over_the_cap_is_429_without_a_model_call(client, fake_model, accounts):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    _poll(client, submit(client, accounts.headers()).json()["job_id"])
    calls = len(fake_model.calls)
    r = submit(client, accounts.headers())
    assert r.status_code == 429
    assert r.json()["reason"] == "quota" and r.json()["feature"] == "photo_push" and r.json()["resets_at"]
    assert len(fake_model.calls) == calls


def test_a_failed_job_is_not_charged(client, fake_model, accounts):
    fake_model.fail("a", RuntimeError("provider down"))
    job_id = submit(client, accounts.headers()).json()["job_id"]
    assert _poll(client, job_id).json()["status"] == "error"
    assert accounts.statuses("photo_push") == ["failed"]


def test_a_rejected_job_is_not_charged(client, fake_model, accounts):
    fake_model.reply("a", "x").reply("a", "y")
    job_id = submit(client, accounts.headers()).json()["job_id"]
    assert _poll(client, job_id).json()["status"] == "rejected"
    assert accounts.statuses("photo_push") == ["failed"]


def test_a_bad_request_is_not_charged(client, accounts):
    r = client.post("/collections/compose", data={"meta": "{}"}, headers=accounts.headers())
    assert r.status_code == 400
    assert accounts.statuses("photo_push") == ["failed"]


def test_polling_takes_a_user_token_unmetered(client, accounts):
    assert client.get("/collections/jobs/" + "0" * 32, headers=accounts.headers()).status_code == 404
    assert accounts.rows == {}


def test_accounts_down_is_503(client, fake_model, accounts):
    accounts.reachable = False
    r = submit(client, accounts.headers())
    assert r.status_code == 503 and r.json()["reason"] == "accounts_unavailable"
    assert fake_model.calls == []


def test_beta_flag_off_rejects_the_beta_token(client, fake_model, accounts, monkeypatch):
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    r = submit(client, BETA)
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"
    assert fake_model.calls == []

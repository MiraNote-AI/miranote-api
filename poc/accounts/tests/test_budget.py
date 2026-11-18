"""The daily cost budgets (design 6.3, budget.py), against a real Postgres.
The last test drives ai_auth against this service, so the 503 reaches the
app with its reason."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import uuid

import pytest
from fastapi.testclient import TestClient

import budget
import main
from tests.conftest import INTERNAL_SECRET, run_sql
from tests.test_usage import SECRET, _ai_service, new_user, reserve, settle

import ai_auth  # noqa: E402 -- on sys.path through tests.test_usage


@pytest.fixture
def start(clean_db, monkeypatch):
    """Start the service with these budget env vars (feature -> USD, "" for
    the global one); budgets are read at startup."""
    monkeypatch.setattr(budget, "_tripped", set())
    for name in [n for n in os.environ if n.startswith(budget.ENV_PREFIX)]:
        monkeypatch.delenv(name)
    clients = []

    def _start(**budgets: str) -> TestClient:
        for scope, amount in budgets.items():
            name = budget.ENV_PREFIX if scope == "total" else "{}_{}".format(budget.ENV_PREFIX, scope.upper())
            monkeypatch.setenv(name, amount)
        client = TestClient(main.app)
        client.__enter__()
        clients.append(client)
        return client

    yield _start
    for client in clients:
        client.__exit__(None, None, None)


def spend(feature, cost, status="ok", ago=dt.timedelta(0)):
    """One row of another (signed-out or deleted) caller: budgets count
    everyone."""
    run_sql(
        "INSERT INTO ai_usage (request_id, user_id, feature, model, units, environment, status, cost_usd, created_at) "
        "VALUES (gen_random_uuid(), NULL, :f, '', 1, 'Production', :s, :c, now() - CAST(:ago AS interval))",
        f=feature, s=status, c=cost, ago=ago,
    )


def test_no_budget_set_changes_nothing(start):
    client = start()
    spend("chat", "1000")
    assert main.app.state.budgets == {}
    assert reserve(client, new_user(), "chat").status_code == 200


def test_feature_budget_pauses_that_feature_only(start):
    client = start(chat="1.00")
    spend("chat", "0.60")
    user = new_user()
    assert reserve(client, user, "chat").status_code == 200
    spend("chat", "0.40")
    r = reserve(client, user, "chat")
    assert r.status_code == 503
    body = r.json()
    assert body["reason"] == "service_paused" and body["feature"] == "chat" and body["detail"]
    tomorrow = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
    assert body["resets_at"] == tomorrow.isoformat() + "T00:00:00+00:00"
    assert "budget_usd" not in body and "spent_usd" not in body
    assert reserve(client, user, "expand").status_code == 200
    # The refused call wrote nothing.
    assert run_sql("SELECT count(*) FROM ai_usage WHERE user_id = :u", u=user)[0][0] == 2


def test_global_budget_counts_every_feature(start):
    client = start(total="1.00")
    spend("chat", "0.50")
    spend("image_generate", "0.50")
    r = reserve(client, new_user(), "mood")
    assert r.status_code == 503 and r.json()["feature"] == "mood"


def test_zero_pauses_at_once(start):
    client = start(image_generate="0")
    assert reserve(client, new_user(), "image_generate").status_code == 503


def test_only_todays_ok_and_pending_rows_count(start):
    client = start(chat="1.00")
    spend("chat", "5", ago=dt.timedelta(hours=25))
    spend("chat", "5", status="failed")
    spend("chat", "0.99", status="pending")
    assert reserve(client, new_user(), "chat").status_code == 200
    spend("chat", "0.01", status="pending")
    assert reserve(client, new_user(), "chat").status_code == 503


def reserve_at(client, user, feature, cost):
    """A reserve as ai_auth sends it, with the call's estimated cost."""
    return client.post(
        "/usage/reserve",
        headers=SECRET,
        json={"user_id": str(user), "feature": feature, "request_id": str(uuid.uuid4()), "cost_usd": cost},
    )


def test_calls_in_flight_count_at_their_estimate(start):
    """Five expands of 0.001 fill a 0.005 budget before any of them has
    settled, and a failed one gives its share back."""
    client = start(expand="0.005")
    ids = [reserve_at(client, new_user(), "expand", "0.001").json()["request_id"] for _ in range(5)]
    assert reserve_at(client, new_user(), "expand", "0.001").status_code == 503
    settle(client, ids[0], "failed")
    assert reserve_at(client, new_user(), "expand", "0.001").status_code == 200


def test_ai_gate_reserves_at_its_cost_estimate(start, monkeypatch):
    """ai_auth sends COST_USD with reserve, so a call still running (or one
    whose settle never arrives) already counts against the budget."""
    client = start()
    monkeypatch.setattr(ai_auth, "TRANSPORT", client._transport)
    monkeypatch.setenv("ACCOUNTS_INTERNAL_SECRET", INTERNAL_SECRET)
    request_id = uuid.uuid4()
    assert ai_auth.reserve(new_user(), "image_generate", request_id, "").status_code == 200
    row = run_sql("SELECT status, cost_usd FROM ai_usage WHERE request_id = :r", r=request_id)[0]
    assert row.status == "pending"
    assert row.cost_usd == budget.Decimal(str(ai_auth.COST_USD["image_generate"]))


def test_a_tripped_budget_logs_once(start, monkeypatch):
    logged = []
    monkeypatch.setattr(budget.log, "warning", lambda msg, extra: logged.append((msg, extra)))
    client = start(chat="1", total="100")
    spend("chat", "1")
    for _ in range(3):
        assert reserve(client, new_user(), "chat").status_code == 503
    assert logged == [("daily budget reached", {"scope": "chat", "budget_usd": "1", "spent_usd": "1.000000"})]


@pytest.mark.parametrize("name, value", [
    ("ACCOUNTS_DAILY_BUDGET_USD_CAHT", "5"),
    ("ACCOUNTS_DAILY_BUDGET_USDCHAT", "5"),
    ("ACCOUNTS_DAILY_BUDGET_USD_CHAT", "five"),
    ("ACCOUNTS_DAILY_BUDGET_USD_CHAT", "-1"),
    ("ACCOUNTS_DAILY_BUDGET_USD", "NaN"),
])
def test_a_malformed_budget_stops_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        budget.from_env()


def test_startup_logs_the_budgets_in_force(start, monkeypatch):
    """So the log shows whether the switch is armed, e.g. when a budget was
    put in the repo root .env, which this service does not read."""
    logged = []
    monkeypatch.setattr(main.log, "info", lambda msg, extra=None: logged.append((msg, extra)))
    start(chat="1.5", total="20")
    assert [extra["budgets"] for msg, extra in logged if msg == "daily budgets"] == [{"chat": "1.5", "total": "20"}]


def test_an_empty_budget_is_unset(monkeypatch):
    monkeypatch.setenv("ACCOUNTS_DAILY_BUDGET_USD_CHAT", " ")
    monkeypatch.setenv("ACCOUNTS_DAILY_BUDGET_USD", "20")
    assert budget.from_env() == {"total": budget.Decimal("20")}


def test_cli_prints_todays_spend(clean_db, monkeypatch, capsys):
    monkeypatch.setenv("ACCOUNTS_DAILY_BUDGET_USD_CHAT", "5")
    spend("chat", "0.25")
    spend("chat", "0.25", status="pending")
    spend("mood", "0.5", ago=dt.timedelta(hours=25))
    assert asyncio.run(budget._main()) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("(UTC) spend so far")
    rows = {line.split()[0]: line.split()[1:] for line in lines[2:]}
    assert rows["chat"] == ["0.5000", "5"]
    assert rows["mood"] == ["0.0000", "-"]
    assert rows["total"] == ["0.5000", "-"]


def test_ai_gate_passes_service_paused_to_the_app(start, monkeypatch):
    client = start(expand="0")
    monkeypatch.setattr(ai_auth, "TRANSPORT", client._transport)
    monkeypatch.setattr(ai_auth, "JWKS", ai_auth._Jwks())
    monkeypatch.setenv("ACCOUNTS_INTERNAL_SECRET", INTERNAL_SECRET)
    user = new_user()
    token = main.app.state.signer.issue(user, dt.datetime.now(dt.timezone.utc))
    r = TestClient(_ai_service()).post(
        "/expand", headers={"Authorization": "Bearer " + token, "X-Request-ID": str(uuid.uuid4())}
    )
    assert r.status_code == 503
    assert r.json()["reason"] == "service_paused" and r.json()["feature"] == "expand"

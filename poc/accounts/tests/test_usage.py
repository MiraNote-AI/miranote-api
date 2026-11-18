"""Entitlement, the ai_usage ledger and the caps, against a real Postgres.
The last section drives ai_auth (the AI services' gate) against this real
service, so the two sides of the reserve/settle contract cannot drift."""

from __future__ import annotations

import datetime as dt
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import main
import quotas
from tests.conftest import INTERNAL_SECRET, run_sql

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ai_auth  # noqa: E402

SECRET = {"X-Internal-Secret": INTERNAL_SECRET}


@pytest.fixture
def client(clean_db):
    with TestClient(main.app) as c:
        yield c


def new_user(deleting: bool = False) -> uuid.UUID:
    return run_sql(
        "INSERT INTO users (apple_sub, apple_team_id, deletion_requested_at) "
        "VALUES (:sub, 'T', CASE WHEN :deleting THEN now() END) RETURNING id",
        sub=str(uuid.uuid4()),
        deleting=deleting,
    )[0][0]


def add_usage(user_id, feature, n=1, status="ok", ago=dt.timedelta(0)):
    run_sql(
        "INSERT INTO ai_usage (request_id, user_id, feature, model, units, environment, status, created_at) "
        "SELECT gen_random_uuid(), :u, :f, '', 1, 'Production', :s, now() - CAST(:ago AS interval) FROM generate_series(1, :n)",
        u=user_id, f=feature, s=status, ago=ago, n=n,
    )


def add_subscription(user_id, environment="Production", expires_in=dt.timedelta(days=10), grace_in=None, revoked=False):
    now = dt.datetime.now(dt.timezone.utc)
    run_sql(
        "INSERT INTO subscriptions (original_transaction_id, user_id, product_id, status, expires_at, "
        "grace_period_expires_at, auto_renew, revoked_at, environment, last_signed_date) "
        "VALUES (:otid, :u, 'app.miranote.plus.monthly', 'active', :exp, :grace, true, :rev, :env, now())",
        otid=str(uuid.uuid4()), u=user_id, exp=now + expires_in,
        grace=now + grace_in if grace_in is not None else None,
        rev=now if revoked else None, env=environment,
    )


def reserve(client, user_id, feature, request_id=None, headers=SECRET):
    return client.post(
        "/usage/reserve",
        headers=headers,
        json={"user_id": str(user_id), "feature": feature, "request_id": str(request_id or uuid.uuid4())},
    )


def settle(client, request_id, status, cost="0.04"):
    return client.post(
        "/usage/settle", headers=SECRET, json={"request_id": str(request_id), "status": status, "cost_usd": cost}
    )


def ledger(user_id):
    return run_sql("SELECT feature, status, cost_usd FROM ai_usage WHERE user_id = :u ORDER BY created_at", u=user_id)


# --- the internal routes are internal ------------------------------------------


def test_reserve_and_settle_need_the_internal_secret(client):
    user = new_user()
    assert reserve(client, user, "chat", headers={}).status_code == 403
    assert reserve(client, user, "chat", headers={"X-Internal-Secret": "wrong"}).status_code == 403
    assert client.post("/usage/settle", json={"request_id": str(uuid.uuid4()), "status": "ok"}).status_code == 403
    assert ledger(user) == []


def test_internal_routes_answer_404_through_the_tunnel(client):
    """cloudflared adds Cf-Connecting-IP; the right secret does not help."""
    r = reserve(client, new_user(), "chat", headers=dict(SECRET, **{"Cf-Connecting-IP": "203.0.113.9"}))
    assert r.status_code == 404


def test_internal_secret_is_required(monkeypatch):
    monkeypatch.delenv("ACCOUNTS_INTERNAL_SECRET")
    with pytest.raises(RuntimeError, match="ACCOUNTS_INTERNAL_SECRET is not set"):
        with TestClient(main.app):
            pass


# --- reserve -------------------------------------------------------------------


def test_free_chat_allows_five_turns_then_429_with_the_reset_time(client):
    user = new_user()
    for _ in range(5):
        assert reserve(client, user, "chat").status_code == 200
    r = reserve(client, user, "chat")
    assert r.status_code == 429
    body = r.json()
    assert body["reason"] == "quota" and body["feature"] == "chat" and body["limit"] == 5
    first = run_sql("SELECT min(created_at) FROM ai_usage WHERE user_id = :u", u=user)[0][0]
    assert dt.datetime.fromisoformat(body["resets_at"]) == first + dt.timedelta(hours=24)
    assert [row.status for row in ledger(user)] == ["pending"] * 5


def test_over_the_cap_resets_at_is_when_the_count_drops_below_it(client):
    """Plus ran out with 10 chats in the window (back to free caps, the
    #95 test script step 5): the next turn frees up when the fifth newest
    row leaves the window, not the oldest."""
    user = new_user()
    add_usage(user, "chat", n=3, ago=dt.timedelta(hours=10))
    add_usage(user, "chat", n=7, ago=dt.timedelta(hours=5))
    r = reserve(client, user, "chat")
    assert r.status_code == 429
    fifth_newest = run_sql(
        "SELECT created_at FROM ai_usage WHERE user_id = :u ORDER BY created_at DESC OFFSET 4 LIMIT 1", u=user
    )[0][0]
    assert dt.datetime.fromisoformat(r.json()["resets_at"]) == fifth_newest + dt.timedelta(hours=24)


def test_rolling_window_forgets_rows_older_than_24_hours(client):
    user = new_user()
    add_usage(user, "expand", n=10, ago=dt.timedelta(hours=25))
    add_usage(user, "expand", n=9, ago=dt.timedelta(hours=23))
    assert reserve(client, user, "expand").status_code == 200
    assert reserve(client, user, "expand").status_code == 429


def test_free_photo_push_is_one_per_rolling_week(client):
    user = new_user()
    add_usage(user, "photo_push", ago=dt.timedelta(days=6))
    assert reserve(client, user, "photo_push").status_code == 429
    run_sql("UPDATE ai_usage SET created_at = now() - interval '8 days'")
    assert reserve(client, user, "photo_push").status_code == 200


def test_free_images_five_generations_and_two_edits_per_utc_month(client):
    user = new_user()
    add_usage(user, "image_generate", n=5)
    add_usage(user, "image_edit", n=1)
    r = reserve(client, user, "image_generate")
    assert r.status_code == 429
    now = dt.datetime.now(dt.timezone.utc)
    resets = dt.datetime.fromisoformat(r.json()["resets_at"])
    assert resets > now and resets.day == 1 and resets.hour == 0 and resets.minute == 0
    # Edits have their own two on Free.
    assert reserve(client, user, "image_edit").status_code == 200
    assert reserve(client, user, "image_edit").status_code == 429


def test_last_months_images_do_not_count(client):
    user = new_user()
    now = dt.datetime.now(dt.timezone.utc)
    since_month_start = now - now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    add_usage(user, "image_generate", n=5, ago=since_month_start + dt.timedelta(minutes=1))
    assert reserve(client, user, "image_generate").status_code == 200


def test_failed_rows_are_not_charged(client):
    user = new_user()
    add_usage(user, "chat", n=5, status="failed")
    assert reserve(client, user, "chat").status_code == 200


def test_a_retry_of_a_counted_request_id_is_refused(client):
    """Pending or ok: the result is not stored, so a rerun would be free."""
    user = new_user()
    request_id = uuid.uuid4()
    assert reserve(client, user, "chat", request_id).status_code == 200
    r = reserve(client, user, "chat", request_id)
    assert r.status_code == 409 and r.json()["reason"] == "duplicate_request"
    settle(client, request_id, "ok")
    assert reserve(client, user, "chat", request_id).status_code == 409
    assert len(ledger(user)) == 1


def test_a_failed_request_retried_with_its_id_is_charged_again(client):
    user = new_user()
    request_id = uuid.uuid4()
    reserve(client, user, "chat", request_id)
    settle(client, request_id, "failed")
    assert reserve(client, user, "chat", request_id).status_code == 200
    assert [row.status for row in ledger(user)] == ["pending"]


def test_request_id_of_another_user_or_feature_is_a_conflict(client):
    request_id = uuid.uuid4()
    first = new_user()
    reserve(client, first, "chat", request_id)
    assert reserve(client, new_user(), "chat", request_id).status_code == 409
    assert reserve(client, first, "expand", request_id).status_code == 409


def test_unknown_or_deleting_user_is_401(client):
    assert reserve(client, uuid.uuid4(), "chat").status_code == 401
    assert reserve(client, new_user(deleting=True), "chat").status_code == 401


def test_unknown_feature_is_400(client):
    assert reserve(client, new_user(), "transcribe").status_code == 400


def test_parallel_reserves_cannot_overshoot_the_cap(client):
    """Nine rows used of ten; eight requests at once take exactly one."""
    user = new_user()
    add_usage(user, "expand", n=9)
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda _: reserve(client, user, "expand").status_code, range(8)))
    assert sorted(codes) == [200] + [429] * 7
    assert len(ledger(user)) == 10


# --- entitlement ---------------------------------------------------------------


def test_plus_gets_plus_caps_and_shares_images(client):
    user = new_user()
    add_subscription(user)
    add_usage(user, "chat", n=5)
    assert reserve(client, user, "chat").status_code == 200
    add_usage(user, "image_generate", n=60)
    add_usage(user, "image_edit", n=39)
    assert reserve(client, user, "image_edit").status_code == 200
    r = reserve(client, user, "image_generate")
    assert r.status_code == 429 and r.json()["plan"] == "plus" and r.json()["limit"] == 100


def test_sandbox_plus_has_its_own_lower_caps(client):
    user = new_user()
    add_subscription(user, environment="Sandbox")
    add_usage(user, "image_generate", n=9)
    assert reserve(client, user, "image_edit").status_code == 200
    r = reserve(client, user, "image_generate")
    assert r.status_code == 429 and r.json()["limit"] == 10
    add_usage(user, "chat", n=30)
    r = reserve(client, user, "chat")
    assert r.status_code == 429 and r.json()["plan"] == "plus" and r.json()["limit"] == 30
    assert reserve(client, user, "expand").status_code == 200
    assert run_sql("SELECT environment FROM ai_usage WHERE feature = 'expand'")[0][0] == "Sandbox"


def test_signed_in_mood_is_thirty_per_day(client):
    user = new_user()
    add_usage(user, "mood", n=29)
    assert reserve(client, user, "mood").status_code == 200
    assert reserve(client, user, "mood").status_code == 429


def test_grace_period_is_still_plus(client):
    user = new_user()
    add_subscription(user, expires_in=-dt.timedelta(days=1), grace_in=dt.timedelta(days=3))
    add_usage(user, "photo_push", n=1, ago=dt.timedelta(days=2))
    assert reserve(client, user, "photo_push").status_code == 200


@pytest.mark.parametrize(
    "kw",
    [
        {"expires_in": -dt.timedelta(minutes=1)},
        {"expires_in": -dt.timedelta(days=3), "grace_in": -dt.timedelta(days=1)},
        {"revoked": True},
    ],
    ids=["expired", "grace-over", "refunded"],
)
def test_expired_or_refunded_is_free(client, kw):
    user = new_user()
    add_subscription(user, **kw)
    add_usage(user, "chat", n=5)
    r = reserve(client, user, "chat")
    assert r.status_code == 429 and r.json()["plan"] == "free"


def test_another_users_subscription_does_not_count(client):
    user = new_user()
    add_subscription(new_user())
    add_usage(user, "chat", n=5)
    assert reserve(client, user, "chat").status_code == 429


def test_production_wins_over_sandbox(client):
    user = new_user()
    add_subscription(user, environment="Sandbox")
    add_subscription(user, environment="Production")
    add_usage(user, "image_generate", n=10)
    assert reserve(client, user, "image_generate").status_code == 200


# --- settle --------------------------------------------------------------------


def test_settle_records_cost_and_failed_is_free(client):
    user = new_user()
    ok_id, failed_id = uuid.uuid4(), uuid.uuid4()
    reserve(client, user, "image_generate", ok_id)
    reserve(client, user, "image_generate", failed_id)
    assert settle(client, ok_id, "ok").json() == {"status": "ok"}
    assert settle(client, failed_id, "failed").json() == {"status": "failed"}
    rows = run_sql("SELECT request_id, status, cost_usd FROM ai_usage")
    assert {(r.request_id, r.status, str(r.cost_usd)) for r in rows} == {
        (ok_id, "ok", "0.040000"),
        (failed_id, "failed", "0.000000"),
    }


def test_settle_only_changes_a_pending_row(client):
    user = new_user()
    request_id = uuid.uuid4()
    reserve(client, user, "chat", request_id)
    settle(client, request_id, "ok")
    assert settle(client, request_id, "failed").json() == {"status": "ok"}
    assert [r.status for r in ledger(user)] == ["ok"]


def test_settle_unknown_request_is_404(client):
    assert settle(client, uuid.uuid4(), "ok").status_code == 404


# --- /me/usage -----------------------------------------------------------------


def test_me_usage_reports_remaining_and_resets(client):
    user = new_user()
    add_usage(user, "chat", n=2, ago=dt.timedelta(hours=1))
    add_usage(user, "expand", n=3, status="failed")
    token = main.app.state.signer.issue(user, dt.datetime.now(dt.timezone.utc))
    r = client.get("/me/usage", headers={"Authorization": "Bearer " + token})
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "free" and body["environment"] == "Production"
    assert set(body["features"]) == quotas.METERED_FEATURES
    chat = body["features"]["chat"]
    assert (chat["limit"], chat["used"], chat["remaining"], chat["window"]) == (5, 2, 3, "rolling_24h")
    assert chat["resets_at"] is not None
    assert body["features"]["expand"]["used"] == 0 and body["features"]["expand"]["resets_at"] is None
    assert body["features"]["image_generate"]["window"] == "utc_month"


def test_me_usage_needs_a_token(client):
    assert client.get("/me/usage").status_code == 401


# --- ai_auth against this service ------------------------------------------------


def _ai_service():
    app = FastAPI()

    @app.post("/expand")
    async def expand(fail: bool = False):
        if fail:
            raise HTTPException(502, detail="provider failed")
        return {"expanded": "..."}

    ai_auth.install(app, metered={"/expand": ai_auth.Metered("expand", model="test-model")})
    return app


@pytest.fixture
def ai_client(client, monkeypatch):
    """An AI service whose gate talks to the real accounts service above."""
    monkeypatch.setattr(ai_auth, "TRANSPORT", client._transport)
    monkeypatch.setattr(ai_auth, "JWKS", ai_auth._Jwks())
    monkeypatch.setenv("ACCOUNTS_INTERNAL_SECRET", INTERNAL_SECRET)
    return TestClient(_ai_service())


def test_ai_gate_meters_against_the_real_ledger(client, ai_client):
    user = new_user()
    token = main.app.state.signer.issue(user, dt.datetime.now(dt.timezone.utc))
    auth = {"Authorization": "Bearer " + token}
    assert ai_client.post("/expand", params={"fail": True}, headers=auth).status_code == 502
    for _ in range(10):
        assert ai_client.post("/expand", headers=auth).status_code == 200
    r = ai_client.post("/expand", headers=auth)
    assert r.status_code == 429
    assert r.json()["reason"] == "quota" and r.json()["feature"] == "expand" and r.json()["resets_at"]
    statuses = [row.status for row in ledger(user)]
    assert statuses == ["failed"] + ["ok"] * 10
    assert run_sql("SELECT DISTINCT model FROM ai_usage")[0][0] == "test-model"


def test_ai_gate_refuses_a_deleted_user_with_a_valid_token(client, ai_client):
    user = new_user(deleting=True)
    token = main.app.state.signer.issue(user, dt.datetime.now(dt.timezone.utc))
    r = ai_client.post("/expand", headers={"Authorization": "Bearer " + token})
    assert r.status_code == 401 and r.json()["reason"] == "unknown_user"

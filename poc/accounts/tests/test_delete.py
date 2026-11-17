"""Account deletion (DELETE /me), Apple's revoke, and the purge and
retention jobs, against a real Postgres. Apple's keys, token and revoke
endpoints are faked through httpx.MockTransport: no network."""

from __future__ import annotations

import asyncio
import datetime as dt
import urllib.parse
import uuid

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from fastapi.testclient import TestClient
from sqlalchemy.pool import NullPool

import apple
import config
import db
import main
import purge
from tests.conftest import INTERNAL_SECRET, TEST_URL, new_ec_key_pem, run_sql
from tests.test_auth import BUNDLE_ID, FakeApple, me, refresh, sign_in
from tests.test_subscriptions import add_chain, api, configured, fake, row, sync  # noqa: F401 -- fixtures

SIWA_KEY_ID = "SIWAKEY123"
DAY = dt.timedelta(days=1)


class FakeAppleAuth(FakeApple):
    """Adds Apple's /auth/token and /auth/revoke; `token_status` and
    `revoke_status` set what they answer."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.token_status = 200
        self.revoke_status = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == apple.APPLE_KEYS_URL:
            return super().handler(request)
        form = dict(urllib.parse.parse_qsl(request.content.decode()))
        self.calls.append((url, form))
        if url == apple.APPLE_TOKEN_URL:
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": "apple-at", "refresh_token": "apple-rt", "id_token": "x"})
        assert url == apple.APPLE_REVOKE_URL
        return httpx.Response(self.revoke_status)


@pytest.fixture
def fake_apple():
    return FakeAppleAuth()


@pytest.fixture
def siwa_key(tmp_path, monkeypatch):
    pem = new_ec_key_pem()
    path = tmp_path / "AuthKey_SIWA.p8"
    path.write_text(pem)
    monkeypatch.setattr(config, "APPLE_SIWA_KEY_PATH", str(path))
    monkeypatch.setattr(config, "APPLE_SIWA_KEY_ID", SIWA_KEY_ID)
    monkeypatch.setattr(apple, "REVOKE_RETRY_SECONDS", 0)
    return pem


@pytest.fixture
def client(clean_db, configured, api, fake_apple, siwa_key, monkeypatch):  # noqa: F811
    monkeypatch.setattr(main, "APPLE_TRANSPORT", httpx.MockTransport(fake_apple.handler))
    with TestClient(main.app) as c:
        c.app.state.appstore.clients = {"Sandbox": api, "Production": api}
        yield c


def delete_me(client, access_token, code="fresh-code"):
    return client.request(
        "DELETE", "/me", headers={"Authorization": "Bearer " + access_token}, json={"authorization_code": code}
    )


def run_purge(now=None):
    async def _run():
        engine = db.create_engine(TEST_URL, poolclass=NullPool)
        try:
            return await purge.run_once(engine, now or dt.datetime.now(dt.timezone.utc))
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def add_usage(user_id, ago=dt.timedelta(hours=1)):
    run_sql(
        "INSERT INTO ai_usage (request_id, user_id, feature, model, units, environment, status, created_at) "
        "VALUES (:r, :u, 'chat', 'm', 1, 'Production', 'ok', now() - CAST(:ago AS interval))",
        r=uuid.uuid4(), u=user_id, ago=ago,
    )


# --- DELETE /me: design 7.1 -------------------------------------------------


def test_delete_revokes_at_apple_ends_every_sign_in_and_purge_keeps_only_pseudonymized_rows(
    client, fake_apple, siwa_key
):
    first = sign_in(client, fake_apple.token()).json()
    second = sign_in(client, fake_apple.token()).json()  # another device
    user = first["user_id"]
    add_chain("7001", user_id=user, linked_ago=40 * DAY)
    add_usage(user)

    r = delete_me(client, first["access_token"])
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": True, "apple_revoked": True}

    # The fresh code was exchanged, then the refresh token Apple gave back
    # was revoked, both with a client secret signed by our SIWA key.
    (token_url, exchange), (revoke_url, revoke) = fake_apple.calls
    assert token_url == apple.APPLE_TOKEN_URL and revoke_url == apple.APPLE_REVOKE_URL
    assert exchange["code"] == "fresh-code" and exchange["grant_type"] == "authorization_code"
    assert revoke["token"] == "apple-rt" and revoke["token_type_hint"] == "refresh_token"
    public = load_pem_private_key(siwa_key.encode(), password=None).public_key()
    for form in (exchange, revoke):
        assert form["client_id"] == BUNDLE_ID
        assert jwt.get_unverified_header(form["client_secret"])["kid"] == SIWA_KEY_ID
        claims = jwt.decode(form["client_secret"], public, algorithms=["ES256"], audience=apple.APPLE_ISSUER)
        assert claims["iss"] == config.APPLE_TEAM_ID and claims["sub"] == BUNDLE_ID

    # Cut off at once: access tokens, both devices' refresh tokens, the guard.
    assert me(client, first["access_token"]).status_code == 401
    assert me(client, second["access_token"]).status_code == 401
    assert refresh(client, first["refresh_token"]).status_code == 401
    assert refresh(client, second["refresh_token"]).status_code == 401
    assert run_sql("SELECT count(*) FROM sessions WHERE revoked_at IS NULL") == [(0,)]
    reserve = client.post(
        "/usage/reserve",
        headers={"X-Internal-Secret": INTERNAL_SECRET},
        json={"user_id": user, "feature": "chat", "request_id": str(uuid.uuid4())},
    )
    assert reserve.status_code == 401

    counts = run_purge()
    assert counts["users"] == 1
    assert run_sql("SELECT count(*) FROM users") == [(0,)]
    assert run_sql("SELECT count(*) FROM sessions") == [(0,)]
    assert run_sql("SELECT user_id FROM ai_usage") == [(None,)]
    assert row("7001").user_id is None
    # Idempotent: a second run finds nothing more to do.
    assert set(run_purge().values()) == {0}


def test_delete_without_a_siwa_key_still_deletes_and_answers_503(client, fake_apple, monkeypatch):
    monkeypatch.setattr(client.app.state.apple_revoker, "_key_pem", None)
    pair = sign_in(client, fake_apple.token()).json()
    r = delete_me(client, pair["access_token"])
    assert r.status_code == 503
    assert r.json()["deleted"] is True and r.json()["apple_revoked"] is False
    assert "not configured" in r.json()["detail"]
    assert fake_apple.calls == []
    assert run_sql("SELECT deletion_requested_at IS NOT NULL FROM users") == [(True,)]
    assert me(client, pair["access_token"]).status_code == 401


def test_a_refused_code_is_not_retried_and_the_account_is_still_deleted(client, fake_apple):
    pair = sign_in(client, fake_apple.token()).json()
    fake_apple.token_status = 400
    r = delete_me(client, pair["access_token"])
    assert r.status_code == 502 and r.json()["deleted"] is True
    assert len(fake_apple.calls) == 1
    assert me(client, pair["access_token"]).status_code == 401


def test_apple_errors_are_retried_then_given_up(client, fake_apple):
    pair = sign_in(client, fake_apple.token()).json()
    fake_apple.revoke_status = 503
    r = delete_me(client, pair["access_token"])
    assert r.status_code == 502 and r.json()["apple_revoked"] is False
    assert [url for url, _ in fake_apple.calls] == [apple.APPLE_TOKEN_URL] + [apple.APPLE_REVOKE_URL] * 3
    assert run_sql("SELECT deletion_requested_at IS NOT NULL FROM users") == [(True,)]


def test_delete_needs_the_code_and_a_valid_token(client, fake_apple):
    pair = sign_in(client, fake_apple.token()).json()
    r = client.request("DELETE", "/me", headers={"Authorization": "Bearer " + pair["access_token"]}, json={})
    assert r.status_code == 422
    assert client.request("DELETE", "/me", json={"authorization_code": "c"}).status_code == 401
    assert run_sql("SELECT deletion_requested_at FROM users") == [(None,)]
    assert fake_apple.calls == []


def test_signing_in_again_gives_a_new_account_that_restore_can_give_the_subscription(
    client, fake_apple, fake, api  # noqa: F811
):
    old = sign_in(client, fake_apple.token()).json()
    # Bought in the app while signed in: Apple's transactions carry the old id.
    add_chain("7002", user_id=old["user_id"], linked_ago=40 * DAY)
    assert delete_me(client, old["access_token"]).status_code == 200

    new = sign_in(client, fake_apple.token()).json()
    assert new["user_id"] != old["user_id"]
    assert run_sql("SELECT count(*) FROM users") == [(1,)]
    assert row("7002").user_id is None

    new_id = uuid.UUID(new["user_id"])
    r = sync(client, new_id, fake, "7002", token=uuid.UUID(old["user_id"]))
    assert r.status_code == 200, r.text
    assert r.json()["plan"] == "plus"
    assert row("7002").user_id == new_id
    assert api.tokens == [("7002", str(new_id))]


# --- Retention: design 7.2 ---------------------------------------------------


def test_retention_deletes_only_rows_past_their_period(clean_db):
    live = run_sql("INSERT INTO users (apple_sub, apple_team_id) VALUES ('s', 'T') RETURNING id")[0][0]
    run_sql(
        "INSERT INTO sessions (user_id, family_id, token_hash, expires_at) VALUES "
        "(:u, gen_random_uuid(), 'expired', now() - interval '1 second'), "
        "(:u, gen_random_uuid(), 'valid', now() + interval '1 day')",
        u=live,
    )
    run_sql(
        "INSERT INTO subscriptions (original_transaction_id, user_id, product_id, status, expires_at, "
        "grace_period_expires_at, auto_renew, environment, last_signed_date) VALUES "
        "('old', NULL, 'p', 'expired', now() - interval '14 months', NULL, false, 'Production', now()), "
        "('owned-old', :u, 'p', 'expired', now() - interval '14 months', NULL, false, 'Production', now()), "
        "('in-grace', NULL, 'p', 'expired', now() - interval '14 months', now() - interval '12 months', "
        "false, 'Production', now()), "
        "('recent', NULL, 'p', 'expired', now() - interval '12 months', NULL, false, 'Production', now())",
        u=live,
    )
    run_sql(
        "INSERT INTO app_store_notifications (notification_uuid, type, signed_date, payload, processed_at) VALUES "
        "('old', 'TEST', now(), '{}', now() - interval '13 months'), "
        "('recent', 'TEST', now(), '{}', now() - interval '11 months')"
    )
    add_usage(live, ago=430 * DAY)
    add_usage(None, ago=430 * DAY)
    add_usage(live, ago=360 * DAY)
    run_sql(
        "INSERT INTO app_attest_keys (key_id, public_key, receipt, sign_count, environment, last_used_at) VALUES "
        "('old', '\\x00', '\\x00', 0, 'production', now() - interval '91 days'), "
        "('recent', '\\x00', '\\x00', 0, 'production', now() - interval '89 days')"
    )
    for ago in ("18 months 1 day", "17 months"):
        run_sql(
            "INSERT INTO taste_events (event_id, user_id, session_id, occurred_at, app_version, build, "
            "tz_offset_minutes, event_version, name, surface, item_type, item_id, action) "
            "VALUES (gen_random_uuid(), :u, gen_random_uuid(), now() - CAST(CAST(:ago AS text) AS interval), '1.0', '1', "
            "0, 1, 'font_choice', 'font_picker', 'font', 'serif-02', 'shown')",
            u=live, ago=ago,
        )

    assert run_purge() == {
        "users": 0,
        "sessions": 1,
        "subscriptions": 2,
        "app_store_notifications": 1,
        "ai_usage": 2,
        "app_attest_keys": 1,
        "taste_events": 1,
    }
    assert run_sql("SELECT token_hash FROM sessions") == [("valid",)]
    assert sorted(r[0] for r in run_sql("SELECT original_transaction_id FROM subscriptions")) == ["in-grace", "recent"]
    assert run_sql("SELECT notification_uuid FROM app_store_notifications") == [("recent",)]
    assert run_sql("SELECT count(*) FROM ai_usage") == [(1,)]
    assert run_sql("SELECT key_id FROM app_attest_keys") == [("recent",)]
    assert run_sql("SELECT count(*) FROM taste_events") == [(1,)]
    assert run_sql("SELECT count(*) FROM users") == [(1,)]

"""The constraints later steps rely on, checked against real Postgres."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from tests.conftest import run_sql

TEAM = "TEAMID0001"


def _user(sub="apple-sub-1", team=TEAM):
    return run_sql(
        "INSERT INTO users (apple_sub, apple_team_id) VALUES (:sub, :team) RETURNING id",
        sub=sub, team=team,
    )[0][0]


def _usage(request_id, user_id=None, feature="chat", device_key_id=None):
    return run_sql(
        "INSERT INTO ai_usage (request_id, user_id, device_key_id, feature, model, units, "
        "environment, status) VALUES (:rid, :uid, :dk, :feature, 'm', 1, 'Sandbox', 'pending') "
        "ON CONFLICT (request_id) DO NOTHING RETURNING request_id",
        rid=request_id, uid=user_id, dk=device_key_id, feature=feature,
    )


def _subscription(otid, user_id, environment="Sandbox", status="active"):
    run_sql(
        "INSERT INTO subscriptions (original_transaction_id, user_id, product_id, status, "
        "expires_at, auto_renew, environment, last_signed_date) VALUES (:otid, :uid, "
        "'app.miranote.plus.monthly', :status, now(), true, :env, now())",
        otid=otid, uid=user_id, env=environment, status=status,
    )


def test_users_get_a_generated_uuid(clean_db):
    assert isinstance(_user(), uuid.UUID)


def test_same_apple_sub_in_the_same_team_is_rejected(clean_db):
    _user()
    with pytest.raises(IntegrityError, match="uq_users_apple_team_sub"):
        _user()


def test_same_apple_sub_in_another_team_is_a_different_user(clean_db):
    # After a move to a company developer account (design 3.4).
    assert _user(team=TEAM) != _user(team="TEAMID0002")


def test_ai_usage_retry_with_the_same_request_id_is_not_charged_twice(clean_db):
    user = _user()
    rid = uuid.uuid4()
    assert _usage(rid, user) == [(rid,)]
    assert _usage(rid, user) == []
    assert run_sql("SELECT count(*) FROM ai_usage") == [(1,)]


def test_ai_usage_plain_insert_of_a_seen_request_id_is_rejected(clean_db):
    user = _user()
    rid = uuid.uuid4()
    sql = (
        "INSERT INTO ai_usage (request_id, user_id, feature, model, units, environment, status) "
        "VALUES (:rid, :uid, 'chat', 'm', 1, 'Sandbox', 'ok')"
    )
    run_sql(sql, rid=rid, uid=user)
    with pytest.raises(IntegrityError, match="ai_usage_pkey"):
        run_sql(sql, rid=rid, uid=user)


def test_ai_usage_device_key_only_on_signed_out_rows(clean_db):
    user = _user()
    assert _usage(uuid.uuid4(), None, feature="mood", device_key_id="key-1")
    with pytest.raises(IntegrityError, match="ck_ai_usage_device_key_signed_out_only"):
        _usage(uuid.uuid4(), user, device_key_id="key-1")
    with pytest.raises(IntegrityError, match="ck_ai_usage_feature"):
        _usage(uuid.uuid4(), user, feature="telepathy")


def test_sessions_token_hash_is_unique(clean_db):
    user = _user()
    sql = (
        "INSERT INTO sessions (user_id, family_id, token_hash, expires_at) "
        "VALUES (:uid, gen_random_uuid(), 'hash-1', now() + interval '30 days')"
    )
    run_sql(sql, uid=user)
    with pytest.raises(IntegrityError, match="sessions_token_hash_key"):
        run_sql(sql, uid=user)


def test_seen_notification_uuid_is_rejected(clean_db):
    sql = (
        "INSERT INTO app_store_notifications (notification_uuid, type, signed_date, payload) "
        "VALUES ('n-1', 'DID_RENEW', now(), '{}'::jsonb)"
    )
    run_sql(sql)
    with pytest.raises(IntegrityError, match="app_store_notifications_pkey"):
        run_sql(sql)


def test_subscription_environment_and_status_are_checked(clean_db):
    user = _user()
    with pytest.raises(IntegrityError, match="ck_subscriptions_environment"):
        _subscription("otid-1", user, environment="Xcode")
    with pytest.raises(IntegrityError, match="ck_subscriptions_status"):
        _subscription("otid-1", user, status="lifetime")


def test_deleting_a_user_drops_sessions_and_pseudonymizes_the_rest(clean_db):
    # The purge (design 7.2): sessions go, subscriptions and ai_usage stay
    # with user_id set to null.
    user = _user()
    run_sql(
        "INSERT INTO sessions (user_id, family_id, token_hash, expires_at) "
        "VALUES (:uid, gen_random_uuid(), 'hash-1', now())",
        uid=user,
    )
    _subscription("otid-1", user)
    _usage(uuid.uuid4(), user, feature="image_generate")
    run_sql("DELETE FROM users WHERE id = :uid", uid=user)

    assert run_sql("SELECT count(*) FROM sessions") == [(0,)]
    assert run_sql("SELECT user_id FROM subscriptions") == [(None,)]
    assert run_sql("SELECT user_id FROM ai_usage") == [(None,)]

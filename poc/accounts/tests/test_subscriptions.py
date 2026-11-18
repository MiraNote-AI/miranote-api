"""Subscription sync, App Store Server Notifications and the backfill,
against a real Postgres. Apple is faked by tests/fake_appstore.py: a local
root CA the verifier is configured to trust, so the real
app-store-server-library checks run, offline. The Server API client is a
recording fake."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import jwt
import pytest
from appstoreserverlibrary.models.LastTransactionsItem import LastTransactionsItem
from appstoreserverlibrary.models.StatusResponse import StatusResponse
from appstoreserverlibrary.models.SubscriptionGroupIdentifierItem import SubscriptionGroupIdentifierItem
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from sqlalchemy.pool import NullPool

import appstore
import backfill
import db
import main
import subscriptions
from tests.conftest import INTERNAL_SECRET, TEST_URL, run_sql
from tests.fake_appstore import APP_APPLE_ID, FakeAppStore

DAY = dt.timedelta(days=1)


class FakeServerApi:
    """Records Set App Account Token; serves Get All Subscription Statuses."""

    def __init__(self):
        self.tokens = []
        self.statuses = {}

    async def set_app_account_token(self, otid, request):
        self.tokens.append((otid, request.appAccountToken))

    async def get_all_subscription_statuses(self, otid):
        return self.statuses[otid]

    async def async_close(self):
        pass


@pytest.fixture(scope="module")
def fake():
    return FakeAppStore()


@pytest.fixture
def configured(fake, tmp_path, monkeypatch):
    root = tmp_path / "root.cer"
    root.write_bytes(fake.root_der())
    monkeypatch.setenv("APPSTORE_ROOT_CERTS", str(root))
    monkeypatch.setenv("APPSTORE_APP_APPLE_ID", str(APP_APPLE_ID))
    monkeypatch.setenv("APPSTORE_ONLINE_CHECKS", "0")


@pytest.fixture
def api():
    return FakeServerApi()


@pytest.fixture
def client(clean_db, configured, api):
    with TestClient(main.app) as c:
        c.app.state.appstore.clients = {"Sandbox": api, "Production": api}
        yield c


def new_user(deleting: bool = False) -> uuid.UUID:
    return run_sql(
        "INSERT INTO users (apple_sub, apple_team_id, deletion_requested_at) "
        "VALUES (:sub, 'T', CASE WHEN :deleting THEN now() END) RETURNING id",
        sub=str(uuid.uuid4()),
        deleting=deleting,
    )[0][0]


def bearer(client, user_id):
    token = client.app.state.signer.issue(user_id, dt.datetime.now(dt.timezone.utc))
    return {"Authorization": "Bearer " + token}


def sync(client, user_id, fake, otid, token=None, tx=None, renewal=None, **tx_kwargs):
    return client.post(
        "/subscriptions/sync",
        headers=bearer(client, user_id),
        json={
            "signed_transaction": tx or fake.transaction(otid, token=token, **tx_kwargs),
            "signed_renewal_info": renewal or fake.renewal(otid, environment=tx_kwargs.get("environment", "Sandbox")),
        },
    )


def notify(client, signed_payload):
    return client.post("/appstore/notifications", json={"signedPayload": signed_payload})


def row(otid):
    rows = run_sql(
        "SELECT user_id, status, auto_renew, auto_renew_product_id, revoked_at, environment, linked_at, "
        "grace_period_expires_at FROM subscriptions WHERE original_transaction_id = :o",
        o=otid,
    )
    return rows[0] if rows else None


def add_chain(otid, user_id=None, linked_ago=None, signed_ago=dt.timedelta(hours=1)):
    run_sql(
        "INSERT INTO subscriptions (original_transaction_id, user_id, linked_at, product_id, status, expires_at, "
        "auto_renew, environment, last_signed_date) VALUES (:o, :u, now() - CAST(:linked AS interval), "
        "'ai.miranote.app.plus.monthly', 'active', now() + interval '10 days', true, 'Sandbox', "
        "now() - CAST(:signed AS interval))",
        o=otid, u=user_id, linked=linked_ago, signed=signed_ago,
    )


# --- /subscriptions/sync: design 5.3 ------------------------------------


def test_sync_own_purchase_gives_plus_to_the_quota_guard(client, fake, api):
    user = new_user()
    r = sync(client, user, fake, "1000", token=user)
    assert r.status_code == 200, r.text
    assert r.json()["plan"] == "plus" and r.json()["environment"] == "Sandbox"
    assert r.json()["status"] == "active"
    assert row("1000").user_id == user and row("1000").linked_at is not None
    assert api.tokens == []  # the token already names the caller
    # The step-3 guard reads entitlement from the row: Sandbox Plus caps.
    reserve = client.post(
        "/usage/reserve",
        headers={"X-Internal-Secret": INTERNAL_SECRET},
        json={"user_id": str(user), "feature": "image_generate", "request_id": str(uuid.uuid4())},
    )
    assert reserve.json()["plan"] == "plus" and reserve.json()["remaining"] == 9


def test_sync_production_chain(client, fake):
    user = new_user()
    otid = "1001"
    r = sync(client, user, fake, otid, token=user, environment="Production")
    assert r.status_code == 200, r.text
    assert r.json()["environment"] == "Production" and row(otid).environment == "Production"


def test_sync_needs_sign_in(client, fake):
    r = client.post(
        "/subscriptions/sync",
        json={"signed_transaction": fake.transaction("1"), "signed_renewal_info": fake.renewal("1")},
    )
    assert r.status_code == 401


def test_sync_of_another_live_accounts_purchase_is_refused(client, fake):
    owner, caller = new_user(), new_user()
    r = sync(client, caller, fake, "1002", token=owner)
    assert r.status_code == 403
    assert row("1002") is None  # nothing stored


@pytest.mark.parametrize("token", ["absent", "deleted_user", "unknown_user"])
def test_sync_reclaims_an_unowned_chain_and_sets_the_token(client, fake, api, token):
    user = new_user()
    otid = "2000"
    if token == "absent":
        app_token = None  # offer code redeemed in the App Store
    elif token == "deleted_user":
        app_token = new_user(deleting=True)
    else:
        app_token = uuid.uuid4()  # its account was purged
    r = sync(client, user, fake, otid, token=app_token)
    assert r.status_code == 200, r.text
    assert r.json()["plan"] == "plus"
    assert row(otid).user_id == user
    assert api.tokens == [(otid, str(user))]


def test_sync_of_a_chain_the_caller_owns_changes_nothing(client, fake, api):
    user = new_user()
    add_chain("2001", user_id=user, linked_ago=dt.timedelta(days=2))
    linked_at = row("2001").linked_at
    r = sync(client, user, fake, "2001", token=None)
    assert r.status_code == 200
    assert row("2001").user_id == user and row("2001").linked_at == linked_at
    assert api.tokens == []


def test_sync_of_a_chain_owned_by_another_account_conflicts(client, fake):
    owner, caller = new_user(), new_user()
    add_chain("2002", user_id=owner, linked_ago=dt.timedelta(days=40))
    r = sync(client, caller, fake, "2002", token=None)
    assert r.status_code == 409
    assert row("2002").user_id == owner


def test_reclaim_waits_30_days_after_the_last_link(client, fake):
    user = new_user()
    add_chain("2003", user_id=None, linked_ago=dt.timedelta(days=29))
    assert sync(client, user, fake, "2003").status_code == 429
    assert row("2003").user_id is None
    add_chain("2004", user_id=None, linked_ago=dt.timedelta(days=31))
    assert sync(client, user, fake, "2004").status_code == 200
    assert row("2004").user_id == user


def test_one_user_links_at_most_3_chains_per_day(client, fake):
    user = new_user()
    for otid in ("3001", "3002", "3003"):
        assert sync(client, user, fake, otid).status_code == 200
    assert sync(client, user, fake, "3004").status_code == 429
    assert row("3004") is None
    # A purchase that carries the caller's own token is not a reclaim.
    assert sync(client, user, fake, "3005", token=user).status_code == 200


def test_a_stale_snapshot_is_not_applied(client, fake):
    user = new_user()
    add_chain("4000", user_id=user, signed_ago=dt.timedelta(0))
    old = dt.datetime.now(dt.timezone.utc) - DAY
    r = sync(
        client, user, fake, "4000",
        tx=fake.transaction("4000", token=user, revoked=True, signed_at=old),
        renewal=fake.renewal("4000", signed_at=old),
    )
    assert r.status_code == 200
    assert row("4000").status == "active"


def test_a_transaction_from_before_a_refund_does_not_undo_it(client, fake):
    # The transaction a client kept from before the refund, sent with renewal
    # info signed after it: the snapshot is as old as its older part.
    user = new_user()
    otid = "4001"
    bought_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=10)
    bought = fake.transaction(otid, token=user, signed_at=bought_at)
    r = sync(client, user, fake, otid, tx=bought, renewal=fake.renewal(otid, signed_at=bought_at))
    assert r.json()["plan"] == "plus"
    refunded_at = bought_at + dt.timedelta(minutes=5)
    notify(
        client,
        fake.notification(
            "REFUND", None,
            fake.transaction(otid, token=user, revoked=True, signed_at=refunded_at),
            fake.renewal(otid, signed_at=refunded_at),
        ),
    )
    r = sync(client, user, fake, otid, tx=bought, renewal=fake.renewal(otid))
    assert r.status_code == 200
    assert (r.json()["status"], r.json()["plan"]) == ("revoked", "free")


@pytest.mark.parametrize(
    "tx_kwargs, renewal_kwargs, status, plan",
    [
        ({}, {}, "active", "plus"),
        ({"expires_in": -DAY}, {"grace_in": DAY, "billing_retry": True}, "grace", "plus"),
        ({"expires_in": -DAY}, {"billing_retry": True}, "billing_retry", "free"),
        ({"expires_in": -DAY}, {"auto_renew": False}, "expired", "free"),
        ({"revoked": True}, {}, "revoked", "free"),
    ],
)
def test_status_is_recomputed_from_the_snapshot(client, fake, tx_kwargs, renewal_kwargs, status, plan):
    user = new_user()
    r = sync(
        client, user, fake, "5000",
        tx=fake.transaction("5000", token=user, **tx_kwargs),
        renewal=fake.renewal("5000", **renewal_kwargs),
    )
    assert r.status_code == 200, r.text
    assert (r.json()["status"], r.json()["plan"]) == (status, plan)


@pytest.mark.parametrize(
    "case",
    ["foreign_key", "foreign_chain", "renewal_of_other_chain", "xcode", "other_bundle", "not_subscription", "garbage"],
)
def test_sync_refuses_data_apple_did_not_sign_for_us(client, fake, case):
    user = new_user()
    tx, renewal = fake.transaction("6000", token=user), fake.renewal("6000")
    if case == "foreign_key":
        # Apple's certificates in the header, someone else's signature.
        claims = jwt.decode(tx, options={"verify_signature": False})
        tx = fake.sign(claims, key=ec.generate_private_key(ec.SECP256R1()))
    elif case == "foreign_chain":
        tx = FakeAppStore().transaction("6000", token=user)
    elif case == "renewal_of_other_chain":
        renewal = fake.renewal("6001")
    elif case == "xcode":
        tx = fake.transaction("6000", token=user, environment="Xcode")
    elif case == "other_bundle":
        tx = fake.transaction("6000", token=user, bundleId="com.example.other")
    elif case == "not_subscription":
        tx = fake.transaction("6000", token=user, type="Consumable")
    else:
        tx = "not.a.jws"
    r = sync(client, user, fake, "6000", tx=tx, renewal=renewal)
    assert r.status_code == 400
    assert row("6000") is None


def test_unconfigured_service_starts_and_answers_503(clean_db, fake, monkeypatch):
    monkeypatch.delenv("APPSTORE_ROOT_CERTS", raising=False)
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 200
        user = new_user()
        assert sync(c, user, fake, "7000", token=user).status_code == 503
        assert notify(c, fake.notification("TEST")).status_code == 503


# --- /appstore/notifications: design 5.4 --------------------------------

YEARLY = "ai.miranote.app.plus.yearly"


@pytest.mark.parametrize(
    "type_, subtype, tx_kwargs, renewal_kwargs, expected",
    [
        ("SUBSCRIBED", "INITIAL_BUY", {}, {}, {"status": "active"}),
        ("DID_RENEW", None, {}, {}, {"status": "active"}),
        ("OFFER_REDEEMED", "INITIAL_BUY", {}, {}, {"status": "active"}),
        ("RENEWAL_EXTENDED", None, {}, {}, {"status": "active"}),
        ("DID_CHANGE_RENEWAL_STATUS", "AUTO_RENEW_DISABLED", {}, {"auto_renew": False}, {"auto_renew": False}),
        ("DID_CHANGE_RENEWAL_PREF", "DOWNGRADE", {}, {"autoRenewProductId": YEARLY}, {"auto_renew_product_id": YEARLY}),
        ("PRICE_INCREASE", "ACCEPTED", {}, {}, {"status": "active"}),
        ("DID_FAIL_TO_RENEW", "GRACE_PERIOD", {"expires_in": -DAY}, {"grace_in": DAY, "billing_retry": True}, {"status": "grace"}),
        ("DID_FAIL_TO_RENEW", None, {"expires_in": -DAY}, {"billing_retry": True}, {"status": "billing_retry"}),
        ("GRACE_PERIOD_EXPIRED", None, {"expires_in": -DAY}, {"billing_retry": True}, {"status": "billing_retry"}),
        ("EXPIRED", "VOLUNTARY", {"expires_in": -DAY}, {"auto_renew": False}, {"status": "expired"}),
        ("REFUND", None, {"revoked": True}, {}, {"status": "revoked"}),
        ("REVOKE", None, {"revoked": True}, {}, {"status": "revoked"}),
        ("REFUND_REVERSED", None, {}, {}, {"status": "active"}),
    ],
)
def test_every_notification_type_recomputes_the_row(client, fake, type_, subtype, tx_kwargs, renewal_kwargs, expected):
    otid = "8000"
    r = notify(
        client,
        fake.notification(
            type_, subtype, fake.transaction(otid, **tx_kwargs), fake.renewal(otid, **renewal_kwargs)
        ),
    )
    assert r.status_code == 200, r.text
    current = row(otid)._mapping
    assert {k: current[k] for k in expected} == expected
    stored = run_sql("SELECT type, subtype, original_transaction_id FROM app_store_notifications")
    assert [tuple(s) for s in stored] == [(type_, subtype, otid)]


def test_refund_revokes_plus_but_keeps_used_quota_and_reversal_reinstates(client, fake):
    user = new_user()
    otid = "8100"
    assert sync(client, user, fake, otid, token=user).json()["plan"] == "plus"
    run_sql(
        "INSERT INTO ai_usage (request_id, user_id, feature, model, units, environment, status) "
        "VALUES (gen_random_uuid(), :u, 'image_generate', 'm', 1, 'Sandbox', 'ok')",
        u=user,
    )
    notify(client, fake.notification("REFUND", None, fake.transaction(otid, token=user, revoked=True), fake.renewal(otid)))
    me = client.get("/me/usage", headers=bearer(client, user)).json()
    assert me["plan"] == "free"
    assert me["features"]["image_generate"]["used"] == 1  # not clawed back
    notify(client, fake.notification("REFUND_REVERSED", None, fake.transaction(otid, token=user), fake.renewal(otid)))
    assert client.get("/me/usage", headers=bearer(client, user)).json()["plan"] == "plus"


@pytest.mark.parametrize("type_", ["CONSUMPTION_REQUEST", "TEST"])
def test_stored_but_not_applied(client, fake, type_):
    otid = "8200"
    payload = (
        fake.notification(type_, None, fake.transaction(otid, revoked=True), fake.renewal(otid))
        if type_ == "CONSUMPTION_REQUEST"
        else fake.notification(type_)
    )
    assert notify(client, payload).status_code == 200
    assert row(otid) is None
    assert run_sql("SELECT type FROM app_store_notifications")[0][0] == type_


def test_a_retried_notification_is_skipped(client, fake):
    otid = "8300"
    signed = fake.notification("DID_RENEW", None, fake.transaction(otid), fake.renewal(otid))
    assert notify(client, signed).status_code == 200
    run_sql("UPDATE subscriptions SET status = 'expired', last_signed_date = now() - interval '1 day'")
    assert notify(client, signed).status_code == 200
    assert row(otid).status == "expired"
    assert run_sql("SELECT count(*) FROM app_store_notifications")[0][0] == 1


def test_applying_and_storing_are_one_transaction(client, fake, monkeypatch):
    async def broken(*args):
        raise RuntimeError("apply failed")

    monkeypatch.setattr(subscriptions, "apply_snapshot", broken)
    with pytest.raises(RuntimeError):
        notify(client, fake.notification("DID_RENEW", None, fake.transaction("8400"), fake.renewal("8400")))
    assert run_sql("SELECT count(*) FROM app_store_notifications")[0][0] == 0


def test_payload_is_kept_without_the_account_token(client, fake):
    user = new_user()
    otid = "8500"
    notify(client, fake.notification("SUBSCRIBED", None, fake.transaction(otid, token=user), fake.renewal(otid)))
    payload = run_sql("SELECT payload FROM app_store_notifications")[0][0]
    data = payload["data"]
    assert "signedTransactionInfo" not in data and "signedRenewalInfo" not in data
    assert data["transactionInfo"]["originalTransactionId"] == otid
    assert "appAccountToken" not in data["transactionInfo"]
    assert str(user) not in str(payload)


def test_notification_links_only_unowned_chains_to_live_users(client, fake):
    live, deleted, other = new_user(), new_user(deleting=True), new_user()
    # A live token on a new chain: owned.
    notify(client, fake.notification("SUBSCRIBED", None, fake.transaction("8600", token=live), fake.renewal("8600")))
    assert row("8600").user_id == live
    # A deleted account's token: stays unowned, no foreign-key failure.
    r = notify(client, fake.notification("DID_RENEW", None, fake.transaction("8601", token=deleted), fake.renewal("8601")))
    assert r.status_code == 200 and row("8601").user_id is None
    # An owned chain is never moved by a notification.
    notify(client, fake.notification("DID_RENEW", None, fake.transaction("8600", token=other), fake.renewal("8600")))
    assert row("8600").user_id == live


def test_a_sync_and_a_notification_linking_the_same_chain_do_not_deadlock(clean_db, configured, fake):
    # Right after a purchase the app's sync and SUBSCRIBED can arrive
    # together. The sync locks the user, then waits for the chain's row,
    # which the notification has locked; the notification then links the
    # chain to that user, and the foreign key check reads the user's row.
    user = new_user()
    otid = "8650"
    add_chain(otid)

    async def interleaved():
        snap = await appstore.from_config().snapshot(fake.transaction(otid, token=user), fake.renewal(otid))
        now = dt.datetime.now(dt.timezone.utc)
        engine = db.create_engine(TEST_URL, poolclass=NullPool)
        try:
            async with engine.connect() as sync_conn, engine.connect() as notify_conn:
                sync_tx, notify_tx = await sync_conn.begin(), await notify_conn.begin()
                await subscriptions.apply_snapshot(notify_conn, snap, now)
                await subscriptions.lock_user(sync_conn, user)
                waiting = asyncio.ensure_future(subscriptions.apply_snapshot(sync_conn, snap, now))
                await asyncio.sleep(0.3)
                await asyncio.wait_for(subscriptions.adopt_by_token(notify_conn, snap, now), 10)
                await notify_tx.commit()
                await asyncio.wait_for(waiting, 10)
                await sync_tx.commit()
        finally:
            await engine.dispose()

    asyncio.run(interleaved())
    assert row(otid).user_id == user


def test_offer_code_redeemed_outside_the_app_is_linked_at_the_next_sync(client, fake, api):
    otid = "8700"
    notify(client, fake.notification("OFFER_REDEEMED", "INITIAL_BUY", fake.transaction(otid), fake.renewal(otid)))
    assert row(otid).user_id is None
    user = new_user()
    r = sync(client, user, fake, otid)
    assert r.status_code == 200 and r.json()["plan"] == "plus"
    assert api.tokens == [(otid, str(user))]


@pytest.mark.parametrize("case", ["foreign_chain", "other_app", "garbage"])
def test_notification_refuses_data_apple_did_not_sign_for_us(client, fake, case):
    if case == "foreign_chain":
        signed = FakeAppStore().notification("TEST")
    elif case == "other_app":
        signed = fake.notification("TEST", environment="Production", appAppleId=1)
    else:
        signed = "garbage"
    assert notify(client, signed).status_code == 400
    assert run_sql("SELECT count(*) FROM app_store_notifications")[0][0] == 0


# --- backfill.py: design 5.5 --------------------------------------------


def _run_backfill(store, ids, default_env="Production"):
    async def _go():
        engine = db.create_engine(TEST_URL, poolclass=NullPool)
        try:
            return await backfill.backfill(engine, store, ids, default_env)
        finally:
            await engine.dispose()

    return asyncio.run(_go())


def test_backfill_recomputes_known_chains(clean_db, configured, fake, api):
    user = new_user()
    add_chain("9000", user_id=user)
    api.statuses["9000"] = StatusResponse(
        data=[
            SubscriptionGroupIdentifierItem(
                lastTransactions=[
                    LastTransactionsItem(
                        originalTransactionId="9000",
                        signedTransactionInfo=fake.transaction("9000", token=user, revoked=True),
                        signedRenewalInfo=fake.renewal("9000"),
                    )
                ]
            )
        ]
    )
    store = appstore.from_config()
    store.clients = {"Sandbox": api, "Production": api}
    assert _run_backfill(store, []) == 1
    assert row("9000").status == "revoked" and row("9000").user_id == user


def test_backfill_needs_the_server_api_key(clean_db, configured):
    store = appstore.from_config()
    assert store.verifiers and not store.clients
    with pytest.raises(appstore.NotConfigured):
        _run_backfill(store, ["1"])

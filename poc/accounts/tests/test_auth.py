"""Sign in with Apple, refresh rotation and sign-out against a real
Postgres. Apple is faked with a locally generated RSA key served as a JWKS
through httpx.MockTransport: no network."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import io
import json
import logging
import time
import uuid

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

import apple
import main
import tokens
from tests.conftest import new_ec_key_pem, run_sql

BUNDLE_ID = "ai.miranote.app"
NONCE = "raw-nonce-from-the-app"


class FakeApple:
    """Apple's signing key and its /auth/keys endpoint."""

    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "fake-apple-kid"
        self.fetches = 0

    def jwks(self):
        jwk = json.loads(RSAAlgorithm.to_jwk(self.key.public_key()))
        return {"keys": [dict(jwk, kid=self.kid, alg="RS256", use="sig")]}

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == apple.APPLE_KEYS_URL
        self.fetches += 1
        return httpx.Response(200, json=self.jwks())

    def token(self, sub="001234.abcdef.0001", nonce=NONCE, key=None, kid=None, **overrides):
        now = int(time.time())
        claims = {
            "iss": apple.APPLE_ISSUER,
            "aud": BUNDLE_ID,
            "sub": sub,
            "iat": now,
            "exp": now + 600,
            "nonce": apple.nonce_hash(nonce),
            "nonce_supported": True,
        }
        claims.update(overrides)
        return jwt.encode(claims, key or self.key, algorithm="RS256", headers={"kid": kid or self.kid})


@pytest.fixture
def fake_apple():
    return FakeApple()


@pytest.fixture
def client(clean_db, fake_apple, monkeypatch):
    monkeypatch.setattr(main, "APPLE_TRANSPORT", httpx.MockTransport(fake_apple.handler))
    with TestClient(main.app) as c:
        yield c


def b64url(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def sign_in(client, token, nonce=NONCE):
    return client.post("/auth/apple", json={"identity_token": token, "nonce": nonce})


def refresh(client, refresh_token):
    return client.post("/auth/refresh", json={"refresh_token": refresh_token})


def me(client, access_token):
    return client.get("/me", headers={"Authorization": "Bearer " + access_token})


# --- /auth/apple -------------------------------------------------------------


def test_valid_identity_token_creates_the_user_and_returns_our_tokens(client, fake_apple):
    r = sign_in(client, fake_apple.token())
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "Bearer" and body["expires_in"] == 3600

    rows = run_sql("SELECT id, apple_sub, apple_team_id FROM users")
    assert [(str(rows[0][0]), rows[0][1], rows[0][2])] == [
        (body["user_id"], "001234.abcdef.0001", "W7U365VR4U")
    ]
    # Only the hash of the refresh token is stored.
    stored = run_sql("SELECT token_hash, expires_at - created_at FROM sessions")
    assert stored == [(tokens.hash_refresh_token(body["refresh_token"]), dt.timedelta(days=60))]

    header = jwt.get_unverified_header(body["access_token"])
    assert header["alg"] == "ES256" and header["kid"] == main.app.state.signer.kid
    assert me(client, body["access_token"]).json() == {"id": body["user_id"]}


@pytest.mark.parametrize(
    "overrides",
    [
        {"aud": "com.example.other"},
        {"iss": "https://evil.example.com"},
        {"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},
        {"nonce": apple.nonce_hash("another-nonce")},
    ],
    ids=["wrong-aud", "wrong-iss", "expired", "bad-nonce"],
)
def test_invalid_identity_token_is_refused_and_creates_nothing(client, fake_apple, overrides):
    r = sign_in(client, fake_apple.token(**overrides))
    assert r.status_code == 401
    assert run_sql("SELECT count(*) FROM users") == [(0,)]
    assert run_sql("SELECT count(*) FROM sessions") == [(0,)]


def test_token_without_nonce_is_refused(client, fake_apple):
    token = fake_apple.token()
    claims = jwt.decode(token, options={"verify_signature": False})
    del claims["nonce"]
    no_nonce = jwt.encode(claims, fake_apple.key, algorithm="RS256", headers={"kid": fake_apple.kid})
    assert sign_in(client, no_nonce).status_code == 401


def test_token_signed_by_another_key_is_refused(client, fake_apple):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert sign_in(client, fake_apple.token(key=other)).status_code == 401
    assert sign_in(client, fake_apple.token(key=other, kid="unknown")).status_code == 401
    assert sign_in(client, "not-a-jwt").status_code == 401


def test_hs256_token_is_refused(client, fake_apple):
    # Algorithm confusion: an HMAC "signature" keyed with Apple's public key.
    # PyJWT will not HMAC with a PEM key, so the token is built by hand.
    public_pem = fake_apple.key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    claims = jwt.decode(fake_apple.token(), options={"verify_signature": False})
    header = {"alg": "HS256", "kid": fake_apple.kid}
    signing_input = b64url(json.dumps(header).encode()) + b"." + b64url(json.dumps(claims).encode())
    signature = hmac.new(public_pem, signing_input, hashlib.sha256).digest()
    forged = (signing_input + b"." + b64url(signature)).decode()
    assert sign_in(client, forged).status_code == 401


def test_repeat_sign_in_is_the_same_user_with_a_new_session(client, fake_apple):
    first = sign_in(client, fake_apple.token()).json()
    second = sign_in(client, fake_apple.token()).json()
    assert first["user_id"] == second["user_id"]
    assert run_sql("SELECT count(*) FROM users") == [(1,)]
    families = run_sql("SELECT DISTINCT family_id FROM sessions")
    assert len(families) == 2

    other = sign_in(client, fake_apple.token(sub="009999.other.0002")).json()
    assert other["user_id"] != first["user_id"]


def test_apple_keys_are_cached_and_unknown_kids_refetch_rarely(client, fake_apple):
    sign_in(client, fake_apple.token())
    sign_in(client, fake_apple.token())
    assert fake_apple.fetches == 1
    # Garbage kids do not make us hammer Apple.
    for _ in range(3):
        assert sign_in(client, fake_apple.token(kid="rotated-away")).status_code == 401
    assert fake_apple.fetches == 1


# --- /auth/refresh -----------------------------------------------------------


def test_refresh_rotates_the_pair(client, fake_apple):
    first = sign_in(client, fake_apple.token()).json()
    r = refresh(client, first["refresh_token"])
    assert r.status_code == 200
    second = r.json()
    assert second["refresh_token"] != first["refresh_token"]
    assert second["user_id"] == first["user_id"]
    assert me(client, second["access_token"]).status_code == 200
    # One family: the old token is marked replaced, not deleted.
    rows = run_sql("SELECT replaced_at IS NOT NULL FROM sessions ORDER BY created_at, replaced_at NULLS LAST")
    assert sorted(r[0] for r in rows) == [False, True]
    assert len(run_sql("SELECT DISTINCT family_id FROM sessions")) == 1
    assert refresh(client, second["refresh_token"]).status_code == 200


def test_replaced_token_within_grace_still_gets_a_pair(client, fake_apple):
    # Two refreshes in parallel at launch must not sign the user out.
    first = sign_in(client, fake_apple.token()).json()
    a = refresh(client, first["refresh_token"])
    b = refresh(client, first["refresh_token"])
    assert a.status_code == 200 and b.status_code == 200
    assert refresh(client, a.json()["refresh_token"]).status_code == 200
    assert refresh(client, b.json()["refresh_token"]).status_code == 200


def test_replaced_token_reused_after_grace_revokes_the_whole_sign_in(client, fake_apple):
    first = sign_in(client, fake_apple.token()).json()
    second = refresh(client, first["refresh_token"]).json()
    other_device = sign_in(client, fake_apple.token()).json()
    run_sql("UPDATE sessions SET replaced_at = now() - interval '11 seconds' WHERE replaced_at IS NOT NULL")

    assert refresh(client, first["refresh_token"]).status_code == 401
    # The thief's copy and the legitimate newest token are both dead now.
    assert refresh(client, second["refresh_token"]).status_code == 401
    # Another sign-in (another family) is untouched.
    assert refresh(client, other_device["refresh_token"]).status_code == 200


def test_unknown_and_expired_refresh_tokens_are_refused(client, fake_apple):
    assert refresh(client, "never-issued").status_code == 401
    first = sign_in(client, fake_apple.token()).json()
    run_sql("UPDATE sessions SET expires_at = now() - interval '1 second'")
    assert refresh(client, first["refresh_token"]).status_code == 401


# --- /auth/signout -----------------------------------------------------------


def test_sign_out_revokes_the_sign_in(client, fake_apple):
    first = sign_in(client, fake_apple.token()).json()
    second = refresh(client, first["refresh_token"]).json()
    other_device = sign_in(client, fake_apple.token()).json()

    assert client.post("/auth/signout", json={"refresh_token": second["refresh_token"]}).status_code == 204
    assert refresh(client, second["refresh_token"]).status_code == 401
    assert refresh(client, other_device["refresh_token"]).status_code == 200
    # Idempotent, and an unknown token says nothing either way.
    assert client.post("/auth/signout", json={"refresh_token": second["refresh_token"]}).status_code == 204
    assert client.post("/auth/signout", json={"refresh_token": "never-issued"}).status_code == 204


# --- current_user and our keys ------------------------------------------------


def test_current_user_refuses_missing_bad_and_expired_tokens(client, fake_apple):
    assert client.get("/me").status_code == 401
    assert me(client, "garbage").status_code == 401
    body = sign_in(client, fake_apple.token()).json()
    signer = main.app.state.signer
    expired = signer.issue(uuid.UUID(body["user_id"]), dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2))
    assert me(client, expired).status_code == 401
    # Signed by a key that is not ours.
    stranger = tokens.Signer(new_ec_key_pem()).issue(uuid.UUID(body["user_id"]), dt.datetime.now(dt.timezone.utc))
    assert me(client, stranger).status_code == 401


def test_current_user_refuses_a_user_being_deleted(client, fake_apple):
    body = sign_in(client, fake_apple.token()).json()
    run_sql("UPDATE users SET deletion_requested_at = now()")
    assert me(client, body["access_token"]).status_code == 401


def test_jwks_serves_our_public_key(client):
    keys = client.get("/.well-known/jwks.json").json()["keys"]
    assert [k["kid"] for k in keys] == [main.app.state.signer.kid]
    assert keys[0]["kty"] == "EC" and keys[0]["crv"] == "P-256" and "d" not in keys[0]


def test_rotation_keeps_the_previous_key_valid(clean_db, monkeypatch):
    old_pem = new_ec_key_pem()
    old = tokens.Signer(old_pem)
    user_id = run_sql(
        "INSERT INTO users (apple_sub, apple_team_id) VALUES ('s', 'W7U365VR4U') RETURNING id"
    )[0][0]
    token_from_old_key = old.issue(user_id, dt.datetime.now(dt.timezone.utc))

    old_public = (
        serialization.load_pem_private_key(old_pem.encode(), None)
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    monkeypatch.setenv("ACCOUNTS_PREVIOUS_PUBLIC_KEY", old_public)
    with TestClient(main.app) as c:
        kids = [k["kid"] for k in c.get("/.well-known/jwks.json").json()["keys"]]
        assert kids == [main.app.state.signer.kid, old.kid]
        assert me(c, token_from_old_key).json() == {"id": str(user_id)}
        # New tokens are signed with the current key only.
        new = main.app.state.signer.issue(user_id, dt.datetime.now(dt.timezone.utc))
        assert jwt.get_unverified_header(new)["kid"] == main.app.state.signer.kid


def test_signing_key_is_required(monkeypatch):
    monkeypatch.delenv("ACCOUNTS_SIGNING_KEY")
    with pytest.raises(RuntimeError, match="ACCOUNTS_SIGNING_KEY is not set"):
        with TestClient(main.app):
            pass


# --- logs ---------------------------------------------------------------------


def test_tokens_never_reach_the_log(client, fake_apple):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(main.JsonFormatter())
    logging.getLogger("accounts").addHandler(handler)
    try:
        identity = fake_apple.token()
        pair = sign_in(client, identity).json()
        second = refresh(client, pair["refresh_token"]).json()
        run_sql("UPDATE sessions SET replaced_at = now() - interval '11 seconds' WHERE replaced_at IS NOT NULL")
        refresh(client, pair["refresh_token"])
        bad = fake_apple.token(aud="com.example.other")
        sign_in(client, bad)
        me(client, second["access_token"])
        me(client, "garbage-access-token")
    finally:
        logging.getLogger("accounts").removeHandler(handler)

    out = buf.getvalue()
    for secret in (identity, bad, pair["refresh_token"], pair["access_token"], second["access_token"]):
        assert secret not in out
    assert "garbage-access-token" not in out
    reasons = [json.loads(line).get("reason") for line in out.splitlines()]
    assert "audience" in reasons

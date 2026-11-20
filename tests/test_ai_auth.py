"""The AI services' gate (ai_auth) against a fake accounts service.

Each service's own tests prove it is wired to the gate with the right
features; these pin the gate itself. The real accounts service's side of
reserve/settle is tested in poc/accounts/tests/test_usage.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

import ai_auth
import beta_auth
from fake_accounts import FakeAccounts


def _app():
    app = FastAPI()

    @app.post("/expand")
    async def expand(fail: bool = False, crash: bool = False):
        if crash:
            raise RuntimeError("provider exploded")
        if fail:
            raise HTTPException(502, detail="provider failed")
        return {"ok": True}

    @app.post("/later")
    async def later(request: Request):
        request.app.state.ticket = ai_auth.take_ticket(request)
        return {"queued": True}

    @app.post("/free")
    async def free():
        return {"ok": True}

    @app.post("/mood")
    async def mood():
        return {"mood": "calm"}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    ai_auth.install(
        app,
        metered={
            "/expand": ai_auth.Metered("expand"),
            "/later": ai_auth.Metered("photo_push"),
            "/mood": ai_auth.Metered("mood"),
        },
        beta_always={"/mood"},
    )
    return app


@pytest.fixture
def fake(monkeypatch):
    fake = FakeAccounts(caps={"expand": 2})
    fake.install(monkeypatch)
    monkeypatch.setenv("BETA_TOKENS", "beta-token")
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    beta_auth._limiter._hits.clear()
    return fake


@pytest.fixture
def client(fake):
    return TestClient(_app(), raise_server_exceptions=False)


BETA = {"Authorization": "Bearer beta-token"}


def test_health_needs_nothing(client):
    assert client.get("/health").status_code == 200


def test_no_token_is_401_missing_token(client):
    r = client.post("/expand")
    assert r.status_code == 401 and r.json()["reason"] == "missing_token"


def test_a_user_token_is_metered_and_settled_ok(client, fake):
    assert client.post("/expand", headers=fake.headers()).status_code == 200
    assert fake.statuses("expand") == ["ok"]
    assert fake.settles[0]["cost_usd"] == ai_auth.COST_USD["expand"]


def test_expired_token_is_401_token_expired(client, fake):
    r = client.post("/expand", headers=fake.headers(expires_in=-10))
    assert r.status_code == 401 and r.json()["reason"] == "token_expired"
    assert fake.rows == {}


def test_a_token_signed_by_another_key_is_401(client, fake):
    other = ec.generate_private_key(ec.SECP256R1())
    token = jwt.encode({"sub": str(fake.user_id), "iss": ai_auth.ISSUER, "aud": ai_auth.AUDIENCE, "exp": 2**31},
                       other, algorithm="ES256", headers={"kid": fake.kid})
    r = client.post("/expand", headers={"Authorization": "Bearer " + token})
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"


def test_garbage_token_is_401(client):
    r = client.post("/expand", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"


def test_over_cap_is_429_with_the_reset_time(client, fake):
    for _ in range(2):
        assert client.post("/expand", headers=fake.headers()).status_code == 200
    r = client.post("/expand", headers=fake.headers())
    assert r.status_code == 429
    assert r.json()["reason"] == "quota" and r.json()["feature"] == "expand" and r.json()["resets_at"]


def test_a_failed_provider_call_is_not_charged(client, fake):
    assert client.post("/expand", params={"fail": True}, headers=fake.headers()).status_code == 502
    assert client.post("/expand", params={"crash": True}, headers=fake.headers()).status_code == 500
    assert fake.statuses("expand") == ["failed", "failed"]
    assert fake.charged("expand") == 0


def test_a_retry_of_a_counted_request_is_409(client, fake):
    headers = fake.headers()
    assert client.post("/expand", headers=headers).status_code == 200
    r = client.post("/expand", headers=headers)
    assert r.status_code == 409 and r.json()["reason"] == "duplicate_request"


def test_a_retry_of_a_failed_request_runs_again(client, fake):
    headers = fake.headers()
    client.post("/expand", params={"fail": True}, headers=headers)
    assert client.post("/expand", headers=headers).status_code == 200
    assert fake.statuses("expand") == ["ok"]


def test_unmetered_route_takes_a_user_token_without_a_reserve(client, fake):
    assert client.post("/free", headers=fake.headers()).status_code == 200
    assert fake.rows == {}


def test_a_deleted_user_is_cut_off_at_once_on_unmetered_routes(client, fake):
    """Design 3.2, 6.1: DELETE /me cuts off at once, not when the access
    token expires (up to 1 h) -- on /describe, /clean and transcription
    too, which never reserve."""
    headers = fake.headers()
    assert client.post("/free", headers=headers).status_code == 200
    fake.users.discard(fake.user_id)
    r = client.post("/free", headers=headers)
    assert r.status_code == 401 and r.json()["reason"] == "unknown_user"
    assert fake.rows == {}


def test_a_handler_that_takes_its_ticket_settles_it_itself(client, fake):
    assert client.post("/later", headers=fake.headers()).status_code == 200
    assert fake.statuses("photo_push") == ["pending"]
    client.app.state.ticket.settle(True)
    assert fake.statuses("photo_push") == ["ok"]


def test_unknown_user_is_401(client, fake):
    r = client.post("/expand", headers=fake.headers(user_id=uuid.uuid4()))
    assert r.status_code == 401 and r.json()["reason"] == "unknown_user"


def test_accounts_down_fails_closed_with_503(client, fake):
    fake.reachable = False
    r = client.post("/expand", headers=fake.headers())
    assert r.status_code == 503 and r.json()["reason"] == "accounts_unavailable"


def test_accounts_down_after_the_keys_are_cached_still_fails_closed(client, fake):
    assert client.post("/free", headers=fake.headers()).status_code == 200
    fake.reachable = False
    # The cached keys still verify the token, but nothing runs without the
    # accounts service: metered routes cannot reserve, unmetered ones cannot
    # tell whether the account was deleted.
    for path in ("/expand", "/free"):
        r = client.post(path, headers=fake.headers())
        assert r.status_code == 503 and r.json()["reason"] == "accounts_unavailable"


def test_jwks_is_fetched_once_not_per_request(client, fake, monkeypatch):
    fetches = []
    handle = fake.handle

    def counting(request):
        if request.url.path == "/.well-known/jwks.json":
            fetches.append(1)
        return handle(request)

    monkeypatch.setattr(ai_auth, "TRANSPORT", httpx.MockTransport(counting))
    for _ in range(3):
        client.post("/free", headers=fake.headers())
    assert len(fetches) == 1


def test_missing_internal_secret_fails_closed(client, fake, monkeypatch):
    monkeypatch.delenv("ACCOUNTS_INTERNAL_SECRET")
    r = client.post("/expand", headers=fake.headers())
    assert r.status_code == 503 and fake.rows == {}


def test_beta_token_is_accepted_by_default_and_not_metered(client, fake):
    assert client.post("/expand", headers=BETA).status_code == 200
    assert fake.rows == {}


def test_beta_flag_off_rejects_the_beta_token(client, fake, monkeypatch):
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    r = client.post("/expand", headers=BETA)
    assert r.status_code == 401 and r.json()["reason"] == "invalid_token"
    assert client.post("/expand", headers=fake.headers()).status_code == 200


def test_beta_always_path_keeps_the_beta_token_with_the_flag_off(client, fake, monkeypatch):
    monkeypatch.setenv("ACCEPT_BETA_TOKENS", "0")
    assert client.post("/mood", headers=BETA).status_code == 200
    assert fake.rows == {}
    assert client.post("/mood", headers=fake.headers()).status_code == 200
    assert fake.statuses("mood") == ["ok"]


def test_without_pyjwt_the_gate_loads_and_the_beta_token_still_works():
    """The beta Mac's service venvs have no PyJWT until someone runs
    `pip install -r requirements.txt` after pulling. Every service imports
    ai_auth at load, so a missing PyJWT must not take the beta token down
    with it; a signed-in call then fails closed."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["jwt"] = None  # as if PyJWT were not installed
        sys.modules["jwt.algorithms"] = None
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import ai_auth

        app = FastAPI()

        @app.post("/expand")
        async def expand():
            return {"ok": True}

        ai_auth.install(app, metered={"/expand": ai_auth.Metered("expand")})
        client = TestClient(app)
        beta = client.post("/expand", headers={"Authorization": "Bearer beta-token"})
        user = client.post("/expand", headers={"Authorization": "Bearer eyJhbGciOiJFUzI1NiIsImtpZCI6ImsifQ.e30.c2ln"})
        print(beta.status_code, user.status_code, user.json()["reason"])
        """
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), BETA_TOKENS="beta-token")
    run = subprocess.run([sys.executable, "-c", script], cwd=str(root), env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.split()[-3:] == ["200", "503", "accounts_unavailable"]


def test_a_failed_jwks_refresh_is_not_retried_on_every_request(fake, monkeypatch):
    """Design 3.2: no network hop per request. Once the cached keys are an
    hour old and the accounts service is down (or hangs, while every
    signed-in request waits on the lock), the keys in hand keep serving and
    the refresh is tried again at most once a minute."""
    now = [1000.0]
    jwks = ai_auth._Jwks(clock=lambda: now[0])
    assert jwks.key(fake.kid) is not None
    fetches = []
    handle = fake.handle

    def counting(request):
        if request.url.path == "/.well-known/jwks.json":
            fetches.append(1)
        return handle(request)

    monkeypatch.setattr(ai_auth, "TRANSPORT", httpx.MockTransport(counting))
    fake.reachable = False
    now[0] += ai_auth.JWKS_TTL_SECONDS + 1
    for _ in range(3):
        assert jwks.key(fake.kid) is not None
    assert len(fetches) == 1
    fake.reachable = True
    now[0] += ai_auth.JWKS_REFETCH_SECONDS + 1
    assert jwks.key(fake.kid) is not None
    assert len(fetches) == 2


def test_a_user_token_has_the_beta_tokens_rate_limit(client, fake, monkeypatch):
    """The unmetered routes (/describe, /clean, transcription, ...) have no
    cap, and each is still a paid model call: the per-token limit that
    bounds the beta token bounds each signed-in user the same way."""
    monkeypatch.setattr(beta_auth, "_limiter", beta_auth.TokenRateLimiter(limit=3))
    auth = {"Authorization": "Bearer " + fake.token()}
    assert [client.post("/free", headers=auth).status_code for _ in range(4)] == [200, 200, 200, 429]
    r = client.post("/expand", headers=dict(auth, **{"X-Request-ID": str(uuid.uuid4())}))
    assert r.status_code == 429 and r.json()["reason"] == "rate_limited"
    assert fake.rows == {}  # refused before anything was reserved
    # Another user, and the beta token, have their own windows.
    other_id = uuid.uuid4()
    fake.users.add(other_id)
    other = {"Authorization": "Bearer " + fake.token(user_id=other_id)}
    assert client.post("/free", headers=other).status_code == 200
    assert client.post("/free", headers=BETA).status_code == 200

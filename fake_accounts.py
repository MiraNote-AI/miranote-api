"""A stand-in accounts service for the AI services' tests (not used at run
time). It signs access tokens with a throwaway key, serves the JWKS and
answers /usage/user, /usage/reserve and /usage/settle like the real one,
keeping the users and the ledger in memory. The real service's side of
the same contract is tested against Postgres in
poc/accounts/tests/test_usage.py, including through ai_auth itself.

    fake = FakeAccounts(caps={"expand": 1})
    fake.install(monkeypatch)          # ai_auth talks to the fake
    headers = fake.headers()           # Authorization + X-Request-ID
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Dict, List, Optional

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm

import ai_auth

INTERNAL_SECRET = "test-internal-secret"
RESETS_AT = "2026-12-01T00:00:00+00:00"


class FakeAccounts:
    def __init__(self, caps: Optional[Dict[str, int]] = None):
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.kid = "fake-kid"
        self.user_id = uuid.uuid4()
        # The live accounts; discard one to delete it.
        self.users = {self.user_id}
        self.caps = caps or {}
        self.reachable = True
        self.rows: Dict[str, dict] = {}  # request_id -> {feature, status}
        self.settles: List[dict] = []

    # -- what the app holds -------------------------------------------------

    def token(self, expires_in: int = 3600, user_id: Optional[uuid.UUID] = None) -> str:
        now = dt.datetime.now(dt.timezone.utc)
        claims = {
            "iss": ai_auth.ISSUER,
            "aud": ai_auth.AUDIENCE,
            "sub": str(user_id or self.user_id),
            "iat": int(now.timestamp()),
            "exp": int((now + dt.timedelta(seconds=expires_in)).timestamp()),
        }
        return jwt.encode(claims, self.key, algorithm="ES256", headers={"kid": self.kid})

    def headers(self, **token_kw) -> Dict[str, str]:
        return {"Authorization": "Bearer " + self.token(**token_kw), "X-Request-ID": str(uuid.uuid4())}

    # -- what the tests read ------------------------------------------------

    def charged(self, feature: str) -> int:
        return sum(1 for r in self.rows.values() if r["feature"] == feature and r["status"] != "failed")

    def statuses(self, feature: str) -> List[str]:
        return [r["status"] for r in self.rows.values() if r["feature"] == feature]

    # -- the service --------------------------------------------------------

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(ai_auth, "TRANSPORT", httpx.MockTransport(self.handle))
        monkeypatch.setenv("ACCOUNTS_INTERNAL_SECRET", INTERNAL_SECRET)
        monkeypatch.setattr(ai_auth, "JWKS", ai_auth._Jwks())

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not self.reachable:
            raise httpx.ConnectError("accounts down", request=request)
        if request.url.path == "/.well-known/jwks.json":
            jwk = json.loads(ECAlgorithm.to_jwk(self.key.public_key()))
            return httpx.Response(200, json={"keys": [dict(jwk, kid=self.kid, alg="ES256", use="sig")]})
        if request.headers.get("X-Internal-Secret") != INTERNAL_SECRET:
            return httpx.Response(403, json={"detail": "internal route"})
        body = json.loads(request.content)
        if request.url.path == "/usage/reserve":
            return self._reserve(body)
        if request.url.path == "/usage/user":
            if uuid.UUID(body["user_id"]) not in self.users:
                return httpx.Response(401, json={"reason": "unknown_user"})
            return httpx.Response(200, json={"user_id": body["user_id"]})
        if request.url.path == "/usage/settle":
            return self._settle(body)
        return httpx.Response(404)

    def _reserve(self, body: dict) -> httpx.Response:
        feature = body["feature"]
        if uuid.UUID(body["user_id"]) not in self.users:
            return httpx.Response(401, json={"reason": "unknown_user"})
        if body["request_id"] in self.rows and self.rows[body["request_id"]]["status"] != "failed":
            return httpx.Response(409, json={"reason": "duplicate_request"})
        cap = self.caps.get(feature)
        if cap is not None and self.charged(feature) >= cap:
            return httpx.Response(
                429,
                json={"reason": "quota", "feature": feature, "plan": "free", "limit": cap, "resets_at": RESETS_AT},
            )
        self.rows[body["request_id"]] = {"feature": feature, "status": "pending"}
        return httpx.Response(200, json={"request_id": body["request_id"]})

    def _settle(self, body: dict) -> httpx.Response:
        self.settles.append(body)
        row = self.rows.get(body["request_id"])
        if row is None:
            return httpx.Response(404, json={"reason": "unknown_request"})
        if row["status"] == "pending":
            row["status"] = body["status"]
        return httpx.Response(200, json={"status": row["status"]})

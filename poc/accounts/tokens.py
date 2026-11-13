"""Our own tokens (design 3.2).

- Access token: ES256 JWT, 1 h, `kid` header, `sub` = our user id. The AI
  services will verify it against /.well-known/jwks.json (step 3).
- Refresh token: opaque, 60 days, only its SHA-256 is stored (sessions).

The kid is the RFC 7638 thumbprint of the public key, so a key carries its
own id and rotation is: new key in ACCOUNTS_SIGNING_KEY, the old public key
in ACCOUNTS_PREVIOUS_PUBLIC_KEY for one access-token lifetime, then drop it.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import secrets
import uuid
from typing import Any, Dict, List

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from jwt.algorithms import ECAlgorithm

ISSUER = "miranote-accounts"
AUDIENCE = "miranote"
ACCESS_TTL = dt.timedelta(hours=1)
REFRESH_TTL = dt.timedelta(days=60)


class InvalidAccessToken(Exception):
    """Not one of our valid access tokens. Never carries the token."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _public_jwk(public_key: ec.EllipticCurvePublicKey) -> Dict[str, str]:
    if not isinstance(public_key.curve, ec.SECP256R1):
        raise ValueError("signing keys must be EC P-256")
    jwk = json.loads(ECAlgorithm.to_jwk(public_key))
    core = {"crv": jwk["crv"], "kty": jwk["kty"], "x": jwk["x"], "y": jwk["y"]}
    thumbprint = hashlib.sha256(json.dumps(core, separators=(",", ":"), sort_keys=True).encode())
    return dict(core, kid=_b64url(thumbprint.digest()), alg="ES256", use="sig")


class Signer:
    """Signs access tokens with the current key; verifies with the current
    and the previous one."""

    def __init__(self, private_pem: str, previous_public_pem: str = ""):
        self._private = load_pem_private_key(private_pem.encode(), password=None)
        current = _public_jwk(self._private.public_key())
        self.kid = current["kid"]
        self._public: Dict[str, Any] = {self.kid: self._private.public_key()}
        self.jwks: List[Dict[str, str]] = [current]
        if previous_public_pem:
            previous_key = load_pem_public_key(previous_public_pem.encode())
            previous = _public_jwk(previous_key)
            self._public[previous["kid"]] = previous_key
            self.jwks.append(previous)

    def issue(self, user_id: uuid.UUID, now: dt.datetime) -> str:
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": str(user_id),
            "iat": int(now.timestamp()),
            "exp": int((now + ACCESS_TTL).timestamp()),
        }
        return jwt.encode(claims, self._private, algorithm="ES256", headers={"kid": self.kid})

    def verify(self, token: str) -> uuid.UUID:
        """The user id in a valid access token."""
        try:
            kid = jwt.get_unverified_header(token).get("kid")
            key = self._public.get(kid)
            if key is None:
                raise InvalidAccessToken("unknown kid")
            claims = jwt.decode(
                token,
                key,
                algorithms=["ES256"],
                audience=AUDIENCE,
                issuer=ISSUER,
                options={"require": ["iss", "aud", "exp", "sub"]},
            )
            return uuid.UUID(claims["sub"])
        except (jwt.PyJWTError, ValueError) as exc:
            raise InvalidAccessToken(type(exc).__name__)


def new_refresh_token() -> str:
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

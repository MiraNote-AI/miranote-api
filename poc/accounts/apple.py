"""Verifying Sign in with Apple identity tokens (design 3.1).

The app runs ASAuthorizationAppleIDProvider with SHA-256(nonce) as the
request nonce and sends us the identity token plus the raw nonce. The token
is an RS256 JWT signed with one of Apple's keys at APPLE_KEYS_URL.

We never exchange the authorization code and never store an Apple refresh
token: deletion re-runs Sign in with Apple (design 7).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

log = logging.getLogger("accounts.apple")

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"

# Apple rotates its keys rarely; a token with an unknown kid triggers a
# refetch, at most once per REFETCH_SECONDS so garbage tokens cannot make us
# hammer Apple.
KEYS_TTL_SECONDS = 24 * 3600
REFETCH_SECONDS = 60
# Clock skew between Apple, the phone and us.
LEEWAY_SECONDS = 60


class InvalidIdentityToken(Exception):
    """The token is not a valid identity token for this app. The message is
    a short reason for the log; it never contains the token."""


class AppleKeysUnavailable(Exception):
    """The token's kid is not cached and the last fetch of Apple's keys
    failed, so we cannot tell a valid token from a bad one. The caller
    should retry later; it is not the token's fault."""


@dataclass(frozen=True)
class AppleIdentity:
    sub: str


class AppleKeys:
    """Apple's public keys by kid, fetched from APPLE_KEYS_URL and cached."""

    def __init__(self, http: httpx.AsyncClient, clock=time.monotonic):
        self._http = http
        self._clock = clock
        self._keys: Dict[str, Any] = {}
        self._fetched_at: Optional[float] = None
        self._last_fetch_failed = False
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get(self, kid: str) -> Any:
        async with self._lock:
            if self._needs_fetch(kid):
                try:
                    await self._fetch()
                    self._last_fetch_failed = False
                except Exception as exc:  # noqa: BLE001 -- fall back to the cache
                    self._last_fetch_failed = True
                    log.warning("apple keys fetch failed", extra={"error": type(exc).__name__})
            key = self._keys.get(kid)
            unavailable = key is None and self._last_fetch_failed
        if unavailable:
            raise AppleKeysUnavailable()
        if key is None:
            raise InvalidIdentityToken("unknown kid")
        return key

    def _needs_fetch(self, kid: str) -> bool:
        if self._fetched_at is None:
            return True
        age = self._clock() - self._fetched_at
        if kid not in self._keys:
            return age >= REFETCH_SECONDS
        return age >= KEYS_TTL_SECONDS

    async def _fetch(self) -> None:
        # Counted as a fetch even if it fails, so a down Apple endpoint is
        # retried once per REFETCH_SECONDS, not on every request.
        self._fetched_at = self._clock()
        r = await self._http.get(APPLE_KEYS_URL)
        r.raise_for_status()
        self._keys = {
            k["kid"]: RSAAlgorithm.from_jwk(json.dumps(k)) for k in r.json()["keys"]
        }


def nonce_hash(raw_nonce: str) -> str:
    """What the app puts in the request's nonce: SHA-256 hex of the raw one."""
    return hashlib.sha256(raw_nonce.encode()).hexdigest()


async def verify_identity_token(
    token: str, raw_nonce: str, keys: AppleKeys, bundle_id: str
) -> AppleIdentity:
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError:
        raise InvalidIdentityToken("malformed")
    if header.get("alg") != "RS256" or not header.get("kid"):
        raise InvalidIdentityToken("bad header")
    key = await keys.get(header["kid"])
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=bundle_id,
            issuer=APPLE_ISSUER,
            leeway=LEEWAY_SECONDS,
            options={"require": ["iss", "aud", "exp", "iat", "sub", "nonce"]},
        )
    except jwt.ExpiredSignatureError:
        raise InvalidIdentityToken("expired")
    except jwt.InvalidAudienceError:
        raise InvalidIdentityToken("audience")
    except jwt.InvalidIssuerError:
        raise InvalidIdentityToken("issuer")
    except jwt.PyJWTError as exc:
        raise InvalidIdentityToken(type(exc).__name__)
    if not hmac.compare_digest(str(claims["nonce"]), nonce_hash(raw_nonce)):
        raise InvalidIdentityToken("nonce")
    return AppleIdentity(sub=str(claims["sub"]))

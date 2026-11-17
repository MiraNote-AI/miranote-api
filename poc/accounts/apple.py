"""Sign in with Apple: verifying identity tokens (design 3.1) and revoking
the app's access at account deletion (design 7.1).

The app runs ASAuthorizationAppleIDProvider with SHA-256(nonce) as the
request nonce and sends us the identity token plus the raw nonce. The token
is an RS256 JWT signed with one of Apple's keys at APPLE_KEYS_URL.

Sign-in never exchanges the authorization code. Deletion re-runs Sign in
with Apple; AppleRevoker exchanges that fresh code for Apple's tokens and
revokes them at once, so no Apple refresh token is ever stored.
"""

from __future__ import annotations

import asyncio
import datetime as dt
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

import config

log = logging.getLogger("accounts.apple")

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"

# Apple rotates its keys rarely; a token with an unknown kid triggers a
# refetch, at most once per REFETCH_SECONDS so garbage tokens cannot make us
# hammer Apple.
KEYS_TTL_SECONDS = 24 * 3600
REFETCH_SECONDS = 60
# Clock skew between Apple, the phone and us.
LEEWAY_SECONDS = 60
# Deletion calls Apple inside the request: a few tries, then give up (7.1).
REVOKE_ATTEMPTS = 3
REVOKE_RETRY_SECONDS = 0.5
CLIENT_SECRET_TTL = dt.timedelta(minutes=5)


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


class RevokeNotConfigured(Exception):
    """No Sign in with Apple key (APPLE_SIWA_KEY_PATH, APPLE_SIWA_KEY_ID)."""


class RevokeFailed(Exception):
    """Apple refused the code or could not be reached after the retries.
    The message is a short reason for the log; never a code or token."""


class AppleRevoker:
    """Exchanges a fresh authorization code and revokes the tokens it gives
    (POST /auth/token, then /auth/revoke). The client secret is an ES256
    JWT signed with the Sign in with Apple key."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        key_pem: Optional[str],
        key_id: str,
        team_id: str,
        client_id: str,
    ):
        self._http = http
        self._key_pem = key_pem
        self._key_id = key_id
        self._team_id = team_id
        self._client_id = client_id

    async def aclose(self) -> None:
        await self._http.aclose()

    def _client_secret(self) -> str:
        now = dt.datetime.now(dt.timezone.utc)
        claims = {
            "iss": self._team_id,
            "iat": int(now.timestamp()),
            "exp": int((now + CLIENT_SECRET_TTL).timestamp()),
            "aud": APPLE_ISSUER,
            "sub": self._client_id,
        }
        return jwt.encode(claims, self._key_pem, algorithm="ES256", headers={"kid": self._key_id})

    async def revoke(self, authorization_code: str) -> None:
        if not self._key_pem or not self._key_id:
            raise RevokeNotConfigured()
        secret = self._client_secret()
        tokens = await self._post(
            APPLE_TOKEN_URL,
            {
                "client_id": self._client_id,
                "client_secret": secret,
                "code": authorization_code,
                "grant_type": "authorization_code",
            },
        )
        # Revoking the refresh token ends the user's whole link to the app.
        hint = "refresh_token" if tokens.get("refresh_token") else "access_token"
        token = tokens.get(hint)
        if not token:
            raise RevokeFailed("no token in exchange")
        await self._post(
            APPLE_REVOKE_URL,
            {"client_id": self._client_id, "client_secret": secret, "token": token, "token_type_hint": hint},
        )

    async def _post(self, url: str, data: Dict[str, str]) -> Dict[str, Any]:
        """Retries a network error or a 5xx; a 4xx (bad or used code) is
        final."""
        for attempt in range(1, REVOKE_ATTEMPTS + 1):
            try:
                r = await self._http.post(url, data=data)
            except httpx.HTTPError:
                reason = "unreachable"
            else:
                if r.status_code == 200:
                    return r.json() if r.content else {}
                if r.status_code < 500:
                    # Apple's body is {"error": "invalid_grant"} and the like.
                    raise RevokeFailed("rejected {}".format(r.status_code))
                reason = "status {}".format(r.status_code)
            if attempt < REVOKE_ATTEMPTS:
                await asyncio.sleep(REVOKE_RETRY_SECONDS)
        raise RevokeFailed(reason)


def revoker_from_config(http: httpx.AsyncClient) -> AppleRevoker:
    key_pem = None
    if config.APPLE_SIWA_KEY_PATH:
        with open(config.APPLE_SIWA_KEY_PATH) as f:
            key_pem = f.read()
    else:
        log.warning("apple revoke not configured: APPLE_SIWA_KEY_PATH unset")
    return AppleRevoker(http, key_pem, config.APPLE_SIWA_KEY_ID, config.APPLE_TEAM_ID, config.APPLE_BUNDLE_ID)

"""Per-user auth and quotas for the AI services (api #95 step 3, design 6.1).

Replaces beta_auth.install on every AI service. A request carries
`Authorization: Bearer <token>`, where the token is either

- our access JWT from the accounts service: verified here against the
  accounts JWKS (fetched once, cached; no per-request call for auth), then
  a metered route reserves one unit with the accounts service before the
  handler runs and settles it after: `ok` on a 2xx answer, `failed` (not
  charged) on anything else; or
- the shared beta token (beta_auth), accepted while ACCEPT_BETA_TOKENS is on
  (default) so TestFlight builds without sign-in keep working. Not metered;
  beta_auth's rate limit still applies. Paths in `beta_always` take it even
  with the flag off (/mood, until App Attest lands: design 6.5).

Fail closed: when the accounts service cannot be reached for a metered
route (or its keys were never fetched), the answer is 503
{"reason": "accounts_unavailable"}, never an unmetered call.

Answers the app can act on, all JSON with a `reason`:
    401 missing_token | invalid_token | token_expired | unknown_user
    409 duplicate_request  (an X-Request-ID already counted; from accounts)
    429 quota  {"reason", "feature", "resets_at", ...}   (from accounts)
    503 accounts_unavailable

The client sends X-Request-ID (a UUID) per user action. A retry with the
id of a failed call runs again; with the id of one still pending or done it
is refused (409), since no result is stored (design 6.1). Without one, a
fresh id is made here.

Environment: ACCOUNTS_URL (default http://127.0.0.1:8007),
ACCOUNTS_INTERNAL_SECRET (same value as the accounts service), and
ACCEPT_BETA_TOKENS (1/0). Like BETA_TOKENS they live in the repo root .env,
which beta_auth loads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import httpx
from fastapi.responses import JSONResponse

import beta_auth

try:
    import jwt
    from jwt.algorithms import ECAlgorithm
except ImportError:
    # PyJWT[crypto] is new in every service's requirements.txt. A venv that
    # has not been reinstalled since must still serve the beta token, so
    # only signed-in calls need it; they answer 503 until it is installed.
    jwt = None  # type: ignore[assignment]

log = logging.getLogger("ai_auth")

ISSUER = "miranote-accounts"
AUDIENCE = "miranote"
JWKS_TTL_SECONDS = 3600
# An unknown kid (a key rotation) refetches the JWKS, at most this often.
JWKS_REFETCH_SECONDS = 60
ACCOUNTS_TIMEOUT_SECONDS = 5.0

# Rough list-price cost of one unit, for ai_usage.cost_usd (design 6.3).
# Estimates to tune from the real bills, not invoices.
COST_USD = {
    "chat": 0.002,
    "expand": 0.001,
    "image_generate": 0.04,
    "image_edit": 0.04,
    "photo_push": 0.01,
    "mood": 0.0005,
}

# Tests swap in an httpx transport (a fake accounts service, or the real one
# through its TestClient).
TRANSPORT: Optional[httpx.BaseTransport] = None


def accounts_url() -> str:
    return (os.environ.get("ACCOUNTS_URL") or "http://127.0.0.1:8007").rstrip("/")


def accept_beta_tokens() -> bool:
    return (os.environ.get("ACCEPT_BETA_TOKENS") or "1").strip().lower() not in ("0", "false", "no", "off")


def _client() -> httpx.Client:
    return httpx.Client(base_url=accounts_url(), timeout=ACCOUNTS_TIMEOUT_SECONDS, transport=TRANSPORT)


class AccountsUnavailable(Exception):
    """The accounts service did not answer, or answered something unusable."""


class _Jwks:
    """Our public keys by kid, fetched from the accounts service."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._keys: Dict[str, Any] = {}
        self._fetched_at: Optional[float] = None
        self._failed_at: Optional[float] = None

    def _fetch(self) -> None:
        try:
            with _client() as client:
                response = client.get("/.well-known/jwks.json")
                response.raise_for_status()
                keys = {k["kid"]: ECAlgorithm.from_jwk(k) for k in response.json()["keys"]}
        except Exception as exc:  # noqa: BLE001 -- any failure means "no keys"
            raise AccountsUnavailable(type(exc).__name__)
        self._keys, self._fetched_at = keys, self._clock()

    def key(self, kid: str) -> Optional[Any]:
        """The key for `kid`, or None when the accounts service does not
        know it. Raises AccountsUnavailable when there are no keys at all."""
        with self._lock:
            now = self._clock()
            stale = self._fetched_at is None or now - self._fetched_at > JWKS_TTL_SECONDS
            unknown = kid not in self._keys and (
                self._fetched_at is None or now - self._fetched_at > JWKS_REFETCH_SECONDS
            )
            # After a failed refresh the keys in hand serve until the next
            # try, a minute on: retrying per request would put a network hop
            # (up to the timeout, under this lock) in front of every call.
            backing_off = (
                bool(self._keys) and self._failed_at is not None and now - self._failed_at <= JWKS_REFETCH_SECONDS
            )
            if (stale or unknown) and not backing_off:
                try:
                    self._fetch()
                    self._failed_at = None
                except AccountsUnavailable:
                    # Keys we already have stay good; none at all is a 503.
                    if not self._keys:
                        raise
                    self._failed_at = now
                    log.warning("jwks refresh failed, using cached keys")
            return self._keys.get(kid)


JWKS = _Jwks()


class InvalidToken(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def verify_access_token(token: str) -> uuid.UUID:
    """The user id in one of our valid access tokens."""
    if jwt is None:
        raise AccountsUnavailable("PyJWT[crypto] is not installed")
    try:
        kid = jwt.get_unverified_header(token).get("kid")
    except jwt.PyJWTError:
        raise InvalidToken("invalid_token")
    key = JWKS.key(kid) if isinstance(kid, str) else None
    if key is None:
        raise InvalidToken("invalid_token")
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["ES256"],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["iss", "aud", "exp", "sub"]},
        )
        return uuid.UUID(claims["sub"])
    except jwt.ExpiredSignatureError:
        raise InvalidToken("token_expired")
    except (jwt.PyJWTError, ValueError):
        raise InvalidToken("invalid_token")


def _internal_headers() -> Dict[str, str]:
    secret = os.environ.get("ACCOUNTS_INTERNAL_SECRET", "").strip()
    if not secret:
        raise AccountsUnavailable("ACCOUNTS_INTERNAL_SECRET is not set")
    return {"X-Internal-Secret": secret}


def reserve(user_id: uuid.UUID, feature: str, request_id: uuid.UUID, model: str) -> httpx.Response:
    """The accounts service's answer: 200 reserved, 401 unknown user, 409
    duplicate or conflicting request_id, 429 quota. Raises
    AccountsUnavailable for anything else."""
    try:
        with _client() as client:
            response = client.post(
                "/usage/reserve",
                headers=_internal_headers(),
                json={"user_id": str(user_id), "feature": feature, "request_id": str(request_id), "model": model},
            )
    except httpx.HTTPError as exc:
        raise AccountsUnavailable(type(exc).__name__)
    if response.status_code not in (200, 401, 409, 429):
        raise AccountsUnavailable("reserve answered {}".format(response.status_code))
    return response


def settle(request_id: uuid.UUID, feature: str, ok: bool) -> None:
    """Never raises: a settle that does not arrive leaves the row pending,
    which still counts -- the safe side for cost."""
    body = {"request_id": str(request_id), "status": "ok" if ok else "failed", "cost_usd": COST_USD[feature] if ok else 0}
    try:
        with _client() as client:
            response = client.post("/usage/settle", headers=_internal_headers(), json=body)
        if response.status_code != 200:
            log.warning("settle answered %s", response.status_code)
    except (httpx.HTTPError, AccountsUnavailable) as exc:
        log.warning("settle failed: %s", type(exc).__name__)


@dataclass(frozen=True)
class Metered:
    """A route that costs money. `applies` narrows it by request (e.g.
    /border only in its model-backed mode)."""

    feature: str
    model: str = ""
    applies: Optional[Callable[[Any], bool]] = None


class Ticket:
    """One reserved unit. A handler whose paid work outlives the response
    (a background job) takes it with take_ticket() and settles it itself."""

    def __init__(self, request_id: uuid.UUID, feature: str):
        self.request_id = request_id
        self.feature = feature
        self.taken = False

    def settle(self, ok: bool) -> None:
        settle(self.request_id, self.feature, ok)


def take_ticket(request) -> Optional[Ticket]:
    """For a handler that settles later: the request's ticket, or None when
    the call is not metered (beta token). The gate then settles only a
    non-2xx answer (failed) and leaves a 2xx to the handler."""
    ticket = getattr(request.state, "ai_ticket", None)
    if ticket is not None:
        ticket.taken = True
    return ticket


def _reject(status: int, reason: str, detail: str, **headers: str) -> JSONResponse:
    # `detail` too: builds before sign-in show it, and read nothing else.
    return JSONResponse({"reason": reason, "detail": detail}, status_code=status, headers=headers or None)


def _request_id(header: Optional[str]) -> uuid.UUID:
    try:
        return uuid.UUID(header or "")
    except ValueError:
        return uuid.uuid4()


def install(
    app,
    metered: Optional[Mapping[str, Metered]] = None,
    beta_always: Iterable[str] = (),
) -> None:
    """Gate every route of `app` except /health. `metered` maps a path to
    its feature. Call it where beta_auth.install was called: before
    CORSMiddleware, so CORS stays outermost (see beta_auth.install)."""
    metered = dict(metered or {})
    beta_always = frozenset(beta_always)

    @app.middleware("http")
    async def _ai_gate(request, call_next):
        path = request.url.path
        if path in beta_auth.EXEMPT_PATHS:
            return await call_next(request)

        token = beta_auth._bearer_token(request.headers.get("Authorization"))
        if token is None:
            return _reject(401, "missing_token", "missing access token", **{"WWW-Authenticate": "Bearer"})

        if token in beta_auth.beta_tokens():
            if not (accept_beta_tokens() or path in beta_always):
                return _reject(401, "invalid_token", "the beta token is no longer accepted; sign in")
            if not beta_auth._limiter.allow(token):
                return JSONResponse({"detail": "beta rate limit exceeded"}, status_code=429)
            return await call_next(request)

        try:
            user_id = await asyncio.to_thread(verify_access_token, token)
        except InvalidToken as exc:
            return _reject(401, exc.reason, "invalid access token", **{"WWW-Authenticate": "Bearer"})
        except AccountsUnavailable:
            log.warning("jwks unavailable")
            return _reject(503, "accounts_unavailable", "sign-in service unavailable, try again shortly")

        spec = metered.get(path)
        if spec is None or request.method != "POST" or (spec.applies is not None and not spec.applies(request)):
            return await call_next(request)

        request_id = _request_id(request.headers.get("X-Request-ID"))
        try:
            answer = await asyncio.to_thread(reserve, user_id, spec.feature, request_id, spec.model)
        except AccountsUnavailable as exc:
            log.warning("reserve unavailable: %s", exc)
            return _reject(503, "accounts_unavailable", "sign-in service unavailable, try again shortly")
        if answer.status_code == 401:
            return _reject(401, "unknown_user", "this account no longer exists; sign in again")
        if answer.status_code in (409, 429):
            return JSONResponse(answer.json(), status_code=answer.status_code)

        ticket = Ticket(request_id, spec.feature)
        request.state.ai_ticket = ticket
        try:
            response = await call_next(request)
        except Exception:
            await asyncio.to_thread(ticket.settle, False)
            raise
        ok = 200 <= response.status_code < 300
        if not (ok and ticket.taken):
            await asyncio.to_thread(ticket.settle, ok)
        return response

    if jwt is None:
        print("[ai_auth] WARNING: PyJWT[crypto] is not installed -- signed-in calls will "
              "answer 503; run pip install -r requirements.txt")
    if not os.environ.get("ACCOUNTS_INTERNAL_SECRET", "").strip() and metered:
        print("[ai_auth] WARNING: ACCOUNTS_INTERNAL_SECRET is empty -- signed-in calls to "
              "{} will answer 503".format(sorted(metered)))
    print("[ai_auth] gate active: user tokens via {}, beta tokens {}".format(
        accounts_url(), "accepted" if accept_beta_tokens() else "refused"))

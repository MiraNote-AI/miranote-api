"""Bearer-token auth and per-token rate limiting for the beta deployment.

The four POC services are reachable over public HTTPS through a Cloudflare
tunnel, so they need a gate. The gate is deliberately thin: one shared token,
no user accounts, no session state.

Two properties carry the design.

The token list is plural. Rotation with a single token is all-or-nothing --
every tester is cut off the moment it changes. With a comma-separated list a
new token is added first, builds go out, and the old one is dropped afterwards.

The token is not a secret. It ships inside a TestFlight build and can be
extracted from the IPA, so the rate limit is the real protection: it bounds how
fast an extracted token can spend Vertex and DeepSeek credits, and it blunts
retry storms from timed-out clients.

This module lives at the repository root and is reached by exporting
PYTHONPATH there. It is deliberately not inside a package named "shared":
poc/image-generation already owns that name locally, and its working directory
sorts ahead of PYTHONPATH on sys.path, so "shared.beta_auth" would resolve to
the POC's own package and fail to import.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Loaded by absolute path, not by search. Each service runs with its own POC
# directory as the working directory, and a bare load_dotenv() from there does
# not reach the repository root -- measured, not assumed. The shared beta token
# therefore lives in one file next to this module instead of being copied into
# all four POC .env files. Existing environment variables still win, so a POC
# can override it locally.
load_dotenv(pathlib.Path(__file__).with_name(".env"))

TOKEN_ENV_VAR = "BETA_TOKENS"
# Shared by every tester, not per person: TestFlight ships one binary, so one
# token goes to all of them. Ten testers holding a conversation is the load
# this absorbs, at one request per chat message.
#
# Raising it costs little, because it was never what protected the expensive
# path. /generate is capped at three concurrent by its own semaphore and each
# takes roughly 30s, so at most about six a minute complete whatever this says.
# What this limit actually restricts is chat and the text endpoints, which are
# the cheap ones. It still bounds an extracted token, which is its purpose.
RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_SECONDS = 60

# /health is exempt because all four services expose it and
# scripts/start_backends.sh polls it for readiness. Requiring a token there
# would break the startup check and buy nothing: it returns no user data.
EXEMPT_PATHS = frozenset({"/health"})

_BEARER_PREFIX = "bearer "


def beta_tokens() -> List[str]:
    """The currently accepted tokens, newest first is not significant.

    Read per request rather than cached at import, so a token can be added or
    revoked by editing the environment and restarting only the service.
    """
    raw = os.environ.get(TOKEN_ENV_VAR, "")
    return [token.strip() for token in raw.split(",") if token.strip()]


class TokenRateLimiter:
    """A sliding window of request timestamps, kept per token.

    A fixed counter reset every 60s would let a caller spend its whole budget
    at the end of one window and again at the start of the next, so the real
    burst is twice the configured limit. The window slides instead.
    """

    def __init__(
        self,
        limit: int = RATE_LIMIT_REQUESTS,
        window: int = RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._limit = limit
        self._window = window
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = {}

    def allow(self, token: str) -> bool:
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            hits = self._hits.setdefault(token, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True


_limiter = TokenRateLimiter()


def _bearer_token(header: Optional[str]) -> Optional[str]:
    if not header or not header.lower().startswith(_BEARER_PREFIX):
        return None
    return header[len(_BEARER_PREFIX) :].strip() or None


def _denial(path: str, authorization: Optional[str]):
    """The gate. Returns the response to send back, or None to let through."""
    if path in EXEMPT_PATHS:
        return None

    token = _bearer_token(authorization)
    if token is None or token not in beta_tokens():
        return JSONResponse(
            {"detail": "missing or invalid beta token"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _limiter.allow(token):
        return JSONResponse({"detail": "beta rate limit exceeded"}, status_code=429)

    return None


def install(app) -> None:
    """Require a beta token for everything this app serves.

    Middleware rather than a route dependency, and not as a matter of taste.
    A mounted sub-application does not inherit FastAPI(dependencies=[...]):
    measured, the route answers 401 while the mounted file answers 200 and
    hands over its contents. voice-to-text mounts StaticFiles at "/" with
    html=True, which catches every unrouted path, so a dependency would leave
    it open to anyone who finds the hostname.

    The gate returns a response instead of raising HTTPException for a related
    reason: middleware runs outside the exception handlers, so a raise turns
    every rejection into a 500.

    Call this BEFORE adding CORSMiddleware. Starlette makes the most recently
    added middleware the outermost, and CORS has to stay outside the gate so
    it can answer a browser preflight, which carries no Authorization header
    and would otherwise be rejected with 401.
    """

    @app.middleware("http")
    async def _beta_gate(request, call_next):
        denial = _denial(request.url.path, request.headers.get("Authorization"))
        if denial is not None:
            return denial
        return await call_next(request)

    # Announced at startup because the failure it guards against is otherwise
    # silent: with no tokens configured every request is rejected and nothing
    # says why.
    tokens = beta_tokens()
    if tokens:
        print("[beta_auth] gate active, {} token(s) configured".format(len(tokens)))
    else:
        print(
            "[beta_auth] WARNING: BETA_TOKENS is empty -- every request except "
            "{} will be rejected with 401".format(sorted(EXEMPT_PATHS))
        )

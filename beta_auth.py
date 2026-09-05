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
from fastapi import HTTPException, Request

# Loaded by absolute path, not by search. Each service runs with its own POC
# directory as the working directory, and a bare load_dotenv() from there does
# not reach the repository root -- measured, not assumed. The shared beta token
# therefore lives in one file next to this module instead of being copied into
# all four POC .env files. Existing environment variables still win, so a POC
# can override it locally.
load_dotenv(pathlib.Path(__file__).with_name(".env"))

TOKEN_ENV_VAR = "BETA_TOKENS"
RATE_LIMIT_REQUESTS = 30
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


def require_beta_token(request: Request) -> Optional[str]:
    """FastAPI dependency: reject anything without a live, unexhausted token."""
    if request.url.path in EXEMPT_PATHS:
        return None

    token = _bearer_token(request.headers.get("Authorization"))
    if token is None or token not in beta_tokens():
        raise HTTPException(
            status_code=401,
            detail="missing or invalid beta token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _limiter.allow(token):
        raise HTTPException(status_code=429, detail="beta rate limit exceeded")

    return token

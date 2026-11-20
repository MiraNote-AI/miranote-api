"""
MiraNote -- Accounts API (port 8007)

Users, sessions, subscriptions and the AI usage ledger (design
docs/specs/2026-11-06-accounts-and-subscriptions-design.md). Steps 1-2
of api #95: the schema, the service around it, and Sign in with Apple with
our own tokens (auth.py). Step 3: entitlements and the AI usage ledger
(usage.py, caps in quotas.py) and the daily cost budgets (budget.py).
Step 5: subscription sync and App Store
Server Notifications (subscriptions.py, Apple's side in appstore.py).
Step 6: account deletion (DELETE /me in auth.py) and the purge and
retention jobs (purge.py), run here a minute after start, then every
hour. Step 8: taste events and their switch (taste.py), off unless
TASTE_EVENTS_ENABLED=1.

    GET /health -> {"status", "db"}; 503 when the database does not answer
    /auth/*, /.well-known/jwks.json, /me, DELETE /me -> see auth.py
    /usage/reserve, /usage/settle (internal), /me/usage -> see usage.py
    /subscriptions/sync, /appstore/notifications -> see subscriptions.py
    /events, /me/preferences -> see taste.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import apple
import appstore
import auth
import budget
import config
import db
import purge
import subscriptions
import taste
import tokens
import usage

log = logging.getLogger("accounts")

# Attributes every LogRecord has; anything else was passed through `extra`.
_RECORD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts (UTC), level, logger, msg and the
    `extra` fields."""

    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RECORD_FIELDS:
                entry[key] = value
        if record.exc_info:
            # The type only: exception messages can carry SQL parameters.
            entry["error"] = record.exc_info[0].__name__
        return json.dumps(entry, default=str)


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("accounts")
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    root.propagate = False


_setup_logging()


# Tests swap in an httpx.MockTransport serving a fake Apple (keys, token
# and revoke endpoints).
APPLE_TRANSPORT = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the engine and probe the database once, so the log says at
    startup whether it is reachable. The service still starts when it is not
    (Neon may be waking up); /health answers 503 until it is."""
    app.state.internal_secret = config.internal_secret()
    app.state.budgets = budget.from_env()
    app.state.taste_events_enabled = config.taste_events_enabled()
    # {} = no budget; so the log shows whether the kill switch is armed.
    log.info("daily budgets", extra={"budgets": {k: str(v) for k, v in sorted(app.state.budgets.items())}})
    app.state.signer = tokens.Signer(config.signing_key_pem(), config.previous_public_key_pem())
    app.state.engine = db.create_engine(config.database_url())
    app.state.apple_keys = apple.AppleKeys(httpx.AsyncClient(timeout=5, transport=APPLE_TRANSPORT))
    app.state.apple_revoker = apple.revoker_from_config(httpx.AsyncClient(timeout=5, transport=APPLE_TRANSPORT))
    app.state.appstore = appstore.from_config()
    ok = await db.ping(app.state.engine, config.HEALTH_DB_TIMEOUT_SECONDS)
    log.info("startup", extra={"db": "ok" if ok else "unreachable"})
    purge_task = asyncio.create_task(purge.run_forever(app.state.engine))
    try:
        yield
    finally:
        purge_task.cancel()
        await app.state.engine.dispose()
        await app.state.apple_revoker.aclose()
        await app.state.apple_keys.aclose()
        await app.state.appstore.aclose()


# No /docs, /redoc or /openapi.json: there is no beta gate in front of this
# service, and the other services serve theirs only behind it.
app = FastAPI(
    title="MiraNote Accounts",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# No beta gate: every route proves its caller itself (an Apple identity
# token, one of our refresh tokens, current_user, or Apple's signature on
# a notification), and the shared beta token would collide with our own
# Bearer tokens. The internal /usage routes check a shared secret instead
# (usage.py). Starlette makes the
# last-added middleware the outermost: CORS -> request log -> route.
app.include_router(auth.router)
app.include_router(usage.router)
app.include_router(subscriptions.router)
app.include_router(taste.router)


# Logs method, path, status and time only. Never the body, the headers or
# the query string: they carry identity tokens, refresh tokens and Apple's
# signed payloads. uvicorn's own access log (which prints the query string)
# is turned off in run.sh for the same reason.
@app.middleware("http")
async def _log_request(request: Request, call_next):
    started = time.monotonic()
    response = await call_next(request)
    log.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.monotonic() - started) * 1000),
        },
    )
    return response


_CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGIN", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    ok = await db.ping(app.state.engine, config.HEALTH_DB_TIMEOUT_SECONDS)
    if ok:
        return {"status": "ok", "db": "ok"}
    return JSONResponse({"status": "error", "db": "unreachable"}, status_code=503)

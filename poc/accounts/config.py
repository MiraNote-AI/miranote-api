"""Configuration for the accounts service.

Everything comes from the environment. poc/accounts/.env is loaded by
absolute path so the service finds it whatever the working directory is;
existing environment variables win.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent

load_dotenv(HERE / ".env")

# How long /health waits for `SELECT 1`. A Neon Free compute suspended after
# 5 minutes idle answers its first query 0.3-0.5 s late (design 8.2).
HEALTH_DB_TIMEOUT_SECONDS = float(os.getenv("ACCOUNTS_HEALTH_DB_TIMEOUT") or 5)


def database_url() -> str:
    """The Postgres URL. Required, no default: a service that silently fell
    back to some other database would be worse than one that does not start.

    Read per call, not at import, so tests and Alembic see the environment
    they set up.
    """
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set -- put it in poc/accounts/.env (see README.md)"
        )
    return url


# Sign in with Apple (design 3.1). The identity token's `aud` is the app's
# bundle id; `sub` is scoped to the developer team, so users are keyed by
# (team, sub) and the team id is stored with every user (design 3.4).
APPLE_BUNDLE_ID = os.getenv("APPLE_BUNDLE_ID") or "ai.miranote.app"
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID") or "W7U365VR4U"


def signing_key_pem() -> str:
    """PEM of the EC P-256 private key that signs our access tokens.
    Required, like DATABASE_URL: a service that made up its own key would
    sign tokens nothing else can verify after a restart."""
    pem = os.environ.get("ACCOUNTS_SIGNING_KEY", "").strip()
    if not pem:
        raise RuntimeError(
            "ACCOUNTS_SIGNING_KEY is not set -- put it in poc/accounts/.env (see README.md)"
        )
    return pem


def previous_public_key_pem() -> str:
    """PEM of the public key that signed tokens before the last rotation, or
    "". It stays in the JWKS and is still accepted for one access-token
    lifetime after a rotation (design 3.2)."""
    return os.environ.get("ACCOUNTS_PREVIOUS_PUBLIC_KEY", "").strip()

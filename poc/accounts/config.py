"""Configuration for the accounts service.

Everything comes from the environment. poc/accounts/.env is loaded by
absolute path so the service finds it whatever the working directory is;
existing environment variables win.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

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


def internal_secret() -> str:
    """Shared secret the AI services send in X-Internal-Secret to
    /usage/reserve and /usage/settle. Required: without it the AI services
    could not meter, and an empty secret must never match."""
    secret = os.environ.get("ACCOUNTS_INTERNAL_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "ACCOUNTS_INTERNAL_SECRET is not set -- put it in poc/accounts/.env (see README.md)"
        )
    return secret


# App Store (design 5). All optional: without them the service still starts
# and /subscriptions/sync and /appstore/notifications answer 503.
#
# Apple's root certificates (DER, e.g. AppleRootCA-G3.cer from
# https://www.apple.com/certificateauthority/), comma-separated paths. Every
# signed transaction, renewal info and notification must chain to one.
# APPSTORE_APP_APPLE_ID is the app's numeric Apple ID in App Store Connect;
# Production notifications must carry it.
def appstore_root_cert_paths() -> List[str]:
    return [p.strip() for p in os.getenv("APPSTORE_ROOT_CERTS", "").split(",") if p.strip()]


def appstore_app_apple_id() -> Optional[int]:
    value = os.getenv("APPSTORE_APP_APPLE_ID", "").strip()
    return int(value) if value else None


# OCSP revocation checks of the signing chain, against Apple. Off only in
# the tests, whose locally made certificates have no OCSP responder.
def appstore_online_checks() -> bool:
    return os.getenv("APPSTORE_ONLINE_CHECKS", "1").strip() != "0"


# The In-App Purchase key (.p8) for the App Store Server API: Set App Account
# Token at sync (5.3) and the backfill command (5.5). Without it a sync still
# links the chain, but Apple keeps the old token until a support fix.
APPSTORE_KEY_PATH = os.getenv("APPSTORE_KEY_PATH", "").strip()
APPSTORE_KEY_ID = os.getenv("APPSTORE_KEY_ID", "").strip()
APPSTORE_ISSUER_ID = os.getenv("APPSTORE_ISSUER_ID", "").strip()

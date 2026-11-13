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

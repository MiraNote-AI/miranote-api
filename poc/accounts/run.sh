#!/bin/bash
# Run the accounts service on loopback for the Cloudflare tunnel
# (beta-accounts.miranote.app -> 127.0.0.1:8007). Same posture as
# scripts/start_backends.sh: loopback only, the tunnel is the only way in.
#
# Needs poc/accounts/.venv and DATABASE_URL (poc/accounts/.env or the
# environment); see README.md. Migrations are a separate, explicit step:
#   .venv/bin/alembic upgrade head
#
# Usage: poc/accounts/run.sh            (ACCOUNTS_PORT overrides 8007)
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
API_ROOT="$(cd "$HERE/../.." && pwd)"
PORT="${ACCOUNTS_PORT:-8007}"

cd "$HERE"
if [ ! -x .venv/bin/uvicorn ]; then
  echo "missing poc/accounts/.venv -- see poc/accounts/README.md" >&2
  exit 1
fi

# beta_auth lives at the repo root.
export PYTHONPATH="$API_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --no-access-log: uvicorn's access log prints the query string; main.py
# logs every request itself, without query, headers or body.
exec .venv/bin/uvicorn main:app --host 127.0.0.1 --port "$PORT" --no-access-log

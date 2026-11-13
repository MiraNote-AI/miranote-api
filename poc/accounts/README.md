# accounts (port 8007)

Users, sessions, subscriptions and the AI usage ledger for the paid flow
(api #95). Design: `docs/specs/2026-11-06-accounts-and-subscriptions-design.md`
(#99). Step 1: the tier 1 schema (section 4.1) as Alembic migrations, and
a FastAPI service with `/health`. Step 2: Sign in with Apple and our own
tokens (section 3). Subscriptions and quotas come in later steps.

Stack: FastAPI, SQLAlchemy 2 (async) + asyncpg, Alembic, PostgreSQL 17.
Only standard Postgres, written for a transaction-mode pooler (no statement
cache, no session-level `SET`), so Neon's pooled URL works as is.

| table | what |
|---|---|
| `users` | one row per Apple `sub` + `apple_team_id` |
| `sessions` | our refresh tokens (hashed) |
| `subscriptions` | one row per Apple renewal chain |
| `app_store_notifications` | App Store notifications, audit and idempotency |
| `ai_usage` | one row per AI request; the only source of quotas |
| `app_attest_keys` | App Attest device keys, not linked to a user |

## Setup

```bash
cd poc/accounts
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # fill DATABASE_URL and ACCOUNTS_SIGNING_KEY
```

`DATABASE_URL` is required; there is no default. Neon's pooled connection
string can be pasted as is (`?sslmode=require&channel_binding=require`):
`db.py` maps `sslmode` to asyncpg's `ssl` and drops `channel_binding`,
which asyncpg does not support. `.env` is gitignored; never commit it.

`ACCOUNTS_SIGNING_KEY` is required too: the EC P-256 private key (PEM)
that signs our access tokens (see Keys below). There is no beta-token
gate on this service: every route checks its own credential.

## Sign-in API

| route | body | answer |
|---|---|---|
| `POST /auth/apple` | `{"identity_token", "nonce"}` | token pair; 401; or 503 when Apple's keys cannot be fetched (retry) |
| `POST /auth/refresh` | `{"refresh_token"}` | new token pair, or 401 (sign in again) |
| `POST /auth/signout` | `{"refresh_token"}` | 204, always |
| `GET /.well-known/jwks.json` | | our public signing keys |
| `GET /me` | `Authorization: Bearer <access>` | `{"id"}`, or 401 |

A token pair is `{"access_token", "refresh_token", "token_type": "Bearer",
"expires_in": 3600, "user_id"}`.

- `/auth/apple` verifies Apple's identity token against
  `https://appleid.apple.com/auth/keys` (cached 24 h; an unknown `kid`
  refetches at most once a minute): RS256, `iss`, `aud` =
  `APPLE_BUNDLE_ID` (default `ai.miranote.app`), `exp`, and
  `nonce` = SHA-256 hex of the raw `nonce` the app sends. Users are keyed
  by (`APPLE_TEAM_ID`, `sub`); the default team is `W7U365VR4U`. The
  authorization code is not exchanged, no Apple refresh token is stored,
  no email or name either. Signing in again while the account is marked
  for deletion purges it first and creates a new, empty account (design
  7.1): its sessions go, its subscriptions and usage stay with no owner.
- Access tokens: ES256 JWT, 1 h, `kid` header, `iss` `miranote-accounts`,
  `aud` `miranote`, `sub` = our user id. `current_user` in `auth.py` is the
  dependency later routes use.
- Refresh tokens: opaque, 60 days, only the SHA-256 is stored in
  `sessions`, rotated on every use. A replaced token used again within
  10 s gets a fresh pair (two refreshes in parallel at launch); after
  that it revokes the whole sign-in (every token of that family).
- Sign-out revokes the sign-in the refresh token belongs to. Access
  tokens already issued stay valid until they expire (at most 1 h).

## Keys

The kid is the RFC 7638 thumbprint of the public key, so there is nothing
else to configure. Make a key:

```bash
openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt
```

To rotate: put the new private key in `ACCOUNTS_SIGNING_KEY`, the old
key's public PEM (`openssl ec -pubout` on the old key) in
`ACCOUNTS_PREVIOUS_PUBLIC_KEY`, restart; after one hour (one access-token
lifetime) remove `ACCOUNTS_PREVIOUS_PUBLIC_KEY` and restart again.

## Migrations

From `poc/accounts`, against the database in `DATABASE_URL`:

```bash
.venv/bin/alembic upgrade head
.venv/bin/alembic current
```

Migrations are never run by the service itself. Try a new migration on a
Neon branch (for example `dev`) before running it on `production`.

## Run

```bash
poc/accounts/run.sh
```

Binds `127.0.0.1:8007` only (`ACCOUNTS_PORT` overrides the port), with
uvicorn's access log off. The service logs one JSON line per request to
stdout: method, path, status and time -- never the body, headers or query
string.

`GET /health` (no token) runs `SELECT 1`: `200 {"status": "ok", "db": "ok"}`,
or `503 {"status": "error", "db": "unreachable"}`. The service starts even
when the database does not answer (a suspended Neon compute wakes up on the
first query); `/health` says so until it does.

## Deploy behind the Cloudflare tunnel

On the Mac that runs the beta services (any user; paths below are relative
to the repo checkout):

1. Pull, then do Setup above and `alembic upgrade head`.
2. Start it: `poc/accounts/run.sh` (or under launchd, see
   `scripts/launchd/README.md`; it is not in `start_backends.sh` yet).
3. In the beta tunnel's cloudflared config (the file the tunnel service
   runs with; see `scripts/start_tunnel.sh`) add an ingress rule before the
   final 404 rule:

   ```yaml
     - hostname: beta-accounts.miranote.app
       service: http://127.0.0.1:8007
   ```

   `127.0.0.1`, not `localhost`: `localhost` can resolve to `::1` and
   answer 502.
4. Route the hostname to the tunnel with the MiraNote Cloudflare account's
   certificate:
   `cloudflared tunnel route dns <tunnel name> beta-accounts.miranote.app`.
5. Restart the tunnel and check
   `https://beta-accounts.miranote.app/health` -> `{"status": "ok", "db": "ok"}`.

## Tests

Every test runs against a real Postgres 17 (design 8.1: never SQLite). The
tests drop and recreate every table, so they read
`ACCOUNTS_TEST_DATABASE_URL`, never `DATABASE_URL`: point it at a
throwaway database only.

```bash
docker run -d --name accounts-pg -p 55432:5432 \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=accounts_test postgres:17
export ACCOUNTS_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:55432/accounts_test'
cd poc/accounts
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

(Homebrew works too: `brew install postgresql@17`, start it, `createdb
accounts_test`, and use that URL.) CI runs the same suite with a
`postgres:17` service container: `.github/workflows/accounts.yml`. Apple
is faked with a locally generated key served as a JWKS through
`httpx.MockTransport`: the tests never reach the network beyond Postgres.

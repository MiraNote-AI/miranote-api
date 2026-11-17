# accounts (port 8007)

Users, sessions, subscriptions and the AI usage ledger for the paid flow
(api #95). Design: `docs/specs/2026-11-06-accounts-and-subscriptions-design.md`
(#99). Step 1: the tier 1 schema (section 4.1) as Alembic migrations, and
a FastAPI service with `/health`. Step 2: Sign in with Apple and our own
tokens (section 3). Step 3: entitlement, the AI usage ledger and its caps
(section 6), used by the AI services through `ai_auth.py` at the repo root.
Step 5: subscription sync from the app and App Store Server Notifications
V2 (section 5). Step 6: account deletion, the purge and retention jobs
(section 7).

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
cp .env.example .env        # fill DATABASE_URL, ACCOUNTS_SIGNING_KEY, ACCOUNTS_INTERNAL_SECRET
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
| `DELETE /me` | `Authorization: Bearer <access>`, `{"authorization_code"}` | see Account deletion |

A token pair is `{"access_token", "refresh_token", "token_type": "Bearer",
"expires_in": 3600, "user_id"}`.

- `/auth/apple` verifies Apple's identity token against
  `https://appleid.apple.com/auth/keys` (cached 24 h; an unknown `kid`
  refetches at most once a minute): RS256, `iss`, `aud` =
  `APPLE_BUNDLE_ID` (default `ai.miranote.app`), `exp`, and
  `nonce` = SHA-256 hex of the raw `nonce` the app sends. Users are keyed
  by (`APPLE_TEAM_ID`, `sub`); the default team is `W7U365VR4U`. The
  authorization code is not exchanged at sign-in, no Apple refresh token
  is ever stored, no email or name either. Signing in again while the account is marked
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

## Account deletion

`DELETE /me` (design 7.1). Before calling it the app runs Sign in with
Apple again and sends the fresh `authorization_code` (single-use, valid
5 minutes). In one request the server:

1. sets `users.deletion_requested_at` and revokes every session of the
   user: from now on `/me`, `/auth/refresh` and the AI guard answer 401;
2. exchanges the code at Apple (`/auth/token`) and revokes the refresh
   token it gets at once (`/auth/revoke`), with a client secret signed by
   the Sign in with Apple key. Network errors and 5xx are tried 3 times;
   nothing is stored and there is no later retry.

| answer | meaning |
|---|---|
| `200 {"deleted": true, "apple_revoked": true}` | done |
| `502 {"deleted": true, "apple_revoked": false, "detail"}` | our data is deleted; Apple refused the code or did not answer |
| `503 {"deleted": true, "apple_revoked": false, "detail"}` | our data is deleted; no Sign in with Apple key configured |
| `401` / `422` | bad access token / no `authorization_code`; nothing deleted |

On any answer with `"deleted": true` the app signs out and shows that the
account is deleted. If Apple was not told, the user can still remove
MiraNote under Sign in with Apple in their Apple ID settings.

The purge (`purge.py`) then deletes the marked account: the service runs
it every hour, `.venv/bin/python purge.py` runs it once by hand. Signing
in again with the same Apple ID purges the old account first and creates
a new, empty one; its subscription has no owner and the first Restore
reclaims it (Subscriptions API below, including the 30-day re-link
limit).

Settings (`.env`, optional; without them deletion answers 503 as above):

| setting | what |
|---|---|
| `APPLE_SIWA_KEY_PATH` | the Sign in with Apple key (.p8; developer account, Certificates, Identifiers & Profiles > Keys, with Sign in with Apple enabled for the app's App ID). Keep it outside the repo |
| `APPLE_SIWA_KEY_ID` | that key's id |

## Data retention

For the privacy policy (design 7.2). Journal pages, photos, stickers,
audio, mood and the life profile are never on this server. The purge and
retention jobs are the same hourly pass (`purge.py`).

| data | while the account exists | after deletion | kept for | why |
|---|---|---|---|---|
| `users` (our id, Apple `sub`, team id) | kept | deleted by the purge, within 24 h | -- | sign-in |
| `sessions` (hashed refresh tokens) | deleted once expired (60 days after issue) | revoked at once, deleted by the purge | -- | staying signed in |
| `subscriptions` (Apple chain, status, dates) | kept | `user_id` set to null; Restore can reclaim the chain | 13 months after the chain's last expiry (or grace end) | refunds, disputes, tax |
| `app_store_notifications` (Apple's payload without `appAccountToken`) | never linked to a user | nothing to strip | 1 year from receipt | audit, Apple retries |
| `ai_usage` (feature, model, units, cost; never content) | kept | `user_id` set to null | 13 months from the request | quotas (at most a month back), cost reports |
| `app_attest_keys` | never linked to a user | untouched | 90 days unused | fraud prevention |
| Apple tokens from the deletion code | never stored | revoked in the request | -- | Apple's revoke requirement |
| Application logs (user id, request id; never content) | | not rewritten | 30 days | debugging |
| Backups | | not rewritten | Neon restore window (6 h on Free, 7 days on Launch) + nightly `pg_dump` in R2, 30 days | recovery |

So deleted data is gone from every copy within 30 days of the purge. The
last two rows rely on host jobs that are not in this repo yet (design
9, steps 11-12: the log deletion job, the nightly `pg_dump` and the R2
30-day lifecycle rule); the policy can state them only once those run.

## Usage API (quotas)

| route | caller | body | answer |
|---|---|---|---|
| `POST /usage/reserve` | AI services | `{"user_id", "feature", "request_id", "units"?, "model"?}` | 200; 401 `unknown_user`; 409 `duplicate_request` / `request_id_conflict`; 429 `quota`; 503 `service_paused` |
| `POST /usage/settle` | AI services | `{"request_id", "status": "ok" \| "failed", "cost_usd"}` | `{"status"}`; 404 `unknown_request` |
| `GET /me/usage` | app, `Authorization: Bearer <access>` | | `{"plan", "environment", "features": {feature: {"limit", "used", "remaining", "window", "resets_at"}}}` |

- Entitlement: Plus while the user owns an unrevoked subscription whose
  `expires_at` or `grace_period_expires_at` is in the future (the
  timestamps, not `status`, decide). A Production chain wins over a
  Sandbox one; a Sandbox entitlement has its own lower caps.
- Caps are all in `quotas.py` (`CAPS`); change a number and restart.
  Features: `chat`, `expand` (expand and polish), `image_generate`,
  `image_edit` (stylize, AI border), `photo_push`, `mood` (signed-in
  only). Windows: rolling 24 h, rolling 7 days, or the UTC month.
- Reserve locks the user's row (`SELECT ... FOR UPDATE`), counts the window
  and inserts a `pending` row in one transaction, so parallel requests
  cannot overshoot a cap. `pending` and `ok` rows count, `failed` rows do
  not. A retried `request_id` runs again only if its row failed.
- A 429 body: `{"reason": "quota", "feature", "plan", "limit",
  "resets_at"}`; `resets_at` is when the next unit frees up.

### Daily cost budgets (kill switch)

Design 6.3, in `budget.py`. Reserve sums today's (UTC) `cost_usd` of the
`ok` and `pending` rows; once a budget is reached, that feature answers
503 for everyone until the next UTC midnight or a raised budget (restart):

```json
{"reason": "service_paused", "feature": "chat", "resets_at": "2026-11-18T00:00:00+00:00", "detail": "..."}
```

The AI services pass it to the app as is. All budgets are off by default
(unset or empty = no budget); set them in `.env`, in USD:

| variable | budget |
|---|---|
| `ACCOUNTS_DAILY_BUDGET_USD` | all features together |
| `ACCOUNTS_DAILY_BUDGET_USD_CHAT`, `_EXPAND`, `_IMAGE_GENERATE`, `_IMAGE_EDIT`, `_PHOTO_PUSH`, `_MOOD` | one feature |

`0` pauses a feature at once. A malformed value or an unknown feature
stops the service at startup. The first refused call of the day logs
`daily budget reached` (scope, budget, spend) once per process.
`.venv/bin/python budget.py` prints today's spend per feature next to the
budgets.

Gaps: a `pending` row costs 0 until it settles, so calls already in flight
can overshoot a budget by their own cost. Beta-token calls never reserve,
so they are neither counted nor paused (only beta_auth's rate limit bounds
them) until `ACCEPT_BETA_TOKENS` is turned off; beta `/mood` stays
unmetered until App Attest (design 6.5).

### Internal routes stay off the tunnel

`/usage/reserve` and `/usage/settle` take `X-Internal-Secret`
(`ACCOUNTS_INTERNAL_SECRET`, the same value in the repo root `.env` for the
AI services). The AI services call them on `127.0.0.1:8007`, never through
the tunnel. The tunnel must not serve them: put this rule **above** the
`beta-accounts` hostname rule (step 3 of the deploy below):

```yaml
  - hostname: beta-accounts.miranote.app
    path: ^/usage/
    service: http_status:404
```

As a second fence, a request carrying cloudflared's `Cf-Connecting-IP`
header is answered 404 by the service itself, secret or not.

## Subscriptions API

| route | caller | body | answer |
|---|---|---|---|
| `POST /subscriptions/sync` | app, `Authorization: Bearer <access>` | `{"signed_transaction", "signed_renewal_info"}` | `{"plan", "environment", "status", "expires_at"}`; 400 invalid signed data; 403 / 409 another account's purchase; 429 relinked too recently or too many links today; 503 not configured |
| `POST /appstore/notifications` | Apple | `{"signedPayload"}` | 200; 400 invalid signed data; 503 not configured (Apple retries) |

- The app sends one chain per call: the `jwsRepresentation` of the
  transaction and of the renewal info from `Product.SubscriptionInfo.Status`,
  after a purchase, on launch when signed in, and on Restore.
- Everything Apple signs is verified with Apple's `app-store-server-library`
  (chain to `APPSTORE_ROOT_CERTS`, Apple's OIDs, our bundle id, and the
  app's Apple ID on Production notifications). Production and Sandbox are
  both accepted and tagged on the row; Xcode / local StoreKit data is not.
- The row is recomputed from the snapshot every time (design 5.1); one
  signed earlier than the row's `last_signed_date` is not applied.
- Ownership (design 5.3): a chain whose `appAccountToken` is the caller is
  the caller's; one naming another live account is refused (403). With no
  token, or a deleted account's, an unowned chain is linked to the caller
  (at most once per chain per 30 days, at most 3 links per user per day)
  and Apple's Set App Account Token is called; a chain owned by someone
  else is refused (409). Every link is logged with its source.
- Notifications (design 5.4) are stored once by `notificationUUID` and
  applied in the same transaction, without `appAccountToken`. A
  notification links only an unowned chain, only to a live account.
  `CONSUMPTION_REQUEST` and `TEST` are stored, not applied. A refund
  removes Plus; used quota stays used.

Settings (`.env`, all optional; without the first two both routes answer
503 and the rest of the service runs):

| setting | what |
|---|---|
| `APPSTORE_ROOT_CERTS` | path(s), comma-separated, to Apple's root certificates in DER: `AppleRootCA-G3.cer` from <https://www.apple.com/certificateauthority/> |
| `APPSTORE_APP_APPLE_ID` | the app's numeric Apple ID (App Store Connect, App Information) |
| `APPSTORE_KEY_PATH`, `APPSTORE_KEY_ID`, `APPSTORE_ISSUER_ID` | the In-App Purchase key (.p8) and its ids (App Store Connect, Users and Access, Integrations). For Set App Account Token and `backfill.py`; without it a sync still links, Apple keeps the old token |

In App Store Connect, set both the Production and the Sandbox Server
Notifications URL (version 2) to
`https://beta-accounts.miranote.app/appstore/notifications`.

### Backfill

No scheduled job in 1.0. After downtime, or when a row looks wrong,
recompute rows from Apple's Get All Subscription Statuses:

```bash
.venv/bin/python backfill.py
.venv/bin/python backfill.py 2000000123456789
```

With no ids it walks every chain in `subscriptions`; an id not in the table
is looked up in Production, or Sandbox with `--sandbox`.

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
       path: ^/usage/
       service: http_status:404
     - hostname: beta-accounts.miranote.app
       service: http://127.0.0.1:8007
   ```

   The first rule keeps the internal usage routes off the internet (see
   Usage API above).

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
`httpx.MockTransport`, along with fake `/auth/token` and `/auth/revoke`
endpoints for deletion: the tests never reach the network beyond Postgres.
The App Store is faked the same way (`tests/fake_appstore.py`): a locally
made root CA, intermediate and leaf with Apple's OIDs sign the JWS, and the
tests configure that root as `APPSTORE_ROOT_CERTS` with
`APPSTORE_ONLINE_CHECKS=0` (no OCSP), so Apple's real verifier runs offline.

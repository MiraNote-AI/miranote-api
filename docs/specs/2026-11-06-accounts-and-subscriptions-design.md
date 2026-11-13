# Accounts and subscriptions -- backend design

- **Date:** 2026-11-06, revised 2026-11-13
- **Status:** Revised after review (Maverick, #99) and the team lead's
  data-plan decisions of 2026-11-13. Ready for a second review; no code
  until approved.
- **Issue:** miranote-api #95
- **Scope:** the smallest backend that can sign a user in, know whether
  they pay, meter AI cost, learn from their choices, and delete them on
  request, ahead of a paid launch in the **US only**. Journal content
  stays on the device. China and the EU are later, separate work
  (section 10).

What changed since the first draft, in one breath: no stored Apple
refresh token; ownership rules for subscription sync, offer codes and
Restore; grace period and full notification handling; an `ai_usage`
ledger for quotas; taste events (tier 2); App Attest for abuse; mood
computed on device where possible; a privacy-label table; Postgres on
a $0 phase 1 (Neon Free + the Mac) before Neon Launch + Fly.io;
section 9 is now a build plan.

## 1. Goals and non-goals

Goals:

1. A user signs in with Apple and gets a MiraNote account.
2. The backend knows each account's subscription state, kept current by
   Apple, not by the app's word.
3. Every server AI endpoint checks identity, entitlement and quota from
   one ledger instead of the shared beta token.
4. We keep the choices a user makes (never the content) so Mira gets
   better at the user's taste -- the one piece of data that builds the
   moat.
5. A user can delete their account from inside the app (Guideline
   5.1.1(v)).

Non-goals:

- Syncing journal pages, photos, stickers or audio to the server.
- Android, web sign-in, email/password accounts, phone sign-in.
- Our own payment flow. All purchases go through StoreKit; introductory
  offers and offer codes cover the launch discount (ios #170).
- Family Sharing for the subscription (off; once on it cannot be turned
  off).
- User profiles and avatars (not in 1.0).

## 2. Product decisions that shape the data

**What Mira is.** Mira is the user's diary butler -- a projection of
their own memories that organizes, writes, retrieves and resurfaces. It
is not a companion chatbot and not a mental-health product. Prompts and
copy stay task-focused ("journal assistant", not "journaling
companion"). A minimal safeguard shows the 988 crisis line if self-harm
text appears. 1.0 is rated 18+; the app still reads Declared Age Range
where the store sends it (TX, UT) -- an iOS work item.

**Mood** means a page's tone/color for the star jar, not an assessment
of the user's mental state. It is computed and stored on the device; the
server never stores it (section 6.5).

**Free vs Plus** (starting values; tuned after two weeks of real
`ai_usage` cost data):

| Feature | Free | Plus |
|---|---|---|
| Journal, canvas, layouts, sticker library, export, cutouts | on device, unlimited | unlimited |
| Mood / star jar | unlimited (on device, or `/mood`) | unlimited |
| Expand / polish | 10 per rolling 24 h | 300 per rolling 24 h |
| Mira chat | 5 turns per rolling 24 h | 200 turns per rolling 24 h |
| Image generation (stickers, backgrounds) | 5 per UTC month, once per device | 100 images per UTC month, shared |
| Image stylize / edit / borders | 2 per UTC month, once per device | (shared with the line above) |
| Photo push (collections) | 1 per rolling 7 days | 1 per rolling 24 h |

Plus = image generation + photo push + high chat caps. Images always
have a cap; "unlimited" is never promised for anything that costs money.

**Price:** $5.99/month and $39.99/year in one subscription group
`Plus`. Launch window: a 2-month introductory offer; early testers get
one-time offer codes. Referral rewards are 1.1 (promotional offers to
the inviter only).

**When to sign in:** not at first launch. The journal works signed out.
Sign in with Apple is asked at the first server-AI use (free or paid),
and always before a purchase so the purchase carries `appAccountToken`.
The one signed-out server call is `/mood`, under App Attest (section
6.5).

## 3. Sign-in

### 3.1 Flow

Sign in with Apple only: one login for an iOS-only app, satisfies
Guideline 4.8, and gives a stable user id (`sub`) with no password.

1. The app runs `ASAuthorizationAppleIDProvider` with a nonce and gets
   an identity token (JWT signed by Apple).
2. `POST /auth/apple` with the identity token. The server verifies it
   against Apple's keys (`https://appleid.apple.com/auth/keys`): issuer,
   audience = our bundle id, expiry, nonce.
3. The server finds or creates the `users` row by
   (`apple_team_id`, `apple_sub`) and returns its own tokens: an access
   token (JWT, ES256, 1 h, `kid` header) and a refresh token (opaque,
   60 days, stored hashed, rotated on every use).
4. The app keeps both in the Keychain and sends
   `Authorization: Bearer <access>` to every backend.

We do **not** exchange the authorization code and do **not** store an
Apple refresh token. Revocation at deletion re-runs Sign in with Apple
(section 7). We do not store the user's email or name.

### 3.2 Our tokens

- **Rotation with reuse detection.** Each refresh token belongs to a
  `family_id`. Using it returns a new pair and marks the old one
  replaced. If a replaced token is used again after a 10-second grace,
  the whole family is revoked (stolen-token case) and the user signs in
  again.
- **Concurrent refresh.** Within the 10-second grace, the just-replaced
  token returns the same new pair, so two parallel requests on launch do
  not log the user out.
- **Signing keys.** ES256 keys with `kid`; the public keys are served at
  `/.well-known/jwks.json` and cached by the AI services. Rotation keeps
  the old key published for 1 h (one access-token lifetime).

### 3.3 Sign in with Apple server-to-server notifications

`POST /auth/apple/notifications` (configured in the developer account),
verified as Apple-signed JWTs:

- `consent-revoked`: revoke all sessions; the user signs in again on the
  next server-AI use.
- `account-deleted`: treat as an account deletion (section 7).
- `email-disabled`, `email-enabled`: logged and ignored (no email
  stored).

### 3.4 Developer account and team id

Decided (2026-11-13): build and ship on the **personal developer
account** (team `W7U365VR4U`). A company/LLC is not on the critical path
for building.

Because SIWA `sub` is scoped to the developer team, `users` stores
`apple_team_id` from day one. A later move to an organization account
works like this:

1. Apple only transfers an app that has at least one released version,
   so the move happens after 1.0 is live.
2. After the transfer, the old team calls Apple's user-migration
   endpoint to turn each `sub` into a transfer identifier; the new team
   exchanges it for the new `sub`. This must finish within 60 days.
3. We update `users.apple_sub` and `users.apple_team_id` per row. Our
   user id (and so `appAccountToken`) does not change, so subscriptions
   stay linked.
4. Side effects to plan for: new App Store Server API and SIWA keys;
   App Attest keys re-attest (the App ID prefix changes); DeviceCheck
   bits reset (one month of free image quota can be granted twice).

## 4. Data model

PostgreSQL. All ids are UUIDs; all times are `timestamptz` in UTC.

### 4.1 Tier 1 -- operational minimum

`users`

| column | type | notes |
|---|---|---|
| id | uuid pk | also sent to StoreKit as `appAccountToken` |
| apple_sub | text | Apple's user id, scoped to the team |
| apple_team_id | text | unique (`apple_team_id`, `apple_sub`) |
| created_at | timestamptz | |
| deletion_requested_at | timestamptz null | set at deletion; the purge job removes the row (section 7) |

`sessions` -- our refresh tokens

| column | type | notes |
|---|---|---|
| id | uuid pk | |
| user_id | uuid fk | |
| family_id | uuid | one sign-in = one family (reuse detection) |
| token_hash | text unique | SHA-256 of the refresh token |
| replaced_at | timestamptz null | set on rotation |
| created_at, expires_at, revoked_at | timestamptz | |

`subscriptions` -- one row per Apple renewal chain

| column | type | notes |
|---|---|---|
| original_transaction_id | text pk | Apple's id for the whole chain |
| user_id | uuid fk null | null = unowned (offer code before sign-in, or deleted account) |
| linked_at | timestamptz null | when `user_id` was last set; used for the reclaim rate limit |
| product_id | text | `app.miranote.plus.monthly` or `.yearly` |
| status | text | `active`, `grace`, `billing_retry`, `expired`, `revoked` (derived, for reporting) |
| expires_at | timestamptz | from the latest transaction |
| grace_period_expires_at | timestamptz null | from renewal info |
| auto_renew | bool | |
| auto_renew_product_id | text null | pending upgrade/downgrade |
| revoked_at | timestamptz null | refund or revoke |
| revocation_reason | int null | Apple's code |
| offer_type, offer_id | text null | kept for ios #170 conversion work; no reports in 1.0 |
| environment | text | `Sandbox` or `Production` |
| last_signed_date | timestamptz | signed date of the newest snapshot applied |

`app_store_notifications` -- audit and idempotency

| column | type | notes |
|---|---|---|
| notification_uuid | text pk | Apple retries; a seen uuid is skipped |
| type, subtype | text | e.g. `DID_FAIL_TO_RENEW / GRACE_PERIOD` |
| original_transaction_id | text null | |
| signed_date | timestamptz | |
| payload | jsonb | decoded, verified payload **without** `appAccountToken` (the owner lives only in `subscriptions.user_id`); kept 1 year |
| processed_at | timestamptz | |

`ai_usage` -- one row per AI request; the only source of quotas

| column | type | notes |
|---|---|---|
| request_id | uuid pk | sent by the client; a retry with the same id is not charged twice |
| user_id | uuid fk null | null only for signed-out `/mood` |
| device_key_id | text null | App Attest key id, set **only** on signed-out rows |
| feature | text | `chat`, `expand`, `image_generate`, `image_edit`, `photo_push`, `mood`, `free_image_grant` |
| model | text | |
| units | int | e.g. images, turns |
| cost_usd | numeric | estimated from the model's list price |
| environment | text | `Sandbox` or `Production` entitlement at request time |
| status | text | `ok` or `failed`; failed rows do not count |
| created_at | timestamptz | |

No prompt, text, image or result is ever stored. Kept 13 months.

`app_attest_keys` -- App Attest device keys (not linked to a user)

| column | type | notes |
|---|---|---|
| key_id | text pk | from `DCAppAttestService` |
| public_key | bytea | from the verified attestation |
| receipt | bytea | for later fraud-metric checks |
| sign_count | bigint | must increase on every assertion |
| environment | text | `development` or `production` |
| created_at, last_used_at | timestamptz | rows unused for 90 days are deleted |

### 4.2 Tier 2 -- taste events

Choices only, never content. This is the moat: what the user picks,
keeps, undoes and regenerates.

`taste_events`

| column | type | notes |
|---|---|---|
| event_id | uuid pk | client-generated; duplicates ignored |
| user_id | uuid fk | |
| session_id | uuid | app session, client-generated |
| occurred_at, received_at | timestamptz | index on `occurred_at` |
| app_version, build | text | |
| tz_offset_minutes | int | |
| event_version | int | one global version that follows the app |
| name | text | registered names only; unknown names rejected |
| surface | text | e.g. `layout_picker`, `sticker_tray` |
| item_type, item_id | text | catalog ids only (layout, font, color, sticker style, tone option) |
| position | int null | rank shown |
| action | text | `shown`, `selected`, `kept`, `undone`, `deleted`, `regenerated` |
| request_id | uuid null | links to `ai_usage` when the item came from AI |

Rules:

- `shown` is logged too; without it a choice has no denominator.
- No text, images, prompts or mood. "Writing tone" comes only from the
  tone option the user picked (a catalog id), never inferred from text.
- Plain table, no partitioning, no `experiments` column in 1.0. A daily
  job deletes rows older than 18 months.
- Collected only when signed in **and** the Settings switch "Use my
  choices to improve and personalize MiraNote" is on (default on).
  Signed out: events are dropped on the device, not queued. Turning the
  switch off stops collection and deletes the user's collected events
  (`DELETE /me/taste-events`).

### 4.3 Profiles

- **Style profile** (preferred layouts, fonts, colors, sticker styles,
  tone option): computed **on device** in 1.0 from the device's own
  choices. The server only aggregates events across users for product
  analytics and model comparison. Server-side or cross-device
  personalization waits until it is needed.
- **Life profile** (people, places, themes): phase 2, on device only,
  sent with a request when Mira needs it, never stored on the server.

### 4.4 Never stored on our server

Journal pages, photos, stickers, audio, mood, life profile, prompts and
AI results. Some of these pass **through** the server to an AI provider
and are discarded when the response is sent; section 7.3 covers what that
means for the privacy label.

## 5. Keeping subscriptions current

### 5.1 One rule: recompute from Apple's snapshot

Every input (app sync, notification, Server API call) yields Apple's
signed transaction info and renewal info; the app sync sends both (5.3).
We verify both with Apple's
`app-store-server-library` (Python) and **recompute the whole row** from
them, instead of a per-notification-type state machine. A snapshot whose
signed date is older than `last_signed_date` is stored (if it came from a
notification) but not applied.

Derived `status`:

1. `revoked` if the transaction has a revocation date.
2. else `active` if `expires_at` is in the future.
3. else `grace` if `grace_period_expires_at` is in the future.
4. else `billing_retry` if renewal info says billing retry.
5. else `expired`.

**Entitled** = a row owned by the user with `revoked_at` null and
(`expires_at` > now or `grace_period_expires_at` > now). It is computed
from timestamps at read time, so a stale `status` never locks anyone
out. Billing Grace Period is turned on in App Store Connect.

### 5.2 Environments

App Review buys with Sandbox accounts in the production build, and
TestFlight purchases are Sandbox too. So the production service accepts
**both** `Production` and `Sandbox` signed data (one verifier per
environment) and tags each row. A Sandbox entitlement gets lower image
caps (10 images per UTC month) so 300 public TestFlight testers cannot
run up a real bill. Both the Production and Sandbox notification URLs in
App Store Connect point at the one live service (the Mac in phase 1,
Fly in phase 2; section 8.2).

Every verified payload must also have `bundleId` = our bundle id.

### 5.3 App sync and ownership

`POST /subscriptions/sync` (signed in). The app finds its chains with
`Transaction.currentEntitlements` and, for each one, sends both
Apple-signed JWS strings of its `Product.SubscriptionInfo.Status`: the
transaction (`transaction.jwsRepresentation`) and the renewal info
(`renewalInfo.jwsRepresentation`). The transaction alone is not enough:
the grace period end, `auto_renew`, the pending product and billing
retry exist only in the renewal info, so a row recomputed from the
transaction alone would lock a grace-period user out (5.1). The app
calls it after a purchase, on launch when signed in, and on Restore.
For each chain:

1. Verify both JWS, `bundleId`, environment (5.2), and that both name
   the same `originalTransactionId`.
2. If `appAccountToken` names the caller, the chain is the caller's.
3. If `appAccountToken` names another **live** user, reject that chain
   (403). A signed transaction is not a secret; this stops a paying
   user from sharing theirs.
4. If `appAccountToken` is absent (offer code redeemed in the App Store,
   purchase made before sign-in) or names no live user (its account was
   deleted, section 7):
   - chain unowned -> link it to the caller, then call the Server API
     **Set App Account Token** with the caller's id (it also replaces an
     old token), so later notifications carry the user;
   - chain owned by the caller -> no change;
   - chain owned by another account -> reject (409).
5. Upsert the row from the snapshot (5.1).

On a 403 or 409 the app shows "this purchase belongs to another
MiraNote account" with a link to support.

**Reclaiming an unowned chain.** A chain becomes unowned when its
account is deleted (section 7). Its transactions keep carrying the
deleted account's id as `appAccountToken` until Set App Account Token
replaces it; that is why step 4 treats a token that names no live user
like an absent one (otherwise the paying user could never get the chain
back). The server cannot tell whether the same Apple ID is asking, so
the rule is: the first account that sends it in a Restore (or a launch
sync) and passes step 4 gets it. Every link is logged (user id, chain
id, source) and rate-limited: a chain can be re-linked at most once per
30 days (`linked_at`), and one user can link at most 3 chains per 24 h.

**A chain owned by another live account** (decided 2026-11-13): 1.0
rejects the sync as above and has no self-service transfer. Support
moves a chain by hand: the user sends the order ID from Apple's receipt
email, support calls the Server API **Look Up Order ID** to confirm the
order belongs to that chain, sets `user_id` to the user's account and
calls Set App Account Token. The move is logged like any other link.
(Look Up Order ID does not work in Sandbox.)

**Restore Purchases** (Guideline 3.1.1) is in the paywall and in
Settings. Signed out -> sign in first -> `AppStore.sync()` -> send
`currentEntitlements` to `/subscriptions/sync`.

### 5.4 App Store Server Notifications V2

`POST /appstore/notifications` (public; trust comes from Apple's JWS
chain). Every notification is stored once by `notification_uuid`; any
that carries signed transaction/renewal info triggers a recompute (5.1).

| Type | Effect |
|---|---|
| `SUBSCRIBED`, `DID_RENEW`, `OFFER_REDEEMED`, `RENEWAL_EXTENDED` | recompute (entitled) |
| `DID_CHANGE_RENEWAL_STATUS`, `DID_CHANGE_RENEWAL_PREF`, `PRICE_INCREASE` | recompute (`auto_renew`, `auto_renew_product_id`) |
| `DID_FAIL_TO_RENEW` (incl. `GRACE_PERIOD`), `GRACE_PERIOD_EXPIRED`, `EXPIRED` | recompute (grace / retry / expired) |
| `REFUND`, `REVOKE` | recompute -> revoked; used quota is not clawed back |
| `REFUND_REVERSED` | recompute -> entitlement reinstated |
| `CONSUMPTION_REQUEST` | stored; not answered in 1.0 (it needs the user's consent) |
| `TEST`, anything else | stored; 200 |

A notification is stored and applied in **one transaction**: if
applying fails, nothing is stored and Apple's retry applies it again (a
`notification_uuid` already stored is skipped, so storing first and
applying later would lose that update).

A notification sets `user_id` only on an unowned row, and only when its
`appAccountToken` names a live user; it never moves an owned chain.
Renewals of a deleted account's chain keep arriving with that account's
id, and writing it to `user_id` would fail the foreign key and make
Apple retry forever. A notification without such a token creates or
updates an unowned row; the next launch sync links it (5.3).

### 5.5 Repair path

In 1.0 the App Store Server API (`Get All Subscription Statuses`) is
called when a sync disagrees with our row, and from a manual backfill
script (Notification History) after any downtime. No scheduled job
until there is real traffic.

The app also reads `Transaction.currentEntitlements` locally so a paid
user is not locked out of local features while offline. The server
stays the authority for server AI.

## 6. AI usage, quotas and abuse

### 6.1 Where the checks run

The AI services (chat, expand, image, photo push, mood) import one
shared guard module. Per request it:

1. verifies the access JWT against our JWKS (no network hop);
2. reads the user's row and entitlement from Postgres (one query; a
   deleted user or a refunded subscription is refused at once, so there
   is no 1 h access-token lag for entitlement);
3. checks the quota from `ai_usage` and inserts a pending row with the
   client's `request_id`;
4. marks the row `ok` with `cost_usd`, or `failed` (not counted).

If Postgres is unreachable the guard **fails closed**: server AI returns
503 and the app shows a retry message. Local features are unaffected.

### 6.2 Quota windows

- Daily caps: rolling 24 h (a UTC day would reset at 4-5 pm in
  California).
- Monthly caps: UTC calendar month.
- Weekly caps (free photo push): rolling 7 days.

Caps live in config, not code, and start at the values in section 2.

### 6.3 Cost kill switch

Each feature has a daily budget in USD (sum of `cost_usd` over the
rolling 24 h). Over budget, that feature returns 503 for everyone until
it drops back or the budget is raised. This bounds the damage of a bug
or an abuse wave.

### 6.4 App Attest and DeviceCheck

- **App Attest.** On first server-AI use the app attests a key; the
  server verifies the attestation and stores the key in
  `app_attest_keys`. Every server-AI request then carries an assertion
  over the request body, checked against the stored key and counter.
  Keys are not linked to a user.
- **DeviceCheck** has two bits per device per team plus a "last updated"
  month; it cannot count. We use one bit only, meaning "the free image
  quota was granted on this device this month". When a free user first
  asks for an image in a month: if the bit is set for this month, no free
  images on this account; otherwise set it and write a
  `free_image_grant` row in `ai_usage` for this user. A user with a grant
  row this month keeps their quota. This stops "new account, same
  phone" farming of free images.

### 6.5 Mood

Hybrid, decided 2026-11-13:

1. On devices with Apple Intelligence, mood runs on device (Apple
   Foundation Models). Nothing reaches the server.
2. Elsewhere the app calls `POST /mood`. It is the one endpoint allowed
   **signed out**, and only with a valid App Attest assertion. Signed-out
   calls are limited per device key (start: 30 per rolling 24 h,
   counted on `ai_usage.device_key_id`); signed-in calls count per user.
3. The server sends the page text to the model, returns the tone/color,
   and discards both. The `ai_usage` row holds no text and no mood.
4. If neither path is available, the page gets a neutral color.

### 6.6 Stateless Mira chat

The client sends the conversation history with each request (capped at
the last 20 turns). The server keeps no session -- the current 1-hour
in-memory session goes away. This keeps "nothing stored" true and lets
two Fly machines serve any request without Redis or a session table.
Photo push compose logs keep only `request_id`.

Photo push in 1.0 is assumed to be an on-device local notification that
calls the compose endpoint, so no APNs device-token table (open question
2).

### 6.7 The beta token

`BETA_TOKENS` (in `beta_auth.py`) can be extracted from any TestFlight
IPA. It keeps working on the beta Mac backends until sign-in builds
ship, and is never accepted by the production AI endpoints.

## 7. Account deletion, retention and privacy label

### 7.1 Flow

1. Settings > Account > Delete account. The confirmation says: your
   account and server data are deleted within 24 hours, your journal
   stays on this device unless you also erase it, and deleting does not
   cancel your Apple subscription (with a link to Apple's subscription
   settings, as Apple requires).
2. The app runs Sign in with Apple again and gets a fresh authorization
   code.
3. `DELETE /me` with that code. The server exchanges it for Apple tokens
   and immediately calls Apple's `/auth/revoke`, then sets
   `deletion_requested_at` and revokes all `sessions`. If Apple's revoke
   fails, deletion still proceeds and the failure is logged for a retry.
4. The app signs out and shows "Your account is deleted" -- that screen
   is the confirmation (we hold no email to send one).
5. The purge job (hourly) runs within 24 hours. Signing in again with
   the same Apple ID before the purge runs purges the old account first,
   then creates a new, empty one.

### 7.2 What the purge does

| Data | On deletion | Kept for |
|---|---|---|
| `users`, `sessions` | deleted | -- |
| `taste_events` | deleted | -- |
| `subscriptions` | `user_id` set to null (pseudonymized); reclaimable by Restore (5.3) | 13 months after the chain's last expiry |
| `app_store_notifications` | nothing to strip: payloads are stored without `appAccountToken` (4.1), so renewals that keep arriving for the chain do not bring the deleted id back | 1 year from receipt |
| `ai_usage` | `user_id` set to null | 13 months from the request |
| `app_attest_keys`, DeviceCheck bits | untouched (not linked to a user) | 90 days unused |
| Application logs | not rewritten; they carry only user id and request id, never content | 30 days |
| Backups | not rewritten | Neon 7-day restore window + nightly `pg_dump` in R2 kept 30 days |

So deleted data is gone from every backup within 30 days. The privacy
policy states these periods.

### 7.3 App Privacy label

What 1.0 declares (all "linked to you" unless stated; none used for
tracking, so no ATT prompt):

| Apple data type | What it is | Purpose |
|---|---|---|
| Identifiers > User ID | `users.id`, Apple `sub` | App Functionality |
| Identifiers > Device ID (not linked) | App Attest key id on signed-out `/mood` | App Functionality (fraud prevention) |
| Purchases > Purchase History | `subscriptions` | App Functionality |
| Usage Data > Product Interaction | `ai_usage` (quotas, cost) | App Functionality |
| Usage Data > Product Interaction | `taste_events` | Analytics (add Product Personalization if a server-side profile ships) |
| User Content > Other User Content, Photos, Audio | page text, photos and audio sent to AI providers (DeepSeek, Alibaba DashScope, Google Vertex), when a provider keeps them longer than real-time processing | App Functionality |

"We do not store it" is not the same as "not collected": under Apple's
definition, content a provider retains counts. The provider API terms
have not been reviewed yet (ios `docs/privacy/ai-data.md` says so); the
last row is declared until that review says otherwise. `ai-data.md` and
the label are updated when sign-in ships. Paywall disclosures follow
Guideline 3.1.2 (price, period, auto-renew, post-intro price, Terms and
Privacy links).

## 8. Hosting

### 8.1 Stack (decided 2026-11-13)

- **PostgreSQL** via SQLAlchemy 2 (async) + asyncpg + Alembic.
- **Provider-agnostic:** only standard Postgres. Moving providers is a
  `DATABASE_URL` change plus one `pg_dump`/`pg_restore`. Code is written
  for a transaction-mode pooler from day one: asyncpg
  `statement_cache_size=0`, no LISTEN/NOTIFY, no session-level `SET`, no
  advisory locks held across transactions.
- Tests run against real Postgres (a service container in CI), not
  SQLite.

### 8.2 Providers, in two phases

Decided 2026-11-13: **phase 1 costs nothing.** Moving to phase 2
changes only `DATABASE_URL` and the host.

- **Phase 1 -- internal testing ($0).**
  - Database: **Neon Free**, `aws-us-east-1`, Postgres 17. Limits we
    accept for internal testing: 0.5 GB, suspend after 5 minutes idle
    (the first query after that is 0.3-0.5 s slower), 6-hour restore
    window.
  - Backups: nightly encrypted `pg_dump` to an R2 bucket
    (`miranote-pg-backups`, R2 free tier), kept 30 days.
  - API: the `accounts/` service runs on the team lead's Mac behind the
    existing Cloudflare tunnel as its own hostname
    (`beta-accounts.miranote.app`), independent of the other beta
    services. Secrets in a local `.env`, never in the repo.
- **Phase 2 -- public TestFlight and 1.0 (recommended, pending
  team-lead confirmation).**
  - Database: **Neon Launch**, same project. No monthly minimum, 7-day
    point-in-time restore, encrypted at rest, branches (`main` = prod,
    `dev`) for testing migrations. Scale-to-zero off (no cold queries on
    Apple's webhooks).
  - API: **FastAPI on Fly.io `iad`**, 2 x shared-cpu-1x 1 GB, always
    running, under a `miranote` Fly org. Same metro as Neon us-east-1
    (1-2 ms per query); the team already runs mirabot on Fly. Domain
    `api.miranote.app` (Cloudflare DNS -> Fly). Secrets in Fly secrets.
  - Backups: unchanged (R2, 30 days).
  - Cost (DB + hosting only, AI excluded): about $20-35/month during
    public TestFlight, $35-55 at 1k users, $120-200 at 10k.
- **Fallback:** Fly Managed Postgres ($38/month, private network,
  10-day restore) with no code change.
- Not chosen: GCP Cloud Run + Cloud SQL (trial ends 2026-12-13, the
  cheapest Cloud SQL tier has no SLA); Supabase (we do not use its auth;
  free projects pause after a week idle). The Mac is fine for internal
  testing only: it has no uptime guarantee for Apple's retries and user
  data must not live on a laptop once real users sign in.

### 8.3 Long requests

Fly's proxy drops a connection after 60 s with no bytes; Cloudflare
returns 524 when no response header arrives within 100 s. Image editing
takes about 100 s. So long endpoints either stream an SSE heartbeat
until the result is ready (recommended: stateless, no job store) or
return `202` and are polled. This applies on both hosts (the Mac is
behind Cloudflare too), so it lands before public TestFlight.

### 8.4 Service layout

- `accounts/` -- new FastAPI service: `/auth/*`, `/me`,
  `/subscriptions/sync`, `/appstore/notifications`, taste events, purge
  and retention jobs, Alembic migrations.
- A shared guard module (section 6.1) imported by every AI service; each
  AI service gets `DATABASE_URL` and the JWKS URL.

## 9. Build plan

| Phase | Database | API host | Cost |
|---|---|---|---|
| 1. Internal testing | Neon Free, `aws-us-east-1`, PG 17, nightly `pg_dump` to R2 | team lead's Mac, Cloudflare tunnel, `beta-accounts.miranote.app` | $0 |
| 2. Public TestFlight and 1.0 | Neon Launch (same project) | Fly.io `iad`, `api.miranote.app` | about $20-55/month |

Going from phase 1 to phase 2 changes only `DATABASE_URL` and the host
(section 8.2). Each step below ends with tests passing in CI against
Postgres.

**Can start now (local Postgres 17, no new accounts):**

1. Alembic migrations for every table in section 4; Postgres service
   container in CI. -> verify: `alembic upgrade head` and `downgrade`
   run clean in CI.
2. `accounts/` skeleton: `/auth/apple` (identity-token checks tested
   with fixture keys), our tokens with rotation, reuse detection and
   concurrent-refresh grace, JWKS, `/me`.
3. Subscription core: recompute-from-snapshot, entitlement function,
   `/subscriptions/sync` ownership rules, reclaim rate limits,
   `/appstore/notifications` idempotency. -> verify: table-driven tests
   with fixture payloads for every row of 5.4 and every branch of 5.3.
4. Guard module: JWT check, entitlement, `ai_usage` ledger, rolling
   windows, Sandbox caps, kill switch, fail-closed. Wire into chat first.
5. Stateless chat (client history) and SSE heartbeat on the long image
   endpoints.
6. App Attest verification and `app_attest_keys` (tested with Apple's
   documented sample attestation); signed-out `/mood` with per-device
   limits.
7. Taste events: `POST /events` (batch), registered-name check,
   `DELETE /me/taste-events`, 18-month retention job.
8. Deletion: `DELETE /me` (Apple revoke mocked), purge job, retention
   jobs for every row of 7.2.

**Phase 1 setup ($0; team lead / account owner):**

9. App Store Connect: subscription group `Plus` with monthly ($5.99) and
   yearly ($39.99), the 2-month introductory offer (not yet live),
   Billing Grace Period on, Family Sharing off; an In-App Purchase key
   for the Server API; a DeviceCheck key; Server Notifications V2 and
   the SIWA server-to-server notification URL ->
   `beta-accounts.miranote.app`.
10. Neon Free: org `miranote`, project in `aws-us-east-1`, Postgres 17,
    branches `main` and `dev`.
11. Cloudflare: tunnel hostname `beta-accounts.miranote.app` on the team
    lead's Mac; R2 bucket `miranote-pg-backups` with 30-day retention.
12. Run `accounts/` on the Mac against Neon; end-to-end Sandbox tests
    from internal TestFlight: purchase, renewal, grace, refund and
    `REFUND_REVERSED`, offer-code redemption outside the app, Restore,
    deletion.

**Phase 2 -- public TestFlight and 1.0:**

13. Upgrade Neon to Launch (scale-to-zero off); Fly.io org `miranote`
    with a deploy token; DNS `api.miranote.app` -> Fly; deploy
    `accounts/` and the AI services to Fly `iad`; repoint the App Store
    and SIWA notification URLs to `api.miranote.app`.
14. iOS (separate issues): sign-in at first server-AI use, paywall with
    3.1.2 disclosures, Restore in paywall and Settings, deletion with
    SIWA re-run, the taste switch, App Attest, on-device mood, Declared
    Age Range.
15. Before charging: Neon paid plan in place, privacy policy and App
    Privacy label updated (section 7.3), production AI endpoints refuse
    the beta token, China-routing trigger checked (section 10).

## 10. Later

- **China-region providers for US traffic.** Some calls go to DashScope
  (Beijing) and DeepSeek today. Moving US traffic to non-China endpoints
  is deferred until **2,000 cumulative users or before we start
  charging, whichever comes first**. Until then the privacy policy and
  `ai-data.md` name China as a processing location. (DOJ 28 CFR 202
  thresholds count US persons over 12 months; public TestFlight's 300 is
  far below them.)
- **Mainland China.** A separate deployment (e.g. Alibaba Cloud) with
  the same schema and its own database (PIPL residency), a China entity,
  ICP and app filing, generative-AI filing, and phone sign-in. Not a
  region of the US deployment.
- **EU.** GDPR review, an EU database region (Neon has Frankfurt), and
  the same tier rules. Not before the US launch.
- **Organization developer account** -- the migration in 3.4.
- Server-side style profile, referral offers (1.1), `CONSUMPTION_REQUEST`
  answers, scheduled Notification History backfill, event partitioning.

## 11. Decided and open

### Decided

- 2026-11-13 -- US market only; China and EU later.
- 2026-11-13 -- Plus = image generation + photo push + high chat caps;
  $5.99/month, $39.99/year, 2-month introductory offer.
- 2026-11-13 -- Sign in with Apple at the first server-AI use, not at
  first launch.
- 2026-11-13 -- Postgres via SQLAlchemy 2 async + asyncpg + Alembic,
  provider-agnostic.
- 2026-11-13 -- Phase 1 (internal testing) costs nothing: Neon Free +
  R2 backups + `accounts/` on the team lead's Mac behind the tunnel.
  Phase 2 (Neon Launch + Fly.io `iad`) is recommended, pending
  confirmation.
- 2026-11-13 -- Personal developer account for now; record
  `apple_team_id`; migrate with transfer identifiers later.
- 2026-11-13 -- Mood hybrid: on device where available, otherwise
  signed-out `/mood` under App Attest; never stored on the server.
- 2026-11-13 -- Tier 2 taste events with their own switch (default on;
  off deletes collected events); style profile on device for 1.0; life
  profile phase 2, device only.
- 2026-11-13 -- No stored Apple refresh token; stateless Mira chat;
  `ai_usage` ledger for quotas; App Attest key table in tier 1.
- 2026-11-13 -- China-provider routing deferred, trigger: 2,000 users or
  before charging, whichever first.

### Open

1. Confirm the phase 2 providers, Neon Launch + Fly.io `iad` (team
   lead), before step 13. Phase 1 is decided.
2. Photo push: on-device local notification (assumed here, no new
   table) or server APNs (adds a device-token table)?
3. When a chain is owned by another live account, 1.0 rejects the sync
   (5.3). Is a support-assisted transfer enough, or do we want
   self-service transfer on Restore?
4. Should turning the taste switch off also stop the on-device style
   ranking (default order instead)? The switch name says "personalize",
   so I'd say yes.

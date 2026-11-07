# Accounts and subscriptions -- backend design

- **Date:** 2026-11-06
- **Status:** Draft for team review (no code until reviewed)
- **Issue:** miranote-api #95
- **Scope:** the smallest backend that can sign a user in, know whether
  they pay, and delete them on request, ahead of a paid launch in the US.
  Journal content stays on the device. Mainland China is a later,
  separate deployment (section 8).

## 1. Goals

1. A user signs in with Apple and gets a MiraNote account.
2. The backend knows each account's subscription state, kept current by
   Apple, not by the app's word.
3. The paid AI endpoints (image, chat, voice, collections) can check
   that state instead of the one shared beta token.
4. A user can delete their account from inside the app (App Store
   Review Guideline 5.1.1(v)).

## 2. Non-goals

- Syncing journal pages, photos or stickers to the server.
- Android, web sign-in, email/password accounts.
- Our own payment flow. All purchases go through StoreKit; App Store
  offer codes and introductory offers cover the launch discount (#170).
- Team/family sharing of a subscription.

## 3. Sign-in

**Sign in with Apple only.** It is the one login an iOS-only app needs,
it satisfies Guideline 4.8 by itself, and it gives a stable user id
(`sub`) without us storing a password.

Flow:

1. The app runs `ASAuthorizationAppleIDProvider` and gets an identity
   token (a JWT signed by Apple) and an authorization code.
2. `POST /auth/apple` with both. The server verifies the JWT against
   Apple's public keys (`https://appleid.apple.com/auth/keys`): issuer,
   audience = our bundle id, expiry, nonce.
3. The server exchanges the authorization code for an Apple refresh
   token and stores it (needed later to revoke on deletion, section 6).
4. The server finds or creates the `users` row by `apple_sub` and
   returns its own tokens: a short access token (JWT, 1 h) and a long
   refresh token (opaque, 60 days, stored hashed, rotated on use).
5. The app keeps both in the Keychain and sends
   `Authorization: Bearer <access>` to every backend.

The beta's shared `BETA_API_TOKEN` keeps working alongside this until
TestFlight builds with sign-in are out, then it is removed.

## 4. Tables

PostgreSQL. All ids are UUIDs; all times are `timestamptz` in UTC.

`users`

| column | type | notes |
|---|---|---|
| id | uuid pk | also sent to StoreKit as `appAccountToken` |
| apple_sub | text unique | Apple's stable user id |
| email | text null | often a private relay address; may be absent after the first sign-in |
| phone | text null | reserved for the China deployment, unused in the US |
| apple_refresh_token | text null | encrypted at rest; used only to revoke |
| created_at | timestamptz | |
| deletion_requested_at | timestamptz null | set when the user deletes; the row is purged after 30 days |

`profiles` -- one per user

| column | type | notes |
|---|---|---|
| user_id | uuid pk, fk users | |
| display_name | text | |
| avatar_url | text null | object storage, not the database |
| updated_at | timestamptz | |

`sessions` -- our refresh tokens

| column | type | notes |
|---|---|---|
| id | uuid pk | |
| user_id | uuid fk | |
| token_hash | text unique | SHA-256 of the refresh token |
| expires_at, revoked_at | timestamptz | |

`subscriptions` -- one row per Apple subscription chain

| column | type | notes |
|---|---|---|
| original_transaction_id | text pk | Apple's id for the whole renewal chain |
| user_id | uuid fk | from `appAccountToken` |
| product_id | text | e.g. `app.miranote.plus.monthly` |
| status | text | `active`, `grace`, `billing_retry`, `expired`, `revoked` |
| expires_at | timestamptz | |
| auto_renew | bool | |
| environment | text | `Sandbox` (TestFlight) or `Production` |
| updated_at | timestamptz | signed date of the last notification applied |

`app_store_notifications` -- audit and idempotency

| column | type | notes |
|---|---|---|
| notification_uuid | text pk | Apple retries; a seen uuid is skipped |
| type, subtype | text | e.g. `DID_RENEW`, `EXPIRED / VOLUNTARY` |
| signed_date | timestamptz | |
| payload | jsonb | the decoded, verified payload |
| processed_at | timestamptz | |

An account is **entitled** when it has a subscription with status
`active` or `grace` and `expires_at` in the future.

## 5. Keeping subscriptions current

Three inputs, one writer:

1. **Purchase from the app.** StoreKit 2 purchase with
   `appAccountToken = users.id`. The app then calls
   `POST /subscriptions/sync` with the signed transaction; the server
   verifies the JWS and upserts the row. This makes the purchase show at
   once, without waiting for Apple's notification.
2. **App Store Server Notifications V2.** `POST /appstore/notifications`
   (public, no bearer token; trust comes from verifying Apple's JWS
   signature chain). Handles `SUBSCRIBED`, `DID_RENEW`,
   `DID_FAIL_TO_RENEW`, `GRACE_PERIOD_EXPIRED`, `EXPIRED`, `REFUND`,
   `REVOKE`, `DID_CHANGE_RENEWAL_STATUS`. A notification older than the
   row's `updated_at` is stored but not applied (they can arrive out of
   order).
3. **App Store Server API** (`Get All Subscription Statuses`) as a
   repair path: called when a sync looks inconsistent, and from a daily
   job for rows that expire within a day.

Apple's `app-store-server-library` (Python) does the JWS verification and
the Server API calls; we do not hand-roll certificate chain checks.

The app also reads `Transaction.currentEntitlements` locally so a paid
user is not locked out while offline. The server stays the authority for
the paid AI endpoints.

## 6. Account deletion

1. Settings > Account > Delete account, with a confirmation.
2. `DELETE /me`: sets `deletion_requested_at`, revokes all `sessions`,
   and calls Apple's `/auth/revoke` with the stored Apple refresh token
   (required for Sign in with Apple apps).
3. The app signs out and keeps the on-device journal unless the user
   also chooses to erase it.
4. A daily job purges rows older than 30 days. `subscriptions` rows are
   detached (user_id null) rather than deleted, for refund and tax
   records.
5. Deleting the account does not cancel the Apple subscription. The
   screen says so and links to Apple's subscription settings, as Apple
   requires.

## 7. Hosting (US)

The beta backends run on a teammate's Mac behind a Cloudflare tunnel.
That is fine for AI calls, but not for this: Apple's notifications and
sign-in need an always-on endpoint, and the data must not live on a
laptop.

Recommendation: **a small FastAPI service on Fly.io + managed
PostgreSQL (Neon or Supabase free tier, US region).**

- The team already deploys mirabot to Fly.io, so there is nothing new to
  learn.
- Free or near-free at our scale; backups are included in the managed
  database.
- The service is new (`accounts/`), separate from the AI POCs. The POCs
  verify our access JWT with a shared public key and ask
  `GET /me/entitlement` (cached for a minute), so they need no database.

Not chosen:
- GCP Cloud Run + Cloud SQL: the $300 trial credit ends 2026-12-13,
  and Cloud SQL has no free tier.
- Keeping it on the Mac: no uptime, so Apple's retries would pile up.

Secrets (Apple key for the Server API, Apple sign-in key, JWT signing
key, database URL) live in Fly secrets, never in the repo.

## 8. Mainland China (later)

Not built now, but the schema leaves room for it:
- Personal data of users in China must be stored in China (PIPL). That
  means a second deployment (e.g. Alibaba Cloud), with the same schema
  and its own database. It is not a region of the US one.
- It needs a China legal entity, ICP filing for the API domain, and the
  app filing number shown in App Store Connect. The entity question is
  open (owner: project lead).
- Phone-number sign-in is common there. That is why `users.phone` exists
  and why sign-in is behind our own tokens, not Apple's directly.

## 9. Rollout

1. Review this doc (team).
2. App Store Connect: create the subscription group and products; turn
   on Server Notifications V2 (Sandbox and Production URLs).
3. `accounts/` service with sign-in, `/me`, deletion and tests; deploy
   to Fly.io.
4. Notifications + sync + entitlement; test end-to-end with Sandbox
   purchases from TestFlight.
5. iOS: sign-in screen, paywall, deletion screen (separate iOS issues).
6. The AI POCs switch from the beta token to access JWTs plus
   entitlement checks.

## 10. Open questions

1. Free vs paid split: which features need a subscription? #170
   suggests on-device cutouts stay free.
2. Products: monthly only, or monthly + yearly? Price points?
3. Do we require sign-in at first launch, or only when a paid feature
   is used? (Recommendation: only when needed. The journal works
   signed-out, because it is local.)
4. Neon or Supabase: both work. Supabase also offers object storage for
   avatars.

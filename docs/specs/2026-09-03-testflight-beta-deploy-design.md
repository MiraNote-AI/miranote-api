# TestFlight internal beta -- hosting and distribution design

- **Date:** 2026-09-03
- **Author:** mengjia (Claude-assisted)
- **Status:** Draft, revised after red-team review
- **Revision:** v4 (2026-09-04) -- v3 records the measured `/cutout`
  latency; v4 records the measured `/transcribe` latency and fills the
  voice timeout row; see section 15.
- **Scope:** deployment and distribution only. Backend changes in
  `poc/image-generation/main.py` and `scripts/start_backends.sh`, a new
  `shared/beta_auth.py`, plus a companion PR in `miranote-ios`
  (`MiraNoteConfig.swift`, `HTTPClient.swift`, `ImageStudio.swift`,
  `MiraCanvasCoordinator.swift`, `project.yml`, `App/Info.plist`, and a
  new asset catalog). No product features change.
- **Reference:** supersedes the LAN-only beta path documented in
  `miranote-ios/docs/RUN_ON_YOUR_PHONE.md` (from api #37 / ios #40).
  That path requires phone and Mac on the same Wi-Fi and re-signing
  every 7 days; this spec replaces it with a public HTTPS endpoint and
  TestFlight distribution.

## 1. Goal

Let up to 10 invited testers install MiraNote from TestFlight and use it
from anywhere, without being on the team's Wi-Fi and without a weekly
re-sign ritual.

Concretely:

- The four POC backends stay on Mengjia's Mac but become reachable over
  public HTTPS.
- The endpoints stop being open to the world.
- The iOS app ships through TestFlight internal testing, which requires
  no Beta App Review.

## 2. Non-goals

- Moving the backends off the Mac. A cloud VM or containerised
  deployment is the right long-term answer, but it is a separate piece
  of work and is not required to validate the beta loop.
- A user account system. One shared bearer token is the whole auth
  model for this stage.
- External TestFlight testing (public link, up to 10,000 testers). That
  needs Beta App Review, a privacy policy URL, and a backend that can
  absorb uncontrolled traffic. Explicitly out of scope; the auth and
  rate-limiting shape chosen here does not have to be redone to get
  there later.
- Docker, CI-driven uploads (fastlane), and multi-environment
  (staging/prod) separation.
- Fixing the product-level behaviour of any POC. This spec touches
  `image-generation` only to correct a concurrency defect that the
  deployment change would otherwise amplify (section 6).

## 3. Verified constraints

This section records what was measured or read from primary sources
rather than assumed, because three plausible-sounding assumptions turned
out to be wrong and the design depends on the corrected values.

### 3.1 Cloudflare proxy read timeout is 125s and cannot be raised

Cloudflare returns error 524 when the origin does not produce a response
within the Proxy Read Timeout. Cloudflare's own documentation gives the
default as **125 seconds**, adjustable only on Enterprise plans (up to
6,000s) via Cache Rules or the Edit Zone Settings API. Free, Pro, and
Business plans are fixed.

Source: <https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524>

### 3.2 The grey-cloud escape hatch does not exist for Tunnel

The standard workaround for 524 -- serve long-running routes on a
subdomain with the proxy disabled (grey cloud) -- is unavailable here.
`cloudflared tunnel route dns` creates a CNAME to
`<tunnel-id>.cfargotunnel.com`, and that name has no public address:

```
$ dig +short cfargotunnel.com A
                                     (no A record)
$ dig <tunnel-id>.cfargotunnel.com A +noall +comments
;; ->>HEADER<<- opcode: QUERY, status: NOERROR
;; flags: qr rd ra; QUERY: 1, ANSWER: 0, AUTHORITY: 0, ADDITIONAL: 1
```

`ANSWER: 0` -- the name resolves to nothing on the public internet; it
is only meaningful inside Cloudflare's edge. A DNS-only record would be
unreachable. The record must be proxied, so **125s is a hard ceiling**
for every request in this design.

### 3.3 Measured latency: a normal request is nowhere near the ceiling

Measured on this Mac against the configured model
(`REMBG_MODEL = "birefnet-general-lite"`, `config.py:32`), 1024x1024
input, three consecutive runs after warm-up:

| Step | Time |
| --- | --- |
| Session/model load (once, at service startup) | 2.2s |
| `remove()` per image | 4.60s / 6.07s / 6.16s |

With `NUMBER_OF_IMAGES = 2` (`config.py:38`), a sticker `/generate`
decomposes as roughly: prompt expansion ~2s, two images generated
concurrently ~10-20s, two background removals ~10-12s. **Typical total
25-40s**, comfortably inside 125s.

The existing 180s client timeout is therefore a defensive ceiling, not
the normal path. The team already tuned this once: `config.py:32`
records that the full `birefnet` model took ~80s per cutout and
"starves the event loop -- too slow for interactive use (phone times out
at 150s and users retry, wedging the queue)". The lite model was chosen
to fix exactly that.

### 3.4 An Individual developer account can host internal testers

App Store Connect Help states: "If you're enrolled as an individual and
add users in App Store Connect, users receive access only to App Store
Connect and are not considered part of your team in the Apple Developer
Program." Internal testers are drawn from App Store Connect users, up
to 100, and internal testing requires no Beta App Review.

Consequence, and this is the operationally important half: added users
get **App Store Connect access only, not Developer Portal access**. On
an Individual account the Account Holder remains the only identity that
can create distribution certificates and provisioning profiles. See
section 9.

Sources:
<https://developer.apple.com/help/app-store-connect/test-a-beta-version/testflight-overview/>,
<https://www.developer.apple.com/help/app-store-connect/test-a-beta-version/add-internal-testers>

### 3.5 `uvicorn --app-dir` replaces the working directory, it does not add to it

uvicorn 0.51.0 `main.py:548-549` performs a single
`sys.path.insert(0, app_dir)`, and `--app-dir` defaults to `""` (the
working directory, `main.py:361`). Passing an explicit `--app-dir`
therefore inserts that path *instead of* the working directory. Since
`text`, `image`, and `voice` are all launched as `main:app` with their
`main.py` in the working directory, adding `--app-dir "$API_ROOT"` to
them would break app resolution. `PYTHONPATH` is used instead
(section 5).

### 3.6 Corrected assumption: the Imagen fallback costs one round trip, once

`_call_model` (`main.py:31-48`) sets a module-level `_imagen_unavailable`
flag the first time Imagen 4 returns 404 and skips it from then on. The
fallback to `gemini-2.5-flash-image` does not add a failed round trip to
every request, only to the first one after a restart. No design
accommodation is needed.

## 4. Architecture: domain and tunnel topology

### 4.1 Domain

Register a domain through Cloudflare Registrar so it lands in a
Cloudflare zone directly, skipping a nameserver migration. Recommend a
`.app` or `.dev` TLD purely on cost (~USD 14/yr against ~USD 70+/yr for
`.ai`). The bundle identifier `ai.miranote.app` does not need a matching
domain; Apple does not verify it.

All beta hosts live under a `beta` second level. `<domain>` below stands
in for the registered name.

### 4.2 Tunnel

Install `cloudflared` on the Mac and create a **named** tunnel. A
`trycloudflare.com` quick tunnel is not an option: its URL changes on
every restart, which is fatal for a hostname compiled into a shipped
TestFlight build.

```yaml
# ~/.cloudflared/config.yml
tunnel: miranote-beta
credentials-file: ~/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: text.beta.<domain>
    service: http://localhost:8001
  - hostname: image.beta.<domain>
    service: http://localhost:8002
  - hostname: chat.beta.<domain>
    service: http://localhost:8003
  - hostname: voice.beta.<domain>
    service: http://localhost:8005
  - service: http_status:404
```

Hostname-per-service is chosen over a single host with path prefixes
because it keeps every existing route (`/chat`, `/generate`,
`/transcribe`, ...) unchanged. The path-prefix alternative would require
editing URL construction in all four iOS service classes and adding
either a FastAPI `root_path` or a tunnel-side rewrite -- an extra
mapping layer with nothing to show for it.

Four CNAMEs are created by `cloudflared tunnel route dns`. TLS
terminates at Cloudflare; no certificate is handled on the Mac.

The credentials file (`~/.cloudflared/<tunnel-id>.json`) is the only
recoverable state of the tunnel: if it is lost, the tunnel must be
deleted, recreated, and its four CNAMEs re-pointed. Back it up to a
password manager when the tunnel is created.

### 4.3 Two independent lifecycles

The tunnel and the backends are managed separately and must not be
coupled:

- **Tunnel:** installed as a launchd service (`cloudflared service
  install`) so it survives reboot and restarts on crash. Testers open
  the app at unpredictable times.
- **Backends:** stay on the existing manual `scripts/start_backends.sh`.

The consequence is a specific, expected failure mode: tunnel up but
backends down yields **502**, and that must be a legible message rather
than a raw status code (section 8.3). New `scripts/start_tunnel.sh` and
`scripts/stop_tunnel.sh` mirror the existing pair; `start_tunnel.sh`
also starts `caffeinate`, because a sleeping Mac takes the tunnel down
with it (section 8.3) and the tunnel must answer even when no backend
is up.

## 5. Backend auth layer

New `shared/beta_auth.py` at the repository root: a FastAPI dependency
that validates `Authorization: Bearer <token>` against `BETA_TOKENS`,
read from the environment with the `load_dotenv()` pattern the POCs
already use.

`BETA_TOKENS` is **comma-separated and accepts several valid tokens at
once**. Rotation is otherwise all-or-nothing: every tester is cut off
the instant the token changes. With a list, a new token is added first,
builds go out, and the old one is removed afterwards.

The four POCs have independent virtualenvs and no shared package. They
are given access to `shared/` by exporting `PYTHONPATH="$API_ROOT"` in
`scripts/start_backends.sh`, leaving `--app-dir` at its default so both
the working directory and the repository root stay on `sys.path`
(section 3.5).

`/health` is explicitly exempt from auth. All four services expose it
and `start_backends.sh` polls it for readiness; requiring a token there
would break the startup check for no benefit.

`beta_auth.py` also carries a per-token rate limit: an in-memory sliding
window, e.g. 30 requests per minute per token, returning 429 when
exceeded. The limit is deliberately cheap rather than precise. A token
can be extracted from the IPA (section 11), and without a limit an
extracted token spends DeepSeek and Vertex credits at full speed or
saturates the Mac's CPU. The limit doubles as protection against retry
storms (section 7).

## 6. Concurrency defect in `/generate`

`poc/image-generation/main.py:300` calls `remove(raw,
session=_rembg_session)` directly inside an `async def` handler. Every
other CPU-bound call in that file is wrapped -- `main.py:125` and
`main.py:352` both use `await asyncio.to_thread(remove, ...)`, and there
are 13 such wrapped call sites in total. Line 300 is the only unwrapped
one, and `_erode_alpha` on line 302 has the same problem.

The effect is that a sticker generation blocks the event loop for the
full duration of background removal -- measured at 10-12s for two
images (section 3.3) -- freezing every other in-flight request,
including `/health`.

This is invisible with one tester on a LAN, which is why it survived.
It is not invisible with ten. Ten concurrent generations serialise into
roughly 120s of pure event-loop blocking, which lands precisely on the
125s ceiling from section 3.1. **The one-line omission is what would
convert a comfortable 30s operation into a timeout.**

Fix: wrap both calls in `asyncio.to_thread`, matching the file's own
established pattern. Blocking becomes CPU contention across the thread
pool instead of serialisation on the event loop.

The fix removes serialisation but not saturation: ten concurrent
generations still queue roughly 120s of CPU work, and on an 8-core Mac
oversubscribed threads slow down every request, each of which may then
miss the 110s client budget. `/generate` therefore also gets an
`asyncio.Semaphore` (initial value 3), acquired at the top of the
handler and released in `finally`. Requests beyond the cap wait in the
event loop instead of fighting for cores, which bounds the worst case
and caps concurrent calls to the Vertex image API. Section 12 makes the
cap prove itself with a load test.

## 7. Timeout budget

The client is made to give up **before** Cloudflare does, so a 524 never
reaches a tester in the normal path.

| Setting | Current | New | Rationale |
| --- | --- | --- | --- |
| `ImageStudio.swift:91,163` request timeout | 180s | 110s | Below the 125s edge ceiling |
| `MiraCanvasCoordinator.swift:122` `imageTimeout` | 150s | 120s | Above the request timeout, so the transport error surfaces rather than being masked by the coordinator |
| Voice transcription request timeout | 60s (URLSession default; `LiveVoiceTranscriptionService.swift:47` sets none) | 110s | Measured worst 84.3s (section 13.3); the 60s default already times out ~1min recordings on today's LAN |

Today the ordering is inverted (request 180s > coordinator 150s), so the
coordinator's generic timeout always fires first for image work and the
real transport error is lost. The new ordering fixes that as well.

A tester now sees a clean `.timedOut` with a sensible message after
110s, instead of watching a spinner for 125s and receiving an
unexplained edge error.

On timeout the app does **not** auto-retry. The `config.py:32` comment
records that timed-out users retry and wedge the queue; with ten
testers a retry storm multiplies load on an already-saturated Mac. The
error message invites one manual retry instead.

## 8. iOS changes

### 8.1 Endpoint configuration

`MiraNoteConfig.Backend` (`MiraNoteConfig.swift:14-41`, device host at `:22`) is the single
source of every service URL, as its own comment states. The device
branch changes from the Bonjour host `Mengs-MacBook-Pro-2099.local` to
the HTTPS subdomains; `base(port:)` becomes `base(subdomain:)`. The
simulator branch keeps `http://localhost:<port>`. No caller changes,
because no path changes.

### 8.2 Auth header

`HTTPClient.send(_:)` (`HTTPClient.swift:43`) is the single choke point
for outbound traffic. `ImageStudio` and `LiveVoiceTranscriptionService`
build their own multipart `URLRequest`s but both hand them to
`client.send`, so injecting `Authorization` in `send` covers every call
site. The token is supplied through an xcconfig-injected Info.plist key
and read by `MiraNoteConfig`; neither the token nor the xcconfig file
that carries it is committed (the file is untracked/gitignored).

### 8.3 Error mapping

Four failure modes are now distinguishable and each needs its own
message. Without this, ten non-technical testers report every one of
them as "the app is broken" and triage is guesswork.

| Condition | Meaning | Remedy |
| --- | --- | --- |
| 502 | Tunnel up, backends not running | Run `start_backends.sh` on the Mac |
| 530 (Cloudflare 1033) | `cloudflared` is down, or the Mac is asleep | Restart the tunnel service; check the Mac is awake and plugged in |
| 401 | Token rotated; build carries the old one | Ship a new TestFlight build |
| `.timedOut` | Image work exceeded the budget | Retry |

`BackendError` already models `.server(status:detail:)` and `.timedOut`;
this is a change to `errorDescription`, not to the error type.

### 8.4 ATS and Info.plist

The `localhost` ATS exception **stays** -- the simulator still uses
plain HTTP against loopback. Device traffic is now HTTPS, so:

- Remove `NSAllowsLocalNetworking` and `NSLocalNetworkUsageDescription`.
  Devices no longer use mDNS, and leaving them in place would prompt
  testers for local-network permission the app does not need.
- Add `ITSAppUsesNonExemptEncryption = false`, so export compliance is
  not re-asked on every upload.

## 9. Signing and App Store Connect

- `project.yml:15` `DEVELOPMENT_TEAM` changes from `FBY8RBCZ9M`
  (Mengjia's free personal team) to the shared account's Team ID.
- `ai.miranote.app` is registered as a Bundle ID under that account.
- Because of section 3.4, being added as an App Store Connect user is
  **not sufficient to build and upload**. One of these must be arranged
  with the Account Holder:
  1. Sign in to Xcode with the shared Apple ID directly (simplest;
     requires access to its two-factor codes), or
  2. Have the Account Holder export a distribution certificate (`.p12`)
     and provisioning profile, then configure signing manually with
     automatic signing turned off.
- Option 1 is strongly preferred: hand-managed `.p12` signing is fragile
  across Xcode version changes.
- The TestFlight "What to Test" description must include one line stating
  that photos and recordings are processed by third-party AI services
  (Google, DeepSeek); testers should not assume everything stays on the
  Mac.
- The App Store Connect app name is globally unique. "MiraNote" may be
  taken; this is only discoverable when the app record is created, so a
  fallback name should be agreed in advance.
- Uploads go through the Xcode Organizer. CI-driven upload is not worth
  the setup at this stage.

## 10. App icon and versioning

The project has no asset catalog at all -- `App/Resources` contains only
fonts. App Store Connect rejects uploads without a 1024x1024 icon, so
this is a hard blocker.

A placeholder is sufficient: internal TestFlight builds are not
reviewed. It will be generated from the app's existing design tokens --
`Palette.swift` (`paper #F4F0E7`, `ink #201C16`, `tan #C9B295`) and the
bundled Fraunces typeface -- as
`App/Resources/Assets.xcassets/AppIcon.appiconset`. `project.yml`
already lists `App/Resources` in `sources`, so xcodegen picks it up. A
designed icon should replace it before any external testing.

`App/Info.plist:19` hardcodes `CFBundleVersion` to `1`. TestFlight
requires a strictly increasing build number per upload, so it becomes
`$(CURRENT_PROJECT_VERSION)` with `CFBundleShortVersionString` as
`$(MARKETING_VERSION)`, both set in `project.yml`. Manual increments are
fine at this scale.

## 11. Operational reality

Two costs are inherent to this design and are accepted deliberately.

**The Mac is a single point of failure.** It must stay awake, powered,
and online. The existing `caffeinate` in `start_backends.sh` only covers
the period while the backends run, and `start_tunnel.sh` adds its own
(section 4.3); the Mac still must stay plugged in and unsleeping for the
whole beta window. If the Mac sleeps, loses network, or reboots, every
tester is down at once. This is tolerable for ten internal testers and
is the main reason this design is labelled a first-pass rather than a
durable deployment.

**A token compiled into the app can be extracted.** Anyone with the IPA
can pull it and call the backends directly, spending DeepSeek and Vertex
credits. Accepted at internal-beta scale, with three mitigations: a
**budget alert on the Vertex project is required, not optional**, the
multi-token design in section 5 keeps rotation cheap, and the per-token
rate limit in section 5 bounds how fast an extracted token can burn.

**Nobody is watching.** Backend crashes and post-reboot failures are
currently discovered by tester complaints. Add a free uptime monitor
(e.g. UptimeRobot) pinging the four `/health` endpoints through the
tunnel every 5 minutes with phone alerts. Zero code, five minutes of
setup.

**Internal builds expire after 90 days.** TestFlight refuses to launch
an expired build, so re-upload at least every ~60 days (a calendar
reminder, not CI). The kill switch for abuse is `stop_tunnel.sh` plus
deleting the active token: both cut access instantly.

## 12. Testing strategy

- Backend (pytest): no token yields 401; wrong token yields 401; correct
  token yields 200; `/health` without a token yields 200.
- Backend (regression): a request to `/generate` does not block a
  concurrent `/health`, which fails before the section 6 fix and passes
  after it.
- iOS: the existing `URLProtocol` stub used by `HTTPClient` tests gains
  an assertion that `Authorization` is set on requests built by all
  three paths (`postJSON`, and the two multipart builders).
- End to end: `curl https://<service>.beta.<domain>/health` for all
  four hosts.
- Backend (load): ten concurrent sticker `/generate` requests, p95
  latency < 110s and no request over 125s. This is the test that proves
  the section 6 semaphore; run it before and after the fix.
- Backend (rate limit): requests beyond the per-token limit yield 429,
  and the window counts correctly across a burst.

## 13. Latency: /cutout and /transcribe measured

### 13.1 /cutout -- measured 2026-09-04

Measured on this Mac against the live service (`127.0.0.1:8002`) with
the production default mode `hybrid_sam_prebg_gray`, three runs per case
after warm-up. All models preload at service startup (rembg,
GroundingDINO, SAM-2); the ~40s cold start happens once, not per
request. Both disambiguation paths were exercised: the photos took the
DINO+Gemini union path (IoU 0.59-0.75) and the small-subject image the
Gemini-only path.

| Case | Latency (3 runs) | Median |
| --- | --- | --- |
| Hybrid cutout, person photo | 30.8s / 32.1s / 36.8s | ~32s |
| Hybrid cutout, small subject | 17.4s / 23.0s / 24.4s | ~23s |
| Hybrid cutout, cartoon | 29.1s (one run) | ~29s |
| Auto (rembg only), person photo | 8.6s / 11.3s / 22.0s | ~11s |
| Auto (rembg only), cartoon | 7.3s / 8.5s / 8.9s | ~8s |

The worst observed value (36.8s) sits roughly 3x under the 110s client
budget. With the section 6 semaphore capped at 3 concurrent generations,
CPU contention is bounded and the worst case stays well inside the 125s
edge ceiling. One outlier (22.0s auto on a photo that otherwise
measures 8-11s) appeared immediately after sustained load -- exactly the
saturation the semaphore exists to prevent. The async-job design is
therefore ruled out; sections 3.3 and 6 stand as written.

### 13.2 /cutout API quirk: prompt travels in the query string

The `prompt` parameter of `POST /cutout` binds from the query string
(`?prompt=person`), not from a multipart form field: a form field named
`prompt` is silently ignored and the request falls back to auto mode.
The iOS client already sends it correctly. Recorded here so backend
callers do not rediscover it the hard way.

### 13.3 /transcribe -- measured 2026-09-04

Measured on this Mac against the live service (`127.0.0.1:8005`) with
the production parameters the iOS client sends (`correct=true`,
`with_emotion=false`), on synthesized speech files of 10s, 1min, 3min,
and 5min:

| Recording | Production total | Whisper alone (`correct=false`) |
| --- | --- | --- |
| 10s | 12.6s | 0.33s |
| 1min | 53.7s / 74.4s | 1.7s |
| 3min | 82.0s / 84.3s | -- |
| 5min | 73.7s / 82.0s | 16.5s |
| 5min, `lang=auto` (two decodes) | 82.1s | -- |

The dominant cost is the DeepSeek correction call, a roughly flat ~60s
per request regardless of recording length; Whisper itself is fast (5min
of audio transcribes in ~16s). The whisper model loads lazily on the
first request, adding ~9s to the first call after service start.

Against the budgets: the worst observed value (84.3s) fits inside the
110s client budget with margin, but the iOS voice client's implicit 60s
URLSession default already fails ~1min recordings on today's LAN -- the
explicit 110s timeout in section 7 is required, not optional. Two
caveats: the ~60s correction cost is fixed, so recordings beyond ~15min
approach the 110s budget (accepted at beta scale; the long-term fix is
a correction-specific timeout or a smaller `max_tokens`), and
`lang=auto` costs no measurable extra time because the two Whisper
decodes are cheap next to the correction call.

## 14. Implementation order

1. Measure `/cutout` latency locally -- done 2026-09-04 (section 13.1):
   worst observed 36.8s, well inside the 110s budget.
2. Measure `/transcribe` latency -- done 2026-09-04 (section 13.3):
   worst observed 84.3s; the voice row in section 7 is filled in.
3. Fix the `asyncio.to_thread` omission, add the `/generate` semaphore,
   and add the concurrency regression and load tests (sections 6, 12).
4. Add `shared/beta_auth.py` with rate limiting, wire `PYTHONPATH`,
   exempt `/health`, add tests (sections 5, 12).
5. Register the domain; create and route the named tunnel; install it as
   a service; back up the credentials file (section 4).
6. iOS: endpoint config, auth header, timeout budget, no auto-retry,
   error messages, ATS cleanup (sections 7-8).
7. Generate the placeholder icon; switch the version keys to build
   settings (section 10).
8. Arrange signing access; create the App Store Connect record; upload
   the first build (section 9).
9. Add internal testers, configure uptime monitoring, and distribute
   (sections 11-12).

Steps 3 and 4 are backend-only and mergeable before any of the
distribution work begins.

## 15. Revision history

- v1 (2026-09-03): initial draft.
- v2 (2026-09-04): red-team review. Added per-token rate limiting
  (section 5), the `/generate` concurrency cap and load test (sections
  6, 12), voice-path timeout coverage (sections 7, 13), tunnel
  keep-awake and credentials backup (sections 4.2-4.3), uptime
  monitoring, the 90-day build expiry, and the kill switch (section
  11), the signing recommendation and privacy line (section 9), and
  corrected drifted line references (sections 3.3, 7).
- v3 (2026-09-04): measured `/cutout` latency (section 13.1) and
  recorded the prompt query-string quirk (section 13.2). Implementation
  step 1 marked done; the async-job redesign is ruled out.
- v4 (2026-09-04): measured `/transcribe` latency (section 13.3) and
  filled the voice timeout row in section 7 (110s). Implementation
  step 2 marked done; both measurement tasks are closed.

# TestFlight internal beta -- hosting and distribution design

- **Date:** 2026-09-03
- **Author:** mengjia (Claude-assisted)
- **Status:** Draft, revised after red-team review
- **Revision:** v5 (2026-09-12) -- corrects four statements the
  implementation disproved and records what the machine actually looks
  like; see section 15.
- **Scope:** hosting and the client changes that talk to it. Backend
  changes in `poc/image-generation/main.py`, all four POC `main.py`
  files and `scripts/start_backends.sh`, new `beta_auth.py` and tunnel
  scripts, plus a companion PR in `miranote-ios` (`MiraNoteConfig.swift`,
  `HTTPClient.swift`, `ImageStudio.swift`,
  `MiraCanvasCoordinator.swift`, `App/Info.plist`). No product features
  change.
- **Ownership:** TestFlight distribution is **not owned by this team**.
  Sections 9 and 10, and the build-expiry note in section 11, describe
  work that now belongs to whoever runs distribution: signing, the App
  Store Connect record, the app icon, version keys, tester management
  and re-uploads. They are kept here because the constraints in them
  were researched and still hold, notably section 3.4 on what an
  Individual account can and cannot do. Everything else -- hosting, the
  tunnel, auth, timeouts and error mapping -- is this team's.
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

The registered domain is `miranote.app`.

**Beta hosts are one level deep**: `beta-text`, `beta-image`,
`beta-chat` and `beta-voice` under `miranote.app`. The natural shape --
`text.beta.miranote.app` -- does not work, and the reason is a hard one.
Cloudflare's free Universal SSL signs a certificate covering the apex
and a single-label wildcard, and nothing else:

```
$ echo | openssl s_client -connect tlstest.miranote.app:443 ... | openssl x509 -ext subjectAltName
X509v3 Subject Alternative Name:
    DNS:miranote.app, DNS:*.miranote.app
```

`*.miranote.app` does not match `text.beta.miranote.app`, so the edge
refuses the handshake before any request is made. Measured side by side
against the same tunnel and zone:

| Hostname | Result |
| --- | --- |
| `tlstest.miranote.app` (one label) | HTTP 404, `ssl_verify=0` |
| `text.beta.miranote.app` (two labels) | HTTP 000, `ssl_verify=1` |

A multi-label wildcard needs the paid Advanced Certificate Manager. The
flat form is free and costs nothing but a hyphen, and it leaves the
unprefixed names (`text.miranote.app`) available for production later.

The hostname is compiled into shipped TestFlight builds, so changing it
after distribution starts costs every tester a new build.

### 4.2 Tunnel

Install `cloudflared` on the Mac and create a **named** tunnel. A
`trycloudflare.com` quick tunnel is not an option: its URL changes on
every restart, which is fatal for a hostname compiled into a shipped
TestFlight build.

```yaml
# ~/.cloudflared/miranote.yml   (NOT config.yml -- see below)
tunnel: <tunnel-id>
credentials-file: /Users/<user>/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: beta-text.miranote.app
    service: http://127.0.0.1:8001
  - hostname: beta-image.miranote.app
    service: http://127.0.0.1:8002
  - hostname: beta-chat.miranote.app
    service: http://127.0.0.1:8003
  - hostname: beta-voice.miranote.app
    service: http://127.0.0.1:8005
  - service: http_status:404
```

The origin is `127.0.0.1`, not `localhost`. `localhost` also resolves to
`::1`, and a service bound only to IPv4 answers an IPv6 connection with
502. The other tunnel on this machine carries the same workaround in its
own config, having hit it first.

**This machine runs two unrelated tunnels, from two different Cloudflare
accounts.** DASGPT (`dasgpt.stream`) owns the default
`~/.cloudflared/config.yml`; MiraNote uses `~/.cloudflared/miranote.yml`.
Three consequences, all measured:

- Every MiraNote cloudflared command must pass `--config` explicitly.
  Loading the default config while naming this tunnel fails with
  `Tunnel not found`; the same command with `--config` succeeds, and so
  does the default config when the tunnel is named by UUID instead.
- Management commands select the account with `--origincert` /
  `TUNNEL_ORIGIN_CERT`, pointing at that account's saved certificate.
  Swapping `cert.pem` by hand, which is what was happening before, is
  unnecessary and is what made the state confusing.
- Running a tunnel needs no certificate at all, only the credentials
  file, so the two tunnels run side by side without interfering.

Hostname-per-service is chosen over a single host with path prefixes
because it keeps every existing route (`/chat`, `/generate`,
`/transcribe`, ...) unchanged. The path-prefix alternative would require
editing URL construction in all four iOS service classes and adding
either a FastAPI `root_path` or a tunnel-side rewrite -- an extra
mapping layer with nothing to show for it.

Four CNAMEs are created by `cloudflared tunnel route dns`, and they must
stay proxied. TLS terminates at Cloudflare; no certificate is handled on
the Mac.

The credentials file (`~/.cloudflared/<tunnel-id>.json`) is the only
recoverable state of the tunnel: if it is lost, the tunnel must be
deleted, recreated, and its four CNAMEs re-pointed. Back it up to a
password manager when the tunnel is created.

### 4.3 Two independent lifecycles

The tunnel and the backends are managed separately and must not be
coupled:

- **Tunnel:** started by `scripts/start_tunnel.sh`, which also holds a
  `caffeinate` tied to the tunnel process. Testers open the app at
  unpredictable times, so it must also survive reboot -- but
  `cloudflared service install` cannot provide that here. It installs a
  single machine-wide service wrapped around the default config, which
  belongs to the other tunnel. A per-tunnel launchd plist is required
  instead, and is still outstanding.
- **Backends:** stay on the existing manual `scripts/start_backends.sh`.

The consequence is a specific, expected failure mode: tunnel up but
backends down yields **502**, and that must be a legible message rather
than a raw status code (section 8.3). New `scripts/start_tunnel.sh` and
`scripts/stop_tunnel.sh` mirror the existing pair; `start_tunnel.sh`
also starts `caffeinate`, because a sleeping Mac takes the tunnel down
with it (section 8.3) and the tunnel must answer even when no backend
is up.

## 5. Backend auth layer

New `beta_auth.py` at the repository root, validating
`Authorization: Bearer <token>` against `BETA_TOKENS`.

**Not `shared/beta_auth.py`.** `poc/image-generation` already owns a
local package named `shared`, and a POC's working directory sorts ahead
of `PYTHONPATH` on `sys.path`, so from that service `import shared`
resolves to its own package and `shared.beta_auth` raises
`ModuleNotFoundError`. Image generation is exactly one of the four
services that needs the gate, and the only one where the original plan
would have failed.

**Installed as middleware, not as a FastAPI dependency.** A mounted
sub-application does not inherit `FastAPI(dependencies=[...])`:
measured, a route answers 401 while a mounted file answers 200 and
serves its contents. `voice-to-text` mounts `StaticFiles` at `/` with
`html=True`, which catches every path no route matched, and
`text-clean-expand` mounts at `/static`. A dependency would have left
both served publicly.

Two mechanics follow from that choice and are easy to get wrong:

- The gate returns a response rather than raising `HTTPException`.
  Middleware runs outside the exception handlers, so a raise produces
  500 instead of the intended status.
- `install(app)` runs **before** `CORSMiddleware` is added. Starlette
  makes the most recently added middleware outermost, and a browser
  preflight carries no `Authorization` header, so a gate outside CORS
  rejects every preflight with 401.

`BETA_TOKENS` is read from a repository-root `.env` loaded by absolute
path. A bare `load_dotenv()` from a POC working directory does not reach
the repository root -- measured -- so the alternative was copying the
shared token into all four POC `.env` files.

`BETA_TOKENS` is **comma-separated and accepts several valid tokens at
once**. Rotation is otherwise all-or-nothing: every tester is cut off
the instant the token changes. With a list, a new token is added first,
builds go out, and the old one is removed afterwards.

The four POCs have independent virtualenvs and no package in common.
They reach `beta_auth` through `PYTHONPATH="$API_ROOT"` exported in
`scripts/start_backends.sh`, leaving `--app-dir` at its default so both
the working directory and the repository root stay on `sys.path`
(section 3.5).

With no token configured nothing is accepted, and `install()` says so at
startup. Fail-open is not an option once the tunnel is public, and a
service that rejects everything with nothing in the log to explain it is
the worse of the two silent failures.

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
the HTTPS hosts; `base(port:)` becomes `base(host:)`. The simulator
branch keeps `http://localhost:<port>`. No caller changes, because no
path changes.

| Service | Device | Simulator |
| --- | --- | --- |
| text | `https://beta-text.miranote.app` | `http://localhost:8001` |
| image | `https://beta-image.miranote.app` | `http://localhost:8002` |
| chat | `https://beta-chat.miranote.app` | `http://localhost:8003` |
| voice | `https://beta-voice.miranote.app` | `http://localhost:8005` |

The simulator path needs the token too: `HTTPClient.send` injects the
header on every request, and the backends now require it on loopback as
well.

### 8.2 Auth header

`HTTPClient.send(_:)` (`HTTPClient.swift:43`) is the single choke point
for outbound traffic. `ImageStudio` and `LiveVoiceTranscriptionService`
build their own multipart `URLRequest`s but both hand them to
`client.send`, so injecting `Authorization` in `send` covers every call
site. The token is supplied through an xcconfig-injected Info.plist key
and read by `MiraNoteConfig`; neither the token nor the xcconfig file
that carries it is committed (the file is untracked/gitignored).

### 8.3 Error mapping

Six failure modes are now distinguishable and each needs its own
message. Without this, ten non-technical testers report every one of
them as "the app is broken" and triage is guesswork.

| Condition | Meaning | Remedy |
| --- | --- | --- |
| 502 | Tunnel up, backends not running | Run `start_backends.sh` on the Mac |
| 530 (Cloudflare 1033) | `cloudflared` is down, or the Mac is asleep | Restart the tunnel; check the Mac is awake and plugged in |
| 401 | No token, or the build carries a rotated one | Ship a build with the current token |
| 429 | This build's token is over its per-minute budget | Wait a minute; it clears on its own |
| 503 | The image provider is out of quota | Wait and try again; not the app's fault and not fixable from the phone |
| `.timedOut` | Image work exceeded the budget | Retry once, manually |

429 and 503 are easy to confuse and have different causes: 429 is our
own rate limit, counted per token, and 503 is the upstream image
provider refusing us. The remedy is the same, but the second is not
something rotating a token or restarting anything will fix.

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

> **Not owned by this team.** Distribution moved elsewhere. This section
> is kept because the constraints in it were researched and still hold --
> section 3.4 in particular, on why adding an App Store Connect user does
> not grant the ability to build and upload. Hand it to whoever runs
> distribution rather than treating it as a task list here.


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

> **Not owned by this team.** Both items exist only to satisfy an App
> Store Connect upload: a 1024x1024 icon is a hard blocker for uploading,
> and TestFlight requires a strictly increasing build number.


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
reminder, not CI). That re-upload belongs to whoever owns distribution,
not to this team.

The kill switch for abuse stays here and is ours: `stop_tunnel.sh` plus
removing the active token from `BETA_TOKENS`. Either cuts access
instantly, and the token list being plural means one tester's build can
be cut off without cutting off the rest.

## 12. Testing strategy

- Backend (pytest): no token yields 401; wrong token yields 401; correct
  token yields 200; `/health` without a token yields 200.
- Backend (regression): a request to `/generate` does not block a
  concurrent `/health`, which fails before the section 6 fix and passes
  after it.
- iOS: the existing `URLProtocol` stub used by `HTTPClient` tests gains
  an assertion that `Authorization` is set on requests built by all
  three paths (`postJSON`, and the two multipart builders).
- End to end: `curl https://beta-<service>.miranote.app/health` for all
  four hosts, plus one authenticated and one unauthenticated call to a
  real route on each, and one call to a mounted path on the two
  services that mount a UI.
- Backend (load): ten concurrent sticker `/generate` requests, p95
  latency < 110s and no request over 125s. This is the test that proves
  the section 6 semaphore.

  **Run 2026-09-11 and it did not produce that proof.** Three requests
  succeeded and seven were rejected by the image provider with
  `429 RESOURCE_EXHAUSTED`, so there is no p95 over ten completed
  generations. What the run did establish is that the cap is not the
  binding constraint at this scale: provider quota is reached first, the
  three that completed took 33.9s, 34.2s and 49.4s against a 110s
  budget, and `/health` answered 65 of 65 probes with a worst case of
  25ms throughout. The semaphore value of 3 remains unvalidated and must
  be re-measured once quota is resolved (api #53). That run is also what
  produced the 503 mapping in section 8.3.
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

1. Measure `/cutout` latency -- **done** 2026-09-04 (section 13.1):
   worst observed 36.8s, well inside the 110s budget.
2. Measure `/transcribe` latency -- **done** 2026-09-04 (section 13.3):
   worst observed 84.3s; the voice row in section 7 is filled in.
3. Fix the `asyncio.to_thread` omission, add the `/generate` semaphore
   and the concurrency tests -- **done** (api #51).
4. Add `beta_auth` with rate limiting, wire `PYTHONPATH`, exempt
   `/health` -- **done** (api #52), and install it on all four services
   while switching the bind to loopback -- **done** (api #58).
5. Register the domain, create and route the named tunnel, script its
   lifecycle -- **done** (api #56). The four hostnames resolve, TLS
   terminates, and an unauthenticated request is refused. **Outstanding:
   a per-tunnel launchd plist**, without which a reboot takes the beta
   down until someone runs the script by hand (section 4.3).
6. iOS: endpoint config, auth header, timeout budget, no auto-retry,
   error messages, ATS cleanup (sections 7-8). **Next.**
7. Placeholder icon and version keys -- **not owned by this team**
   (section 10).
8. Signing, App Store Connect record, first upload -- **not owned by
   this team** (section 9).
9. Internal testers and distribution -- **not owned by this team**.
   Uptime monitoring of the four `/health` endpoints stays with us
   (section 11).

Unplanned but done along the way: quota rejections from the image
provider now map to 503 rather than a bare 500 (api #54, section 8.3),
found by running the section 12 load test.

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
- v5 (2026-09-12): corrects four statements the implementation
  disproved, each forced by a measurement rather than a preference.
  Beta hostnames are one label deep (`beta-text.miranote.app`) because
  free Universal SSL does not cover a second label and the two-level
  form fails the TLS handshake outright (section 4.1). The auth module
  is a top-level `beta_auth`, not `shared/beta_auth`, because
  `poc/image-generation` owns a local package named `shared` that
  shadows the repository root from its own working directory -- and it
  is one of the four services that needs the gate (section 5). The gate
  is middleware rather than a route dependency, because a dependency
  does not reach mounted sub-applications and `voice-to-text` mounts at
  `/` (section 5). Tunnel ingress points at `127.0.0.1`, not
  `localhost`, which also resolves to `::1` (section 4.2).

  Added: this machine runs two tunnels from two Cloudflare accounts,
  which makes `--config` mandatory and `cloudflared service install`
  unusable (sections 4.2-4.3); 429 and 503 rows in the error mapping
  table (section 8.3); the concrete endpoint table for the iOS work
  (section 8.1). Marked sections 9 and 10 and the build-expiry note as
  owned outside this team; the kill switch stays here. Implementation
  steps 3, 4 and most of 5 are done; a per-tunnel launchd plist is the
  one piece of step 5 still outstanding.

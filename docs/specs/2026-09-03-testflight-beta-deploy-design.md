# TestFlight internal beta -- hosting and distribution design

- **Date:** 2026-09-03
- **Author:** mengjia (Claude-assisted)
- **Status:** Draft, awaiting implementation plan
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
(`REMBG_MODEL = "birefnet-general-lite"`, `config.py:12`), 1024x1024
input, three consecutive runs after warm-up:

| Step | Time |
| --- | --- |
| Session/model load (once, at service startup) | 2.2s |
| `remove()` per image | 4.60s / 6.07s / 6.16s |

With `NUMBER_OF_IMAGES = 2` (`config.py:18`), a sticker `/generate`
decomposes as roughly: prompt expansion ~2s, two images generated
concurrently ~10-20s, two background removals ~10-12s. **Typical total
25-40s**, comfortably inside 125s.

The existing 180s client timeout is therefore a defensive ceiling, not
the normal path. The team already tuned this once: `config.py:12`
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
`scripts/stop_tunnel.sh` mirror the existing pair.

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

## 7. Timeout budget

The client is made to give up **before** Cloudflare does, so a 524 never
reaches a tester in the normal path.

| Setting | Current | New | Rationale |
| --- | --- | --- | --- |
| `ImageStudio.swift:91,163` request timeout | 180s | 110s | Below the 125s edge ceiling |
| `MiraCanvasCoordinator.swift:110` `imageTimeout` | 150s | 120s | Above the request timeout, so the transport error surfaces rather than being masked by the coordinator |

Today the ordering is inverted (request 180s > coordinator 150s), so the
coordinator's generic timeout always fires first for image work and the
real transport error is lost. The new ordering fixes that as well.

A tester now sees a clean `.timedOut` with a sensible message after
110s, instead of watching a spinner for 125s and receiving an
unexplained edge error.

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
and read by `MiraNoteConfig`; it is not committed.

### 8.3 Error mapping

Four failure modes are now distinguishable and each needs its own
message. Without this, ten non-technical testers report every one of
them as "the app is broken" and triage is guesswork.

| Condition | Meaning | Remedy |
| --- | --- | --- |
| 502 | Tunnel up, backends not running | Run `start_backends.sh` on the Mac |
| 530 (Cloudflare 1033) | `cloudflared` itself is down | Restart the tunnel service |
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
the period while the backends run. If the Mac sleeps, loses network, or
reboots, every tester is down at once. This is tolerable for ten
internal testers and is the main reason this design is labelled a
first-pass rather than a durable deployment.

**A token compiled into the app can be extracted.** Anyone with the IPA
can pull it and call the backends directly, spending DeepSeek and Vertex
credits. Accepted at internal-beta scale, with two mitigations: a
**budget alert on the Vertex project is required, not optional**, and
the multi-token design in section 5 keeps rotation cheap.

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

## 13. Open question

`/cutout` latency has **not** been measured. It runs a heavier pipeline
than `/generate` -- GroundingDINO plus Gemini bounding-box detection
concurrently, then SAM-2 segmentation -- and measuring it requires
downloading the SAM-2 checkpoint and the DINO model plus a billed Gemini
call, which was not worth doing during design.

It is not a blocker: the 110s client timeout from section 7 bounds the
failure cleanly either way. But **measuring it is the first task of
implementation**, before the tunnel is exposed, because it is the one
number that could still force the async-job design that sections 3.3 and
6 otherwise rule out.

## 14. Implementation order

1. Measure `/cutout` latency locally (section 13).
2. Fix the `asyncio.to_thread` omission and add the concurrency
   regression test (section 6).
3. Add `shared/beta_auth.py`, wire `PYTHONPATH`, exempt `/health`,
   add tests (section 5).
4. Register the domain; create and route the named tunnel; install it as
   a service (section 4).
5. iOS: endpoint config, auth header, timeout budget, error messages,
   ATS cleanup (sections 7-8).
6. Generate the placeholder icon; switch the version keys to build
   settings (section 10).
7. Arrange signing access; create the App Store Connect record; upload
   the first build (section 9).
8. Add internal testers and distribute.

Steps 2 and 3 are backend-only and mergeable before any of the
distribution work begins.

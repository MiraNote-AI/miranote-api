# REVIEW.md -- guidance for reviewers of miranote-api (human or AI)

Read by the run-loop fresh-context check and pre-merge review, and by
anyone reviewing a pull request here. The rules live in `CONTRIBUTING.md`
and the verify-repo skill; this file says what failure looks like in this
repo and where to look hardest.

## What failure looks like here

Most damaging first.

1. **An open door.** A new route or service that skips the bearer-token
   gate and rate limit in `beta_auth.py`. The POC services are reachable
   over public HTTPS through a Cloudflare tunnel.
2. **A contract change the app does not know about.** A renamed or removed
   response field, or a changed status code or error shape, in an endpoint
   the iOS app calls. The app ships separately through TestFlight, so older
   builds keep calling the old contract.
3. **A fallback that reports success.** A retry or provider fallback that
   turns an empty image, a refusal or a provider error into a normal
   response, so the app shows nothing or the wrong thing and nobody is told
   why.
4. **Secrets in the open.** A key or token in code, in a log line or in a
   response; a committed `.env`.
5. **Code that breaks Python 3.9.** `X | None`, `match`, or other syntax and
   library calls newer than 3.9: the runtime is 3.9.
6. **Tests that leave the machine.** A test that reaches a real model, the
   network or a key. Every suite stays offline, with model calls stubbed.
7. **Serving before ready.** A service that accepts requests before its
   models or providers have loaded, or that hides a broken provider until
   the first request.

## Sensitive paths (review at `high` + `/security-review`)

- `beta_auth.py` -- the token gate and rate limit in front of every service
- `poc/*/main.py` -- the HTTP endpoints the app calls
- `poc/*/*client*.py` -- calls to model and speech providers
- `.env.example`, `start-all.sh`, `.github/workflows/`

## Skip

- `poc/*/demo_data/`

## How to report

- Every finding cites file:line evidence; never a guess from a name.
- Tag each finding IMPORTANT (breaks or misleads), NIT (preference) or
  PRE-EXISTING (not introduced by this diff). At most 3 NITs.
- Formatting is not a finding.
- Start the summary with the count; say plainly when there is nothing to
  fix.

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
- `poc/*/*client*.py`, `poc/photo-collections/llm.py`,
  `poc/image-generation/shared/vertex_client.py`,
  `poc/image-generation/generate/` (the DashScope and Gemini providers),
  `poc/*/config.py` -- calls to model and speech providers, and where
  their keys are read
- `.env.example`, `poc/*/.env.example` -- the token and provider-key names
- `scripts/`, `scripts/launchd/`, `start-all.sh` -- the beta deploy: the
  Cloudflare tunnel, launchd jobs that start and restart every backend; a
  wrong bind address or a lost env file takes the beta down or opens it
- `.github/workflows/`

## Skip

- `poc/*/demo_data/`
- `poc/image-generation/test_api.py` (a live smoke script, not a test),
  `poc/photo-collections/tools/`, `poc/retrieval/scripts/` -- manual
  scripts, never run by CI or the services. Not `vision_bench/`: it is
  the binary that serves `/cutout`.

## How to run the suites

`.github/workflows/tests.yml` runs these on every PR, one Python 3.9 venv
per suite (`pip install -r poc/<suite>/requirements.txt`), no keys:

```
PYTHONPATH=. LLM_API_KEY=x python -m pytest poc/chatbot/tests
PYTHONPATH=. python -m pytest tests    # beta_auth, in the chatbot venv
PYTHONPATH=. python -m pytest poc/retrieval/tests
```

and from `poc/voice-to-text`, `poc/text-clean-expand` and
`poc/photo-collections`: `PYTHONPATH=.:../.. python -m pytest tests/`.
The image-generation suite (Python 3.13) is not in CI yet: api#104.

## How to report

- Every finding cites file:line evidence; never a guess from a name.
- Tag each finding IMPORTANT (breaks or misleads), NIT (preference) or
  PRE-EXISTING (not introduced by this diff). At most 3 NITs.
- Formatting is not a finding.
- Start the summary with the count; say plainly when there is nothing to
  fix.

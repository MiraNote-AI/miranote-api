# photo-collections (port 8006)

Photo Trigger V1 backend. The app sends up to 10 small photos (768 px, no
EXIF) of one "life fragment" plus per-photo metadata; the service runs two
Qwen calls with program checks between them and returns the finished Chinese
push / story / ending content.

```
call A (look) -> check A -> templates + question menus -> call B (write)
  -> check B -> pick detail sentence + question -> memory
```

Every rule (prompts, schemas, checks, word lists, selection) comes from the
spec `feature-photo-trigger/v1/后端-AI调用-V1.md` and the question bank
`v1/题库-V1.md`. All Chinese text lives in `prompts/*.txt`.

## Setup

```bash
cd poc/photo-collections
python3.9 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # fill LLM_API_KEY (same DashScope key as voice-to-text)
```

The beta token is `BETA_TOKENS` in the repo-root `.env` (shared by every
service). Without it every request except `/health` answers 401.

## Run

```bash
# from the repo root
scripts/start_backends.sh            # all services, loopback only
# or just this one, from poc/photo-collections:
PYTHONPATH=../.. .venv/bin/uvicorn main:app --host 127.0.0.1 --port 8006
```

`GET /health` (no token) reports the startup probe: `llm` is `ok`,
`unconfigured` (no key) or `unreachable` (key / URL / model wrong).

## API

Submit (multipart, `Authorization: Bearer <token>`):

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  -F 'meta={"group_key":"2026-10-18_8F3A","date":"2026-10-18","weekday":7,
            "start_time":"14:03","end_time":"17:40","place":"达拉斯植物园",
            "raw_count":23,"covers_all":true,"recent_question_ids":["B4"],
            "photos":[{"id":"P1","taken_at":"2026-10-18T14:03:00-05:00",
                       "labels":["food_drink"],"faces":0,"favorite":false,
                       "edited":false,"represents":1}]}' \
  -F photos=@P1.jpg \
  localhost:8006/collections/compose
# -> 202 {"job_id": "..."}
```

Poll every 5 s:

```bash
curl -s -H "Authorization: Bearer $TOKEN" localhost:8006/collections/jobs/<job_id>
```

| Status | Body |
|---|---|
| 200 | `{"status": "running"}` |
| 200 | `{"status": "ok", "memory": {...}}` (shape: handoff section 5.2) |
| 200 | `{"status": "rejected", "reason": "sensitive" \| "skip_screenshots" \| "too_few_selected" \| "check_failed", "message": "..."}` |
| 200 | `{"status": "error", "reason": "timeout" \| "model_error", "message": "..."}` |
| 400 | `{"status": "error", "reason": "bad_request", "message": "..."}` on submit: meta not JSON / missing field / 0 or > 10 photos / filename stems != meta ids / a file > 5 MB. No job, no model call. |
| 404 | `{"status": "expired"}`: unknown job, or older than 15 min. Resubmit. |

Limits: call A 90 s, call B 30 s, each retried once; a job is capped at
240 s (`timeout`). Jobs live in memory for 15 min. Photos are never written
to disk by this service.

## Model call modes

Settled by `tools/probe_model.py` against `qwen3.7-flash` on 2026-10-31
(a synthetic image, no personal photo):

| Call | Mode | Request |
|---|---|---|
| A (look) | `thinking_stream` | `enable_thinking=true`, `thinking_budget=1500`, `stream=true` + `include_usage`, no `response_format`; the program parses the JSON |
| B (write) | `json_schema` | `enable_thinking=false`, `response_format` JSON Schema strict (AI doc 3.3) |

Both fallbacks (`json_object`) also worked and can be switched to with
`PHOTO_CALL_A_MODE` / `PHOTO_CALL_B_MODE` without a code change.

Cost uses the token counts the provider returns: input 0.2, output 0.8 CNY
per million tokens (thinking tokens are output).

## Readings of the spec (decided from the first real runs)

AI doc 4 is followed as written; where it is silent the code reads it so:

- **Nouns** are grounded when they are a substring of any subject, detail or
  text of the selected photos (jieba splits the same phrase differently in
  different contexts, so token-set membership rejected the spec's own good
  examples).
- **uses**: the particles 的 / 了 / 着 are dropped on both sides, and the
  `name(type)` / `文字：` labels the B prompt itself prints are removed before
  matching.
- **Peak bonus (+2)** scores only the photo ids the model cited; templates
  T1 / T2 are fallback levels (AI doc 6) and never take it.
- **Peak first (user decision, 2026-10-31):** after every check, if any
  surviving model sentence cites the peak photo, the detail is picked among
  those only (same scoring and tie-breaks); otherwise among all survivors,
  model sentences and T1 / T2 alike; then F0. Call B is asked for at least
  two sentences about the peak.
- **A photo id in a model sentence** ("P1, P3, ...") is rejected -- a sanity
  check, not an AI doc 4 rule.
- **A4's {画面} menu** (program-side, added after H1 round 4) only takes
  details with a sight verb (`prompts/sight_verbs.txt`: light, things,
  change), not people's actions; the prompt alone did not stop fills like
  "站在蓝色幕布前". A1 keeps every action phrase.
- **skip=true** from call A is a verdict (`skip_screenshots`, or `sensitive`
  when any photo has a sensitive flag), even with nothing selected.
- **T1's n** sums `represents` over every sent photo showing X, so "you took
  n" is true of the whole group.

## Log

One JSON line per job in `logs/photo-collections/compose.jsonl` (repo root,
git-ignored): raw A/B output, every rejected candidate and question with its
reason, the final picks, `bodySource` / `questionSource`, per-call latency,
tokens and cost. No image bytes, no key.

## Tests

```bash
cd poc/photo-collections
PYTHONPATH=.:../.. .venv/bin/python -m pytest tests -v
```

Offline: every model call is the `fake_model` fixture in `tests/conftest.py`.

## Tools (real network, not tests)

- `tools/probe_model.py` -- which request shapes the model accepts.
- `tools/run_h1.py` -- acceptance H1: runs the prototype's 80 photos through
  the running service and writes a summary table outside the repo.

# photo-collections (port 8006)

Photo Trigger V1 backend. The app sends up to 10 small photos (768 px, no
EXIF) of one "life fragment" plus per-photo metadata; the service runs three
Qwen calls with program checks between them and returns the finished Chinese
push / story / ending content.

```
call A (look) -> check A -> templates -> call B (write; sees the photos)
  -> program checks -> call C (check against the photos)
  -> pick detail sentence, up to 3 questions -> memory
```

Every rule (prompts, schemas, checks, word lists, selection) comes from the
spec `feature-photo-trigger/v1/后端-AI调用-V1.md`, as changed by the copy
refine handoff `v1/H-交接-VSCode-文案refine.md` (2026-11-04; it wins where
they differ). All Chinese text lives in `prompts/*.txt`.

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

## Deploy (the beta Mac behind the Cloudflare tunnel)

Once this service is merged into `dev`, on the Mac that runs the beta
(design: `docs/specs/2026-09-03-testflight-beta-deploy-design.md` 4.2):

1. Pull, then create `poc/photo-collections/.venv` and `.env` as in
   Setup (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`). The beta token is
   the existing `BETA_TOKENS` in the repo-root `.env`.
2. Start it with the others: `scripts/start_backends.sh` (8006 is in its
   list; it binds 127.0.0.1 only).
3. In `~/.cloudflared/miranote.yml` (outside the repo -- never
   `config.yml`, which belongs to another tunnel on that machine) add an
   ingress rule before the final 404 rule:

   ```yaml
     - hostname: beta-collections.miranote.app
       service: http://127.0.0.1:8006
     - service: http_status:404
   ```

   `127.0.0.1`, not `localhost`: `localhost` can resolve to `::1` and
   answer 502.
4. Create the DNS record with the MiraNote account's certificate:

   ```bash
   cloudflared --config ~/.cloudflared/miranote.yml \
     --origincert <path to the MiraNote account cert.pem> \
     tunnel route dns miranote-beta beta-collections.miranote.app
   ```

5. Restart the tunnel: `launchctl kickstart -k gui/$(id -u)/ai.miranote.beta-tunnel`
   (or `scripts/stop_tunnel.sh && scripts/start_tunnel.sh` when it is not
   run by launchd; `start_tunnel.sh` already checks `beta-collections`).
6. Check `https://beta-collections.miranote.app/health`: `status: ok` and
   `llm: ok`.

## API

Submit (multipart, `Authorization: Bearer <token>`):

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  -F 'meta={"group_key":"2026-10-18_8F3A","date":"2026-10-18","weekday":7,
            "start_time":"14:03","end_time":"17:40","place":"达拉斯植物园",
            "raw_count":23,"covers_all":true,"recent_question_types":["sound"],
            "bucket":"scenery","tz":"America/Chicago",
            "photos":[{"id":"P1","taken_at":"2026-10-18T14:03:00-05:00",
                       "labels":["food_drink"],"faces":0,"favorite":false,
                       "edited":false,"represents":1}]}' \
  -F photos=@P1.jpg \
  localhost:8006/collections/compose
# -> 202 {"job_id": "..."}
```

`meta.bucket` is required and must be one of the seven ids, in tie order:
`food` (美食饮品), `scenery` (自然风光), `animal` (宠物与动物), `city`
(城市街头), `indoor` (室内居家), `event` (活动演出), `other` (其他). The
phone decides it; `memory.bucket` is the Chinese name of that id. Labels
no longer pick a bucket (they still feed call A). A missing, non-string or
unknown `bucket` answers 400. `meta.tz` (an IANA zone name such as
`America/Chicago`) is optional (it may be null or absent), a string of at most 64
characters, and is only written to the log; anything else answers 400.

Poll every 5 s:

```bash
curl -s -H "Authorization: Bearer $TOKEN" localhost:8006/collections/jobs/<job_id>
```

Poll many groups in one request (the phone sends at most 7 ids):

```bash
curl -s -H "Authorization: Bearer $TOKEN" "localhost:8006/collections/jobs?ids=<id1>,<id2>"
# -> 200 {"jobs": {"<id1>": <body as below>, "<id2>": {"status": "expired"}}}
```

Every distinct non-empty id is a key; an unknown or expired id reads
`{"status": "expired"}` inside a 200. Empty `ids` or more than 20 distinct
ids answer 400 `bad_request`.

| Status | Body |
|---|---|
| 200 | `{"status": "running"}` |
| 200 | `{"status": "ok", "memory": {...}}` (shape: handoff section 5.2) |
| 200 | `{"status": "rejected", "reason": "sensitive" \| "skip_screenshots" \| "too_few_selected" \| "check_failed", "message": "..."}` |
| 200 | `{"status": "error", "reason": "timeout" \| "model_error", "message": "..."}` |
| 400 | `{"status": "error", "reason": "bad_request", "message": "..."}` on submit: meta not JSON / missing field / a wrong type (weekday 1-7, raw_count >= 1, start/end time strings, string lists for labels and recent_question_types, faces int, favorite/edited bool, represents >= 1, taken_at ISO 8601) / 0 or > 10 photos / filename stems != meta ids / a file > 5 MB or not a JPEG, PNG or HEIC. No job, no model call. |
| 404 | `{"status": "expired"}`: unknown job, or older than 1 h. Resubmit. |

`memory.ending.questions` is 1-3 `{text, type, source}` in the model's
order (`source` `llm`, or `fallback_B1/B2/B3` when none survived, which
then gives all three); the phone picks the one to show (copy refine
handoff 4.3, 5.7). `meta.recent_question_types` (types on the two nights
before and after) is shown to call B as types to avoid; unknown types are
ignored. `memory.generation.copyVersion` is `copy_v2`.

Limits: call A 180 s, call B 60 s, call C 40 s, each retried once (calls B
and C measured on 2026-11-04; see config.py); a retry only gets
what is left of the job, which is capped at 300 s (`timeout`). (User
decision 2026-11-01: the spec's 90 s / 240 s, and then 120 s for call A,
were too tight for 10-photo groups, whose call A with thinking took
80-120 s and sometimes more.) Jobs live in memory for 1 h. Photos are never written
to disk by this service.

## Concurrency and retention

| Variable | Default | Meaning |
|---|---|---|
| `PHOTO_WORKERS` | 40 | jobs running at once (35 groups = 5 phones x 7) |
| `PHOTO_LLM_WORKERS` | 80 | model calls in flight; keep at least 2 x `PHOTO_WORKERS`, or startup logs a warning |
| `PHOTO_JOB_TTL_SECONDS` | 3600 | how long a job can be polled before it reads 404 `expired` |

## Model call modes

Settled by `tools/probe_model.py` against `qwen3.7-flash` on 2026-10-31
(a synthetic image, no personal photo):

| Call | Mode | Request |
|---|---|---|
| A (look) | `thinking_stream` | `enable_thinking=true`, `thinking_budget=1500`, `stream=true` + `include_usage`, no `response_format`; the program parses the JSON |
| B (write) | `json_schema` | `enable_thinking=false`, `response_format` JSON Schema strict (`SCHEMA_B`); images in the request since 2026-11-04 |
| C (check) | `json_object` | `enable_thinking=false`, `response_format` json_object; images in the request |

Both fallbacks (`json_object`) also worked and can be switched to with
`PHOTO_CALL_A_MODE` / `PHOTO_CALL_B_MODE` / `PHOTO_CALL_C_MODE` without a
code change (`json_schema` for C uses `SCHEMA_C`).

Cost uses the token counts the provider returns: input 0.2, output 0.8 CNY
per million tokens (thinking tokens are output).
A call cut at its time limit returns no token counts, so it is logged at
cost 0, but the provider may still bill it (the request is abandoned, not
cancelled upstream). Expect real spend to run slightly above the log when
timeouts happen.

A result that completes after the job cap is not delivered: it is
answered and logged as `timeout` (the log adds `lateResult: true` and the
status it would have had). An unexpected error inside the pipeline still
writes the log line, with the raw output and cost of the calls made, and
answers `error` / `model_error`.

## Readings of the spec

The copy refine handoff (2026-11-04) is followed as written; where it is
silent the code reads it so:

- **Call A's notes** are one `description` per photo (40-80 characters);
  a selected photo without one fails check A. `subjects` stay (templates,
  debug page).
- **Program checks on a detail sentence**: at most 22 characters after
  trimming and dropping trailing full stops; the lexicon
  (`prompts/lexicon_push_body.txt`; a word inside a photo's text, a subject
  name or the category's allowed words passes, except that subject names
  never shield the person-word category, marked `@nosubjects`); no photo id;
  no "lianpai". Nouns and numbers are no longer matched against the notes:
  call C judges them against the photos.
- **Call C** sees the selected photos (the cover marked) and only what the
  program passed. A sentence it calls ungrounded, or a question ungrounded
  or unsafe, is dropped (logged as `c_ungrounded:` / `c_unsafe:` with C's
  reason). Whether a sentence reaches the cover is C's `mentions_cover`:
  the cover id is added to or removed from its `photo_ids` accordingly.
  C failing twice leaves templates, F0 and the fallback questions only, and
  skips the cover rewrite. Templates are never sent to C.
- **Pick**: AI rank 1 +2, rank 2 +1; +1 for two or more photos; -2 when the
  only subjects are everyday things (`prompts/common_objects.txt`) and no
  light or colour word (`prompts/feature_words.txt`) appears. Ties: model
  before template, then shorter.
- **Peak first and the cover rewrite (user decision, 2026-10-31)** are
  unchanged in shape: when no checked model sentence reaches the cover, one
  extra call B for the cover only (no retry), checked by the program and by
  call C on the cover alone. Cover sentences of the first round that were
  cut only for length are shown back to it to shorten (the model tends to
  overshoot 22 characters). Still none: a model sentence about another
  usable photo makes that photo the cover; a template or F0 keeps it.
- **T1 / F0** name the cover's weekday and daypart; **T2** is "from X, to Y"
  with the first and last selected photos' first subjects.
- **Questions**: at most 22 characters, none of the forbidden words, no
  he / she (the word for "other" is allowed), a type from
  `prompts/question_types.txt`, the `people` type only when a selected photo
  has people, no repeated text; then call C. Up to 3 survivors in the
  model's order; none -> B1, B2, B3 (`prompts/question_bank.txt`, types in
  `fallback_types`). The bank's other questions and the old A-skeleton menu
  lists (`action_verbs`, `sight_verbs`) are no longer used.
- **skip=true** from call A is a verdict (`skip_screenshots`, or `sensitive`
  when any photo has a sensitive flag), even with nothing selected.
- **T1's n** sums `represents` over every sent photo showing X, so the count
  is true of the whole group.

## Log

One JSON line per job in `logs/photo-collections/compose.jsonl` (repo root,
git-ignored): raw A/B/C output, every rejected candidate and question with its
reason, call C's verdicts, the final picks, `bodySource` / `questionSource`, per-call latency,
tokens and cost, plus `bucket` and `tz` from meta. No image bytes, no key.

Lines older than 30 days are purged at startup and then every 24 h
(`logclean.py`; the file is rewritten atomically under the same lock the
writer uses). A line with no readable `ts` is never deleted.
`tools/log_stats.py [path]` prints counts by status and reason, mean latency
and total cost, skipping test entries whose `group_key` starts with `cc_`.

## Tests

```bash
cd poc/photo-collections
PYTHONPATH=.:../.. .venv/bin/python -m pytest tests -v
```

Offline: every model call is the `fake_model` fixture in `tests/conftest.py`.

## Tools (real network, not tests)

- `tools/log_stats.py` -- reads the compose log (no network).
- `tools/probe_model.py` -- which request shapes the model accepts.
- `tools/run_h1.py` -- acceptance H1: runs the prototype's 80 photos through
  the running service and writes a summary table outside the repo.

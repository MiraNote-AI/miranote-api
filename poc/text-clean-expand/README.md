# Text Clean & Expand POC

FastAPI service that transforms a user's free-form text using an
OpenAI-compatible LLM. Eight endpoints, each with its own prompt:

| Endpoint | Purpose |
|---|---|
| `POST /clean` | Fix typos, grammar and punctuation, and structure long text (bullet points, line breaks). Adds nothing, and leaves correct sentences as written |
| `POST /expand` | Finish the note: the note exactly as written, then one or two closing sentences |
| `POST /polish` | Final editing pass -- word choice + flow, no restructuring |
| `POST /rewrite` | Rewrite one block following the user's own instruction ("make it warmer") |
| `POST /shorten` | Produce a shorter version. `target`: 30% / 50% / tweet |
| `POST /keywords` | Extract 5-10 keywords with salience scores (1-10) |
| `POST /caption` | 1-2 sentence caption. `style`: instagram / diary / tweet |
| `POST /mood` | Page mood for the star jar: `{"mood": "joyful"/"calm"/"tender"/"low"/"tense"/"neutral"}` |

Bilingual: every text endpoint answers in the text's main language and keeps words from the other language (names, places, terms) as written. A reply in the wrong language gets one rewrite only when the text is clearly Chinese or clearly English (at least twice as many Han characters as English words, or the reverse); a mixed note is left to the model. `/keywords` keeps each term in the language it appears in; `/mood` always returns an English label.

## `POST /expand`

Expand finishes the note; it does not rewrite it (#132).

- Request: `{"text": str, "context"?: str}`. Response:
  `{"original": str, "expanded": str}`, where `expanded` is `text`
  unchanged, byte for byte, followed by the addition.
- The model is asked for the addition only: one or two sentences that
  close the note, in the writer's first-person voice, with no facts the
  note does not state. The server builds `expanded` itself: a reply that
  repeats the note has it taken off the front (whitespace aside), and the
  addition is cut to its last two sentences (a sentence ends at `.` `!`
  `?` before a space or the end, or at the full-width ones). The
  grounding check reads the addition; invented facts get one rewrite, and
  a failed rewrite is a 502 (not charged).
- Joining: one space after an English note, nothing after a Chinese one;
  a note with no main language goes by its last character. A note that
  ends in whitespace (a line break) keeps it, with nothing added, and an
  addition that starts with a comma or a full stop follows directly.
- No usable addition (an empty reply, or the note repeated and nothing
  more): a 502 `{"detail": "LLM reply added nothing to the note"}`,
  settled failed (not charged) like a failed grounded rewrite, and the
  service logs `Expand: the model gave no addition`. A retry usually
  brings one.

## `POST /rewrite`

The editor's free-form request ("make it warmer", "more formal", "add an
ending") applied to the block being written.

- Request: `{"text": str, "instruction": str, "context"?: str}`. `text`
  is the block, 1-3000 characters; `instruction` is the user's words,
  trimmed, 1-500 characters; `context` is optional nearby note text, at
  most 8000 characters, used for reference only and never rewritten.
  The block's cap is the lower one because the rewrite of a whole block
  has to fit in the reply's 2048 tokens (about 3400 Chinese characters
  on deepseek-chat), and a longer reply would outlast Cloudflare's 100 s
  timeout in front of the beta. The caps count Unicode code points (Python's `len`), not bytes: a
  Chinese character is one, an emoji with a skin tone or a variation
  selector (U+FE0F) is two or more, though Swift's `String.count` calls
  it one. A client cutting to the cap counts `unicodeScalars`.
- Response: `{"original": str, "rewritten": str}`, in the block's
  language whatever the instruction's language.
- Errors: 422 for an empty or missing `instruction` / `text` or a field
  over its cap; 502 (not charged) when the model call fails or its reply
  comes back empty or cut off at max_tokens. Auth and metering are the
  gate's (`ai_auth.py`): 401 `missing_token` / `invalid_token` /
  `token_expired` / `unknown_user`; 429 `quota` with `feature: "expand"`
  (the cap /expand and /polish share; the paywall), 429 `rate_limited`
  (per-user per-minute limit; retry, not the paywall); 409
  `duplicate_request`; 503 `accounts_unavailable` / `service_paused`.
- The instruction and context reach the model as tagged data in the user
  turn; the prompt (`prompts/rewrite.txt`) refuses to take instructions
  from them beyond the edit itself.

## Setup

```bash
cd poc/text-clean-expand
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # fill in LLM_API_KEY
```

## Run

```bash
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8001 --reload
```

UI at <http://localhost:8001/>. Or use `./start-all.sh` at the repo root to bring up all four POCs.

## Curl examples

```bash
curl -s -X POST http://localhost:8001/polish \
  -H 'Content-Type: application/json' \
  -d '{"text":"morning light warm. coffee strong. happy."}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/rewrite \
  -H 'Content-Type: application/json' \
  -d '{"text":"morning light. coffee.","instruction":"make it warmer"}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/shorten \
  -H 'Content-Type: application/json' \
  -d '{"text":"long text here","target":"tweet"}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/keywords \
  -H 'Content-Type: application/json' \
  -d '{"text":"We shipped voice transcription beta to 10 partners.","max":5}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/caption \
  -H 'Content-Type: application/json' \
  -d '{"text":"Long journal entry...","style":"diary"}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/mood \
  -H 'Content-Type: application/json' \
  -d '{"text":"Picnic with Mia in the park. Photo: two kids laughing."}' | python3 -m json.tool
```

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```

## Quote action (NEW, depends on retrieval server)

The 7th Text-tab action -- `Quote` -- does NOT call this server.
Instead it calls the retrieval POC at `http://localhost:8004/quotes`
(see `poc/retrieval/`). Make sure that server is running, or use
`./start-all.sh` from the repo root.

Sub-controls: `Lang` (auto / en / zh / both) and `Max` (1-5).
Result is rendered as quote cards with author, source, match %, and a
one-sentence "why" line. Zero matches is a valid response.

# Text Clean & Expand POC

FastAPI service that transforms a user's free-form text using an
OpenAI-compatible LLM. Six endpoints, each with its own prompt:

| Endpoint | Purpose |
|---|---|
| `POST /clean` | Restructure messy input into a readable note |
| `POST /expand` | Develop the user's input as if they wrote a longer version |
| `POST /polish` | Final editing pass -- word choice + flow, no restructuring |
| `POST /shorten` | Produce a shorter version. `target`: 30% / 50% / tweet |
| `POST /keywords` | Extract 5-10 keywords with salience scores (1-10) |
| `POST /caption` | 1-2 sentence caption. `style`: instagram / diary / tweet |

Bilingual: every endpoint preserves the input language (English in -> English out, any Chinese in -> Chinese out).

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

curl -s -X POST http://localhost:8001/shorten \
  -H 'Content-Type: application/json' \
  -d '{"text":"long text here","target":"tweet"}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/keywords \
  -H 'Content-Type: application/json' \
  -d '{"text":"We shipped voice transcription beta to 10 partners.","max":5}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/caption \
  -H 'Content-Type: application/json' \
  -d '{"text":"Long journal entry...","style":"diary"}' | python3 -m json.tool
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

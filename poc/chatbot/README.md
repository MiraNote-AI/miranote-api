# Chatbot POC

A FastAPI demo of multi-turn chat with native OpenAI-style function
calling. The agent answers questions about the markdown documents in
`DOCS_ROOT` using three read-only tools: `list_docs`, `read_doc`,
`search_docs`.

Design spec: `docs/specs/2026-05-28-chatbot-with-tools-design.md`.

## Setup

```bash
cd poc/chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: put your LLM_API_KEY in, pick a provider
```

DeepSeek is the default. Any OpenAI-compatible provider that supports
tool calling works (Gemini's OpenAI shim, OpenAI proper, Moonshot).

## Run

```bash
cd poc/chatbot
PYTHONPATH=../.. uvicorn main:app --reload --port 8002
```

Then open <http://localhost:8002/>.

## Try it (curl)

```bash
# First turn (no session_id -> server mints one and returns it)
curl -s -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"what files are available?"}' | python3 -m json.tool

# Follow-up using the returned session_id
curl -s -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<paste-here>","message":"summarize the Q3 plans"}' | python3 -m json.tool
```

## Three demo questions

Try these in the web UI against the bundled `demo_data/docs/`:

1. **English / `list_docs`:** _"What docs do we have?"_
2. **English / `read_doc`:** _"What ships in Q3 2026?"_
3. **中文 / `search_docs`:** _"团队里谁负责 iOS?"_

You should see tool chips under each assistant reply showing exactly
which tools were called and what came back.

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v
```

## Tools available to the model

| Name | Purpose | Caps |
|---|---|---|
| `list_docs(subdir)` | List files under a subdir of `DOCS_ROOT`. | 200 files |
| `read_doc(path)` | Read a UTF-8 file. | 32 KB truncated |
| `search_docs(query, max_hits)` | Case-insensitive substring search. | 200 files, 160-char snippet |

All tools resolve paths under `DOCS_ROOT`; anything escaping is rejected
with `{"error": "...outside DOCS_ROOT"}`.

## Configuration

See `.env.example`. The important knobs:

- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` -- provider switch.
- `DOCS_ROOT` -- the directory the agent can read from.
- `MAX_TOOL_ITERATIONS` -- safety cap on the tool-call loop.
- `SESSION_TTL_SECONDS` -- idle eviction for in-memory sessions.

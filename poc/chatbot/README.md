# Chatbot POC (backend)

A FastAPI demo of multi-turn chat with native OpenAI-style function
calling. The agent answers questions about the documents in `DOCS_ROOT`
using four tools: `list_docs`, `read_doc`, `search_docs`, and
`set_docs_root` (lets the agent or the user switch the docs directory
at runtime).

`read_doc` is polymorphic by file extension: plain text / markdown, PDF
(.pdf), Word (.docx), and images (OCR via tesseract).

**This POC ships no UI of its own.** The shared web UI lives in
`poc/text-clean-expand/static/index.html` and talks to this server's
`/chat`, `/health`, `/sessions/*`, and `/config` endpoints cross-origin.

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

### Optional: image OCR

To let `read_doc` extract text from images you also need the tesseract
binary on the system:

```bash
brew install tesseract tesseract-lang   # macOS
```

Without it, `read_doc` on a `.png/.jpg/...` returns a friendly error
pointing back here. All other formats work out of the box.

## Run

```bash
# from the repo root
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m uvicorn poc.chatbot.main:app \
    --port 8003 --reload
```

Then start the unified UI (separate terminal):

```bash
cd poc/text-clean-expand
PYTHONPATH=../.. uvicorn main:app --reload --port 8001
```

Open <http://localhost:8001/> and click the **Chat** tab.

## Try it (curl)

```bash
# First turn -- server mints a session_id
curl -s -X POST http://localhost:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"what files are available?"}' | python3 -m json.tool

# Follow-up with the returned session_id
curl -s -X POST http://localhost:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<paste>","message":"summarize the Q3 plans"}' | python3 -m json.tool

# Switch docs root at runtime
curl -s -X POST http://localhost:8003/config \
  -H 'Content-Type: application/json' \
  -d '{"docs_root":"/some/other/dir"}' | python3 -m json.tool
```

## Demo questions

Try these in the Chat tab of the unified UI against the bundled
`demo_data/docs/` (markdown ADRs + a PDF + a DOCX + a PNG):

1. **English / markdown / `list_docs`:** _"What docs do we have?"_
2. **English / markdown / `read_doc`:** _"What ships in Q3 2026?"_
3. **中文 / markdown / `search_docs`:** _"团队里谁负责 iOS?"_
4. **English / PDF / `read_doc`:** _"What are the four open questions
   in the architecture snapshot PDF?"_
5. **中文 / DOCX / `read_doc`:** _"在 Q2 回顾会上 mengjia 接的 action
   item 有哪些？"_
6. **English / image / `read_doc`:** _"What's still un-checked on the
   whiteboard TODO?"_  (requires tesseract -- see Setup)
7. **Switch docs root:** type a new path into the "Docs root" row at
   the top of the Chat tab and click Apply, OR ask the agent _"switch
   docs to /tmp/somefolder"_.

Tool chips under each assistant reply show which tools were called and
what came back.

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v
```

## Tools available to the model

| Name | Purpose | Caps |
|---|---|---|
| `list_docs(subdir)` | List files under a subdir of `DOCS_ROOT`. | 200 files |
| `read_doc(path)` | Read a file. Dispatches on extension: text/markdown, PDF, DOCX, image (OCR). | 32 KB truncated |
| `search_docs(query, max_hits)` | Case-insensitive substring search across **UTF-8 text files only** -- PDFs/DOCX/images are invisible to it; use `read_doc` for those. | 200 files, 160-char snippet |
| `set_docs_root(path)` | Switch the docs directory. Only called when the user explicitly asks. Validates that the new path exists and is a directory. | -- |

All `read_doc`/`list_docs`/`search_docs` calls resolve paths under
`DOCS_ROOT`; anything escaping is rejected with
`{"error": "...outside DOCS_ROOT"}`.

## HTTP API

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/chat` | `{session_id?, message}` | Main chat endpoint. |
| GET  | `/sessions/{sid}` | -- | Dump full history (debug). |
| DELETE | `/sessions/{sid}` | -- | Clear a session. |
| GET  | `/health` | -- | `{status, model, tools, docs_root}`. |
| POST | `/config` | `{docs_root}` | Validate + replace `docs_root` at runtime. Returns the resolved absolute path. |

## Configuration

See `.env.example`. The important knobs:

- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` -- provider switch.
- `DOCS_ROOT` -- initial directory. Can be changed at runtime via
  `POST /config` (UI Docs root row) or the `set_docs_root` tool.
- `MAX_TOOL_ITERATIONS` -- safety cap on the tool-call loop.
- `SESSION_TTL_SECONDS` -- idle eviction for in-memory sessions.

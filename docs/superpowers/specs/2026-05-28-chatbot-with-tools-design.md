# Chatbot with tool calling — POC design

- **Date:** 2026-05-28
- **Author:** mengjia (Claude-assisted)
- **Status:** Draft, awaiting implementation plan
- **Scope:** `miranote-api/poc/chatbot/` (new POC, sibling of `voice-to-text/` and `text-clean-expand/`)
- **Reference:** Extends the POC pattern established in `poc/text-clean-expand/main.py`
  and `poc/voice-to-text/main.py` (FastAPI + OpenAI-compatible client + `prompts/`
  + `static/index.html`).

## 1. Goal

Add a chatbot POC that demonstrates two capabilities the existing POCs do not:

1. **Multi-turn conversation** — the model sees prior turns and can be asked
   follow-ups.
2. **Native function calling** — the model decides on its own when to invoke
   tools, and the server executes them and returns the result for the model
   to use in its final answer.

The starter tool kit is intentionally small: file-system tools scoped to a
configurable docs directory, so the canonical demo is _"ask the agent
questions about the markdown docs in `./demo_data/docs/`"_. The POC is the
foundation; future tools (web search, calculators, MiraNote-internal
read APIs, etc.) plug in through the same dispatcher.

## 2. Non-goals (v1)

- Auth or multi-user separation.
- Persistent sessions across restarts (Redis / SQLite).
- Streaming responses.
- Vector search / RAG. v1 uses substring search, which is fine at
  `demo_data/` scale.
- Write / exec / network tools. v1 ships read-only fs tools only.
- Production hardening — concurrency limits, rate limits, retries beyond
  the existing 502-on-error pattern in `text-clean-expand`.

## 3. File layout

```
miranote-api/poc/chatbot/
    main.py                # FastAPI app, routing only
    chat_loop.py           # Tool-calling loop (pure logic, easily testable)
    tools.py               # Tool JSON-schema registry + dispatcher
    tools_fs.py            # list_docs / read_doc / search_docs implementations
    session.py             # In-memory session store with TTL eviction
    prompts/
        system.txt         # System prompt (English; names available tools)
    static/
        index.html         # Single-page chat UI
    demo_data/
        docs/              # Sample markdown docs the chatbot answers about
            product_overview.md
            roadmap_2026.md
            team.md
    .env.example
    requirements.txt
    README.md
```

Pattern alignment notes:
- `prompts/system.txt` is loaded at startup (Rule 3 — no CJK in source files;
  prompts are allowlisted under `**/prompts/*.txt`).
- `static/` and `demo_data/` are both allowlisted, so bilingual content is OK
  there.
- `README.md` is allowlisted under `poc/*/README.md`, so it can include
  bilingual usage examples.

## 4. Tools (v1, read-only)

All tools are read-only and operate against `DOCS_ROOT` (env var, default
`./demo_data/docs`). Every path argument is resolved with `Path.resolve()`
and rejected unless `.is_relative_to(DOCS_ROOT.resolve())`. Errors are
returned as `{"error": "<message>"}` so the model can recover.

| Name | Inputs | Output | Caps |
|---|---|---|---|
| `list_docs` | `subdir: str` (optional, default ".") | `[{"path": "...", "size_bytes": N}]` | scans at most 200 files |
| `read_doc` | `path: str` | `{"path": "...", "content": "..."}` | content truncated to 32 KB UTF-8 |
| `search_docs` | `query: str`, `max_hits: int = 20` | `[{"path": "...", "line": N, "snippet": "..."}]` | scans at most 200 files; snippet 160 chars; case-insensitive substring |

The tool registry in `tools.py` exposes:

```python
TOOLS: list[dict]              # OpenAI tool-schema list passed to the model
def dispatch(name: str, args: dict) -> dict   # routes to tools_fs.* and returns JSON-serializable result
```

This indirection means adding a non-fs tool later is a one-file change.

## 5. Chat loop

`chat_loop.run_turn(session_id, user_message) -> ChatTurnResult`:

1. Load session history. If the `session_id` is missing or unknown, mint a
   new uuid4 and seed the history with a single
   `{"role": "system", "content": <prompts/system.txt>}` message.
2. Append `{"role": "user", "content": user_message}`.
3. Loop, up to `MAX_TOOL_ITERATIONS` (default 6):
   1. `resp = client.chat.completions.create(model=MODEL, messages=history, tools=TOOLS, tool_choice="auto")`
   2. `msg = resp.choices[0].message`
   3. If `msg.tool_calls`:
      - Append the assistant message (including `tool_calls`) to history.
      - For each `call`: `result = tools.dispatch(call.function.name, json.loads(call.function.arguments))`.
        Wrap exceptions as `{"error": str(e)}`.
        Append `{"role": "tool", "tool_call_id": call.id, "name": ..., "content": json.dumps(result)}` to history.
      - Continue loop.
   4. Else: append the assistant text, break.
4. If the loop runs to completion without the model returning a
   text-only message (i.e. the cap was hit while still mid-tool-loop),
   append and return a synthetic assistant message:
   `"(stopped: hit MAX_TOOL_ITERATIONS — partial tool use, no final answer)"`.
   The cap-hit branch is the `reply` value; it is never empty.
5. Trim session history if it now exceeds `MAX_HISTORY_MESSAGES` (default 40,
   excluding the system prompt) by dropping oldest non-system messages in
   pairs.
6. Return `ChatTurnResult(session_id, reply, tool_trace)` where `tool_trace`
   is a list of `{name, args, result_preview}` for the UI to render.

Failure modes: any exception from `client.chat.completions.create`
re-raises as `HTTPException(502, detail=...)`, matching the pattern in
`text-clean-expand/main.py`.

## 6. HTTP API

| Method | Path | Body | Response |
|---|---|---|---|
| `POST` | `/chat` | `{session_id?: str, message: str}` | `{session_id, reply, tool_trace[]}` |
| `GET` | `/sessions/{id}` | — | `{messages: [...]}` (debug; full history) |
| `DELETE` | `/sessions/{id}` | — | `{status: "deleted"}` |
| `GET` | `/health` | — | `{status, model, tools: [names], docs_root}` |
| `GET` | `/` | — | serves `static/index.html` |
| `GET` | `/static/*` | — | static mount |

`POST /chat` validation: `message` is non-empty, length <= 4000 chars
(Pydantic). `session_id`, if provided, must exist; otherwise a fresh one is
created and returned.

## 7. UI (`static/index.html`)

Single page, vanilla HTML / CSS / JS — same warm/cream palette and
`Inter`-fallback font stack as the existing two POCs to keep the family
look. Layout:

- Header: title + subtitle.
- Main column (max-width 920px):
  - Message list. User bubbles right-aligned (warm tint), assistant
    left-aligned (panel white).
  - Under each assistant turn: collapsible chips for tool calls
    (`tool: search_docs`, click to expand args + truncated result).
  - Composer: textarea + Send button (Cmd/Ctrl+Enter to send), Reset
    button (DELETEs the session, starts a new one).
- Footer: tiny status line (model name + docs root, from `/health`).

No build step. State stored client-side in a single `messages` array and a
`sessionId` string; both wiped by Reset.

## 8. Environment

```
LLM_API_KEY=...                                  # required
LLM_BASE_URL=https://api.deepseek.com            # optional; omit for OpenAI
LLM_MODEL=deepseek-chat                          # default if unset
DOCS_ROOT=./demo_data/docs                       # default
MAX_TOOL_ITERATIONS=6
MAX_HISTORY_MESSAGES=40
SESSION_TTL_SECONDS=3600
```

Tool-calling compatibility: DeepSeek (`deepseek-chat`), Gemini's OpenAI
shim, and OpenAI `gpt-4o` all implement the OpenAI tools schema. Moonshot
also supports it. Provider switching remains a `.env` edit, no code change.

## 9. Safety

- **Path traversal:** every fs tool resolves the requested path and
  rejects anything not under `DOCS_ROOT.resolve()`. Reject early with
  `{"error": "path outside DOCS_ROOT"}`.
- **Read-only:** no write / exec / network tools in v1. Reviewers should
  treat the addition of a write tool as a separate spec.
- **Iteration cap:** prevents a malformed model from looping forever on
  tool calls.
- **Read size cap:** 32 KB per `read_doc` to prevent context blowup on a
  large file. Truncation is signalled in the returned payload
  (`"truncated": true`).
- **History cap:** prevents unbounded token growth on long sessions.
- **No secrets in tool output:** docs under `DOCS_ROOT` are
  user-curated; the server does not read env, git config, etc.

## 10. Testing strategy

- Unit tests for `tools_fs.py` — path-traversal rejection, size cap,
  search hit ordering. No network, no LLM.
- Unit tests for `chat_loop.py` using a fake `client` that returns
  scripted `tool_calls` then a final answer — verifies the loop handles
  multi-step tool use, error wrapping, and iteration cap.
- Manual demo via `static/index.html` against `demo_data/docs/` with
  three canonical questions documented in the README.

## 11. Demo content

Three small markdown files under `demo_data/docs/`, each ~30 lines:

- `product_overview.md` — what MiraNote is (one-paragraph pitch + bullet
  features).
- `roadmap_2026.md` — quarterly milestones, written so questions like
  _"what ships in Q3?"_ have a clear answer.
- `team.md` — a couple of roles + responsibilities, so _"who owns the
  iOS app?"_ has a clear answer.

Content is bilingual (English + Chinese) to exercise the same
multi-language path as the other POCs. Files live under `demo_data/`
which is allowlisted by Rule 3.

## 12. Open follow-ups (post-v1)

- Streaming chat (SSE) — UX win, modest server complexity.
- Vector-index tool — if `search_docs` substring becomes the bottleneck.
- Tool: `fetch_url(url)` for live web context (introduces SSRF and rate
  considerations — own spec).
- Persistent sessions on SQLite if anyone wants to come back to a
  conversation after restart.
- Auth (header token) once we point this at anything beyond `demo_data/`.

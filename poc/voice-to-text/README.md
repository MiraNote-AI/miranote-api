# Voice-to-Text POC

FastAPI service that transcribes audio with [Whisper](https://github.com/openai/whisper)
(local) and optionally post-corrects the output with an LLM (any
OpenAI-compatible provider).

This is a POC. No auth, no upload size limit, no SLAs. Run locally.

## Quick start

```bash
cd poc/voice-to-text
pip install -r requirements.txt
cp .env.example .env             # edit -- see "LLM configuration" below
uvicorn main:app --host 0.0.0.0 --port 8000
```

First run downloads the Whisper model (~1.5 GB for `medium`).

## LLM configuration

Post-correction is **optional**. Without an LLM key, `/transcribe`
returns raw Whisper output -- no punctuation, possible homophone errors,
no English-typo fixes. With an LLM key, you also get a `corrected_text`
field that has been cleaned up against
[`prompts/correction.txt`](prompts/correction.txt).

Three env vars control the LLM:

| Variable        | Required | Default                                                  | Notes |
|-----------------|----------|----------------------------------------------------------|-------|
| `LLM_API_KEY`   | optional | empty (disables correction)                              | empty -> raw Whisper output only |
| `LLM_BASE_URL`  | optional | `https://generativelanguage.googleapis.com/v1beta/openai` | any OpenAI-compatible endpoint |
| `LLM_MODEL`     | optional | `gemini-2.5-flash`                                       | must be served by the configured base URL |

### Provider examples

Any provider that speaks the OpenAI chat-completions protocol works.
Copy-paste one of the following into your `.env`:

**Gemini (default)**
```bash
LLM_API_KEY=AIza...
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_MODEL=gemini-2.5-flash
```

**DeepSeek**
```bash
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

**Moonshot**
```bash
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=moonshot-v1-8k
```

**OpenAI**
```bash
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

**Local (vLLM, Ollama via OpenAI shim, etc.)**
```bash
LLM_API_KEY=dummy
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## Whisper configuration

| Variable          | Default  | Notes |
|-------------------|----------|-------|
| `WHISPER_MODEL`   | `medium` | `tiny` / `base` / `small` / `medium` / `large` -- larger = slower + more accurate + bigger download |

## Endpoints

**`POST /transcribe`** -- multipart audio upload

| Query param | Default | Notes |
|-------------|---------|-------|
| `correct`   | `true`  | if `false`, skip LLM correction even when `LLM_API_KEY` is set |

Response:

```json
{
  "language": "zh",
  "raw_text": "...",
  "corrected_text": "...",
  "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]
}
```

`corrected_text` is `null` only when `correct=false`. When `LLM_API_KEY`
is unset, it falls back to `raw_text` (same content, easier client code).

**`GET /health`**

```json
{"status": "ok", "whisper_model": "medium", "llm_model": "deepseek-chat"}
```

`llm_model` is `null` when no LLM key is configured.

## Known POC limitations

- No request auth, no rate limit, no upload size cap -- do not expose to
  the public internet
- No tests
- Dependencies pinned with lower bounds only
- Single-process; not benchmarked under load
- LLM retry is 3 attempts with linear 45/90s backoff on HTTP 429; other
  errors fail fast and return raw Whisper output

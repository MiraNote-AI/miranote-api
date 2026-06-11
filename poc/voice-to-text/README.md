# Voice-to-Text POC

FastAPI service that transcribes audio with [Whisper](https://github.com/openai/whisper)
(local) and optionally post-corrects the output with an LLM (any
OpenAI-compatible provider).

This is a POC. No auth, no upload size limit, no SLAs. Run locally.

> New here? Start with [SETUP.md](SETUP.md) -- a linear first-time
> setup guide. This README is the reference (env vars, response
> schema, provider matrix).

## Quick start

```bash
cd poc/voice-to-text
pip install -r requirements.txt
cp .env.example .env             # edit -- see "LLM configuration" below
uvicorn main:app --host 0.0.0.0 --port 8000
```

First run downloads the Whisper model (~1.5 GB for `medium`).

## Try it without recording anything

`demo_data/` ships three short TTS clips (Chinese, Chinese + English
mixed, English) so you can hit `/transcribe` immediately. See
[`demo_data/README.md`](demo_data/README.md) for what each file
exercises and copy-paste curl commands.

## Web UI

Once the server is running, open <http://localhost:8000/> in a browser.
A single page is served from `static/index.html` with two modes:

- **Upload file** -- pick any audio file from disk and transcribe it.
- **Record (mic)** -- record straight from the browser microphone using
  `MediaRecorder`, then transcribe the resulting clip. Requires
  microphone permission on first click. Works on `localhost` without
  HTTPS; for any other origin the browser will refuse mic access without
  a secure context.

Both modes show `raw_text`, `corrected_text`, the `correction_status`
badge, and the Whisper segments. There's a checkbox to toggle the
`correct` query param per request.

The UI is plain HTML + vanilla JS (no build step, no CDN dependencies)
so it works offline once the page is loaded.

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
LLM_MODEL=deepseek-v4-flash
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
| `lang`      | `zh`    | one of `zh` or `en`. Forces Whisper's language; auto-detect is disabled because on short / noisy clips it misfires (we observed Mandarin classified as Javanese). `zh` is the right choice for Chinese audio **and** for Chinese + English code-switching -- the multilingual model handles inline English. `en` is for pure English audio. Other values return 422. |

Response:

```json
{
  "language": "zh",
  "raw_text": "...",
  "corrected_text": "...",
  "correction_status": "ok",
  "segments": [{"start": 0.0, "end": 4.2, "text": "..."}]
}
```

`correction_status` is one of:

| Value     | `corrected_text` | When |
|-----------|------------------|------|
| `ok`      | LLM-corrected string | LLM call succeeded |
| `skipped` | `null`               | `correct=false`, or no `LLM_API_KEY`, or `raw_text` is empty |
| `failed`  | `null`               | LLM call errored after retries (rate-limit exhausted, quota, network, ...). Server stderr logs the underlying error. |

Clients that just want "best available text" should read `corrected_text
or raw_text`. Clients that care whether the LLM ran should check
`correction_status`.

**`GET /health`**

```json
{"status": "ok", "whisper_model": "medium", "llm_model": "deepseek-v4-flash"}
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

## Acoustic emotion analysis

`/transcribe?with_emotion=true` (default true) and the new `/emotion`
endpoint run a HuggingFace audio-classification model on the uploaded
audio and return a 7-class emotion label (angry, disgust, fear, happy,
neutral, sad, surprise) plus per-class scores.

**Model:** `hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0`
- ~1.3 GB, auto-downloads to `~/.cache/huggingface/` on first emotion request (one-time, ~30 sec on first call)
- XLSR-53 multilingual backbone; trained on English emotion corpora but the author reports cross-lingual generalisation to Chinese and French (empirical claim, not benchmarked)

**Override the model** via env var `EMOTION_MODEL=<repo-id>`.

**Curl:**

```bash
# Bundled with transcription
curl -s -X POST "http://localhost:8000/transcribe?correct=true&with_emotion=true&lang=en" \
  -F file=@demo_data/en_short.m4a | python3 -m json.tool

# Standalone
curl -s -X POST http://localhost:8000/emotion -F file=@demo_data/en_short.m4a | python3 -m json.tool
```

**Response shape (transcribe with emotion):**

```json
{
  "language": "en", "raw_text": "...", "corrected_text": "...",
  "correction_status": "ok",
  "segments": [...],
  "emotion": {
    "label": "happy", "confidence": 0.83,
    "all_scores": [{"label": "happy", "score": 0.83}, ...]
  },
  "emotion_status": "ok"
}
```

`emotion` is `null` and `emotion_status` is `"failed"` if classification raised; `null` and `"skipped"` if `with_emotion=false`.

## Tests

```bash
cd poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```

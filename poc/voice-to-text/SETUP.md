# Local setup -- voice-to-text POC

First-time setup for a new teammate cloning `miranote-api` and running
the voice-to-text POC locally. For reference details (full LLM
provider matrix, response schema, every env var), see
[README.md](README.md). This file is the linear "do these steps in
order" version.

## What is in this repo right now

`MiraNote-AI` ships as five sibling repos. As of today only one thing
is runnable end-to-end: the **voice-to-text POC** at
`poc/voice-to-text/`. The other repos (`miranote-web`, `miranote-ios`,
`mirabot`, `.github`) are still mostly governance plus skeleton.

So "test the repo locally" == "run the POC".

## Prerequisites

- Python 3.10 or newer
- `ffmpeg` on `PATH` -- Whisper shells out to it for audio decoding

macOS:
```bash
brew install python@3.11 ffmpeg
```

Ubuntu / Debian:
```bash
sudo apt-get install python3-venv python3-pip ffmpeg
```

Verify:
```bash
python3 --version
ffmpeg -version | head -n 1
```

## Step 1: clone and enter the POC

```bash
git clone git@github.com:MiraNote-AI/miranote-api.git
cd miranote-api/poc/voice-to-text
```

## Step 2: virtualenv and dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`openai-whisper` pulls in PyTorch, so expect a multi-hundred-MB
download on the first install.

## Step 3: configure `.env`

```bash
cp .env.example .env
```

Two common cases:

- **Whisper only (raw transcripts, no LLM cleanup).** Leave
  `LLM_API_KEY` empty. `/transcribe` returns `raw_text`,
  `corrected_text: null`, and `correction_status: "skipped"`.
- **Whisper plus LLM post-correction.** Put an OpenAI-compatible API
  key into `LLM_API_KEY`. Default base URL points at Gemini; switch
  `LLM_BASE_URL` and `LLM_MODEL` together if you use a different
  provider. README.md has copy-pasteable blocks for Gemini, DeepSeek,
  Moonshot, OpenAI, and local vLLM / Ollama.

Optional: set `WHISPER_MODEL=small` (or `base`) for a faster, smaller
download while you are just sanity-checking. The default `medium` is
~1.5GB on first run.

## Step 4: start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The first request triggers the Whisper model download if it has not
happened yet, so the first `/transcribe` call will look slow. After
that the model is cached under `~/.cache/whisper/` and startup is
fast.

## Step 5: smoke test

Browser:
- Open <http://localhost:8000/>.
- Try the **Record (mic)** tab. The browser asks for mic permission
  on first click.
- Try the **Upload file** tab with any short audio clip.

Or via curl:
```bash
curl http://localhost:8000/health
curl -F "file=@some.m4a" "http://localhost:8000/transcribe?lang=zh"
```

`/health` should return immediately. The first `/transcribe` may take
tens of seconds because of model load.

## Troubleshooting

- **`FileNotFoundError: ... ffmpeg`** -- ffmpeg is not on `PATH`.
  Re-run the prereqs step.
- **Mic button does nothing in browser** -- the page must be loaded
  over `localhost` or HTTPS. `http://<lan-ip>:8000` is rejected by the
  browser as an insecure origin for `MediaRecorder`.
- **`/transcribe` returns `correction_status: "failed"`** -- the LLM
  call errored. Server stderr has the underlying message; most often
  this is a bad or rate-limited key. `raw_text` is still usable.
- **`/transcribe` returns 422 on `lang`** -- only `zh` and `en` are
  accepted. Use `zh` for Mandarin or Mandarin plus English mixed;
  `en` for pure English.
- **Port 8000 already in use** -- pass `--port 8001` (or any free
  port) to `uvicorn`.

## Governance checks (only if you edit `MiraNote-AI/.github`)

If your work touches the rule registry, run these locally before
opening a PR:

```bash
PYTHONPATH=. python3 -m checks.contributing_format . --mode source
PYTHONPATH=. python3 -m checks._meta.all_rules_have_checks .
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji .
PYTHONPATH=. python3 -m checks.claude_md_size . --max 80
PYTHONPATH=. python3 -m checks.skills_registry .
PYTHONPATH=. python3 -m unittest discover checks/tests -v
```

For any other repo, CI runs these on your PR -- you do not need to
run them by hand.

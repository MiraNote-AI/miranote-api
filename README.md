# miranote-api

Backend POCs for MiraNote. Five small FastAPI services, one per
capability; the iOS app talks to all of them on localhost.

| Service | Port | Needs |
|---|---|---|
| text-clean-expand (polish / expand / captions) | 8001 | `.env` with `LLM_API_KEY` |
| voice-to-text (transcription) | 8005 | `.env` with `LLM_API_KEY` |
| image-generation (generate / cutout / stylize / describe) | 8002 | `.env` with GCP `PROJECT_ID` + gcloud ADC |
| chatbot (journal chat, drafts, titles) | 8003 | `.env` with `LLM_API_KEY` |
| retrieval (quote corpus) | 8004 | `.env` with `LLM_API_KEY` |

## Quick start

Each POC keeps its own virtualenv and `.env`:

```bash
cd poc/<name>
python3 -m venv .venv                       # image-generation: use python3.13
.venv/bin/pip install -r requirements.txt
cp .env.example .env                        # then fill in the values below
```

Then run everything at once from the repo root:

```bash
bash start-all.sh    # Ctrl-C stops all; skips any POC missing .venv or .env
```

`start-all.sh` looks for `.venv` (dot prefix). Per-POC READMEs cover
endpoints and options.

## LLM API key (ports 8001 / 8003 / 8004 / 8005)

These four call an OpenAI-compatible chat API. In each `.env`:

```
LLM_API_KEY=sk-...        # ask the team for the shared DeepSeek key
LLM_BASE_URL=...          # only if not using the default provider
```

The chatbot defaults to `deepseek-v4-flash`; a thinking-mode model is
fine -- the server retries empty completions and returns 502 rather
than a blank reply.

## Image service (port 8002)

Two kinds of dependencies:

1. **Vertex AI (cloud)** -- stylize, border, describe, and the bbox
   detector inside a prompted cutout run on Gemini models in your GCP
   project:

   ```bash
   gcloud auth application-default login   # once per machine
   # .env: PROJECT_ID=<your-gcp-project>, LOCATION=global
   ```

   `LOCATION` must be `global`: the Gemini image id in `config.py` is
   served only from that endpoint and a regional value makes Vertex
   answer 404 for it.

2. **Aliyun Bailian / DashScope (cloud)** -- `/generate` only. The image
   model is `z-image-turbo` and the prompt expander is `qwen-turbo`.
   Needs `DASHSCOPE_API_KEY`, which is the **same key** the
   voice-to-text service uses for transcript correction.

   There is no fallback to a second image model: out of credit answers
   503, a content block answers 502. Rolling back to Gemini is one
   environment variable (`IMAGE_MODEL=`, see
   `poc/image-generation/.env.example`), and `GET :8002/health` reports
   which model each mode is really using.

3. **Local models (downloaded automatically)** -- background removal
   and cutout run on-device. On FIRST startup the service downloads,
   via the Hugging Face hub, roughly 3-4 GB total:

   - SAM 2.1 Large (segmentation, runs on Apple `mps`)
   - GroundingDINO tiny (text-guided box detection)

   First boot therefore takes a few minutes and needs network + disk;
   later boots load from the local cache in seconds. An `HF_TOKEN` env
   var is optional (higher rate limits only). There is nothing to
   install by hand.

4. **Apple Vision (built, not downloaded)** -- every background removal
   goes through `GenerateForegroundInstanceMaskRequest`, which has no
   Python binding, so a small Swift helper runs beside the service.
   **macOS 15+ only.** `scripts/start_backends.sh` compiles it; the
   binary is a build artifact and is not in the repo.

   Nothing falls back to rembg when it is missing -- every cutout
   answers 503 instead, because a silent downgrade meant 12-80s per
   image with no visible symptom. `GET :8002/health` reports
   `cutout.vision_ready`; check it before trusting a healthy-looking
   service. rembg is still installed and reachable by name, and
   `poc/image-generation/config.py` documents the rollback.

   Requires Python 3.13 (torch >= 2.5): `brew install python@3.13`,
   then `/opt/homebrew/bin/python3.13 -m venv .venv`.

## Smoke tests

```bash
for p in 8001 8002 8003 8004 8005; do curl -s -o /dev/null -w "$p %{http_code}\n" localhost:$p/docs; done
poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests -q
cd poc/image-generation && .venv/bin/python3 -m unittest discover tests
```

The iOS repo's README maps app features to these ports.

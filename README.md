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

1. **Vertex AI (cloud)** -- generation, stylize, and describe run on
   Gemini models in your GCP project:

   ```bash
   gcloud auth application-default login   # once per machine
   # .env: PROJECT_ID=<your-gcp-project>, LOCATION=us-central1
   ```

   If the project has no Imagen access (Vertex answers 404 for every
   `imagen-*` model -- true for `oxeai-dev`), `/generate` automatically
   falls back to `gemini-2.5-flash-image` (Nano Banana). Nothing to
   configure; the server logs the switch once.

2. **Local models (downloaded automatically)** -- background removal
   and cutout run on-device. On FIRST startup the service downloads,
   via the Hugging Face hub, roughly 3-4 GB total:

   - rembg `birefnet-general` (background removal)
   - SAM 2.1 Large (segmentation, runs on Apple `mps`)
   - GroundingDINO tiny (text-guided box detection)

   First boot therefore takes a few minutes and needs network + disk;
   later boots load from the local cache in seconds. An `HF_TOKEN` env
   var is optional (higher rate limits only). There is nothing to
   install by hand.

   Requires Python 3.13 (torch >= 2.5): `brew install python@3.13`,
   then `/opt/homebrew/bin/python3.13 -m venv .venv`.

## Smoke tests

```bash
for p in 8001 8002 8003 8004 8005; do curl -s -o /dev/null -w "$p %{http_code}\n" localhost:$p/docs; done
poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests -q
cd poc/image-generation && .venv/bin/python3 -m unittest discover tests
```

The iOS repo's README maps app features to these ports.

## Branches & environments

Two long-lived branches, standard across every MiraNote-AI repo:

- **`main` = prod.** Always demo-ready. Nothing is committed to it
  directly; it changes only through a `dev -> main` pull request that the
  team merges on purpose when we want a new stable/demo build.
- **`dev` = default branch.** All day-to-day work merges here first.

Flow:

```text
feature/<topic>  ->  PR into dev  ->  CI green  ->  merge into dev
      ...  (dev accumulates and integrates work)  ...
when we decide it is demo-ready:
dev  ->  PR into main  ->  merge  =>  prod updated
```

Open every new PR against `dev` (the default branch). `main` is touched
only by the deliberate `dev -> main` promotion PR, so prod always holds
exactly what the team chose to ship.

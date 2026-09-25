# MiraNote — Image Generation Service

A FastAPI service that powers MiraNote's sticker / illustration features on two
clouds plus local vision models: `/generate` runs on Aliyun Bailian (DashScope),
everything else on Google Vertex AI, and background removal and segmentation run
on this machine (Apple Vision, SAM 2.1, GroundingDINO).

It exposes four image pipelines behind one app:

| Endpoint    | What it does                                            | Models used |
|-------------|---------------------------------------------------------|-------------|
| `/generate` | Text-to-image sticker & background generation           | `z-image-turbo` (DashScope), `qwen-turbo` (prompt expansion), Apple Vision (background removal) |
| `/cutout`   | Background removal + prompt-guided subject cutout        | Apple Vision, SAM 2.1, GroundingDINO, Gemini 2.5 Flash (bbox) |
| `/stylize`  | Image-to-image style transfer                           | Gemini 2.5 Flash Image ("Nano Banana") |
| `/border`   | Sticker outlines / AI decorative borders                | Pillow (`outline`), Gemini 2.5 Flash Image (`ai_outline`) |
| `/describe` | One sentence about a photo, for page context             | `gemini-2.5-flash` (vision) |
| `/health`   | Liveness check, and which models are in use             | — |

## /generate runs on DashScope

The image model is `z-image-turbo` and the prompt expander is `qwen-turbo`, both
on Aliyun Bailian. It was the fastest and cheapest of six models on a 90-image
comparison, at `IMAGE_PRICE_CNY` 0.10 per image.

Three things to know:

- **It needs `DASHSCOPE_API_KEY`** (see `.env.example`). This is the same key
  `poc/voice-to-text` uses for transcript correction -- one account covers both.
  Without it `/generate` answers 502 naming the variable; nothing else breaks.
- **Vertex is still mandatory.** `/describe`, `/stylize`, `/border` and the bbox
  detector inside a prompted `/cutout` all still call Gemini, so ADC and
  `PROJECT_ID` are as required as before. This was not a de-Google-ing.
- **The rollback is one environment variable.** `IMAGE_MODEL=<config.MODEL_ID>`
  puts all three modes back on Gemini with no code change; `image_providers.py`
  routes that id to `generate/gemini_image.py`, which is the pre-DashScope call
  moved out of `main.py` unchanged. `GET /health` reports which model each mode
  is actually using, so you can confirm the override took.

`PROMPT_EXTEND` stays `False`, guarded by a unit test: DashScope's own prompt
rewriting would discard what `qwen-turbo` just produced, and on `z-image-turbo`
it is also the difference between 0.10 and 0.20 CNY an image.

**A failure is a failure.** There is no fallback to a second image model -- that
would make both the bill and the look of a page unpredictable. Out of credit
answers 503; a content block or anything untriaged answers 502.

## Background removal runs on Apple Vision

Every cutout -- a prompted `/cutout`, a `/cutout` with no prompt, and the matte
a generated sticker goes through -- uses Apple's
`GenerateForegroundInstanceMaskRequest`. On the 17-image bench a prompted
cutout went from 23.0s to 5.5s end to end and the no-prompt path from 12.55s
to 0.12s, better on three images and worse on none.

Two consequences worth knowing before you run this:

- **It needs macOS 15+ and a built helper.** Vision has no Python binding, so
  the work happens in a resident Swift subprocess
  (`vision_bench/vision_bench`). It is a build artifact, not in the repo.
  `scripts/start_backends.sh` builds it; to do it by hand:
  ```bash
  swiftc -O -parse-as-library vision_bench/vision_bench.swift -o vision_bench/vision_bench
  ```
- **Nothing downgrades itself.** There is deliberately no step-down to rembg
  when the helper is missing -- that would serve rembg at 12-80s per image
  while every log line and response still looked normal. Instead every cutout
  answers **503** with the reason, and `/health` reports
  `cutout.vision_ready`. Check it before assuming the service is fine:
  ```bash
  curl -s localhost:8002/health | python3 -m json.tool
  ```

rembg is still installed, still tested, and still reachable by name
(`/cutout?mode=auto`, `"matte": "rembg"`). Rolling back to it as the default
is a deliberate edit of three constants -- see **ROLLING BACK TO REMBG** in
`config.py`.

The iOS app runs Apple Vision itself for the two pure-removal paths, so it
asks `/generate` to skip the matte with `"matte": "none"`.

## Prerequisites

- **Python 3.13**
- A **DashScope (Aliyun Bailian) API key** for `/generate`.
- A **Google Cloud project** with the **Vertex AI API** enabled and access to
  the Gemini models, for everything else. The Gemini image id in `config.py` is
  served only from the `global` endpoint, so `LOCATION` must be `global` even
  though only a rollback calls it.
- **Application Default Credentials (ADC)** configured locally:
  ```bash
  gcloud auth application-default login
  ```
- First run downloads model weights (SAM 2.1, GroundingDINO), so the initial
  startup takes a while and needs network access. rembg's weights are fetched
  only if something actually asks for rembg by name.

## Setup

```bash
# 1. Create the venv and install dependencies
./setup.sh
source venv/bin/activate

# 2. Configure your project
cp .env.example .env      # then edit .env
```

`.env` holds:

```
PROJECT_ID=your-gcp-project-id
LOCATION=global
```

> `.env` is git-ignored — never commit your real project id.

## Running

```bash
source venv/bin/activate
uvicorn main:app --port 8001
```

Wait for `Application startup complete.` (the server preloads the Vertex client,
SAM, GroundingDINO and the Apple Vision helper at startup; rembg is built on
first use, because nothing reaches it on the default paths). The API is then at
`http://localhost:8001`.

## API reference

All image responses return the image as a **base64-encoded PNG** in the `image`
field.

### `POST /generate` — JSON body

| Field     | Type   | Notes |
|-----------|--------|-------|
| `command` | string | `"sticker"` or `"background"` |
| `prompt`  | string | subject prompt |
| `expand`  | bool   | expand the prompt via Gemini before generating (default `true`) |
| `matte`   | string | sticker only. `"vision"` (default), `"rembg"`, or `"none"` to get the sticker with its background still on -- how a client that mattes on its own device asks for the raw image. Echoed back as `matte_used`. |

```bash
curl -s -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"command":"sticker","prompt":"a cute red apple","expand":true}'
```

Returns `images` (a list of base64 PNGs). Stickers have their background removed
unless `matte` says otherwise; backgrounds are returned as-is. A Vision failure
here answers **503** and discards the generated image rather than quietly
re-cutting it with a different remover.

### `POST /cutout` — multipart upload + query params

| Param    | Notes |
|----------|-------|
| `file`   | the image to cut out |
| `prompt` | optional. Empty → whole-foreground matte. Set → prompt-guided cutout |
| `mode`   | with a prompt: `hybrid_sam_prebg_vision` (default), `hybrid_sam_prebg_gray`, `hybrid_sam_union`. Without one: `vision` (default) or `auto` (rembg). An unknown value is a 400, never a silent fallback |

```bash
curl -s -X POST "http://localhost:8001/cutout?prompt=the%20cat" \
  -F "file=@demo_data/2.jpeg"
```

### `POST /stylize` — multipart upload + query params

| Param         | Notes |
|---------------|-------|
| `file`        | source image |
| `style`       | preset key (e.g. `impressionist`) |
| `prompt`      | custom style description (used when no/unknown preset) |
| `temperature` | 0 = faithful to the original; higher = more creative |

### `POST /border` — multipart upload + query params

| Param        | Notes |
|--------------|-------|
| `file`       | subject image (ideally a cutout PNG with transparent background) |
| `mode`       | `outline` (pure Pillow stroke) or `ai_outline` (Gemini decorative border) |
| `color`, `width` | `outline` mode: stroke color / width |
| `prompt`, `style`, `band_ratio`, `paste_back` | `ai_outline` mode |
| `debug_dir`  | optional; if set, intermediate frames are written there |

```bash
# pure outline (no API call)
curl -s -X POST "http://localhost:8001/border?mode=outline&color=%23FFFFFF&width=32" \
  -F "file=@test_output/2cut.png"

# AI decorative border
curl -s -X POST "http://localhost:8001/border?mode=ai_outline&prompt=white%20crumpled%20paper&band_ratio=0.03&paste_back=False" \
  -F "file=@test_output/2cut.png"
```

## Testing

`test_api.py` is a **manual** test catalog (not an automated suite — no asserts).
It requires the server running on `:8001` and calls paid Vertex APIs. The default
run uses two sample images committed under `demo_data/`, so it works out of the
box. `test_input/` and `test_output/` are git-ignored: put your own images in
`test_input/` to run the extra (commented) catalog examples; results are written
to `test_output/`.

```bash
# in one terminal
uvicorn main:app --port 8001
# in another
python test_api.py
```

Each pipeline section leaves one example line active; the rest are a commented
catalog of extra examples you can enable one at a time.

## Configuration

All tunables live in [`config.py`](config.py), grouped by pipeline (model ids,
aspect ratios, cutout defaults, the Vision helper, SAM/GroundingDINO settings,
border defaults, …).
Change a value there to change behavior for the corresponding endpoint.

## Project structure

```
main.py            FastAPI app — the 5 endpoints
config.py          all tunables, grouped by pipeline
generate/          /generate  — prompt_expander, generate_presets, prompt .txt files
cutout/            /cutout    — bbox_detector, grounding_dino, sam_segmenter,
                                vision_matte (owns the Swift helper process)
vision_bench/      vision_bench.swift — the Apple Vision helper; the binary
                                beside it is built, not committed
stylize/           /stylize   — stylizer, style_presets
border/            /border    — border, border_presets
shared/            vertex_client — the shared Vertex AI genai client + response helpers
test_api.py        manual test / demo catalog
demo_data/         committed sample images used by the default test run
test_input/        your own extra images (git-ignored; not committed)
test_output/       generated results (git-ignored)
```

## Notes

- Image generation goes through the current **`google-genai`** SDK
  (`from google import genai`); the client is a shared singleton in
  `shared/vertex_client.py`.
- Local vision models run on Apple Silicon (`mps`) where configured (see
  `SAM2_DEVICE` / `GROUNDING_DEVICE` in `config.py`).

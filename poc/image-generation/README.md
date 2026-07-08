# MiraNote — Image Generation Service

A FastAPI service that powers MiraNote's sticker / illustration features on top of
Google Vertex AI (Imagen 4 + Gemini image models) plus local vision models
(rembg, SAM 2.1, GroundingDINO).

It exposes four image pipelines behind one app:

| Endpoint    | What it does                                            | Models used |
|-------------|---------------------------------------------------------|-------------|
| `/generate` | Text-to-image sticker & background generation           | Imagen 4 (`imagen-4.0-generate-001`), Gemini 2.5 Flash (prompt expansion), rembg (background removal) |
| `/cutout`   | Background removal + prompt-guided subject cutout        | rembg, SAM 2.1, GroundingDINO, Gemini 2.5 Flash (bbox) |
| `/stylize`  | Image-to-image style transfer                           | Gemini 2.5 Flash Image ("Nano Banana") |
| `/border`   | Sticker outlines / AI decorative borders                | Pillow (`outline`), Gemini 2.5 Flash Image (`ai_outline`) |
| `/health`   | Liveness check                                          | — |

## Prerequisites

- **Python 3.13**
- A **Google Cloud project** with the **Vertex AI API** enabled, and access to
  Imagen 4 and Gemini image models in your region (default `us-central1`).
- **Application Default Credentials (ADC)** configured locally:
  ```bash
  gcloud auth application-default login
  ```
- First run downloads model weights (SAM 2.1, GroundingDINO, rembg `birefnet-general`),
  so the initial startup takes a while and needs network access.

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
LOCATION=us-central1
```

> `.env` is git-ignored — never commit your real project id.

## Running

```bash
source venv/bin/activate
uvicorn main:app --port 8001
```

Wait for `Application startup complete.` (the server preloads the Vertex client,
rembg, SAM, and GroundingDINO at startup). The API is then at
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

```bash
curl -s -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"command":"sticker","prompt":"a cute red apple","expand":true}'
```

Returns `images` (a list of base64 PNGs). Stickers have their background removed;
backgrounds are returned as-is.

### `POST /cutout` — multipart upload + query params

| Param    | Notes |
|----------|-------|
| `file`   | the image to cut out |
| `prompt` | optional. Empty → full-image rembg (`auto`). Set → prompt-guided cutout |
| `mode`   | `hybrid_sam_prebg_gray` (default) or `hybrid_sam_union` |

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
aspect ratios, rembg model, SAM/GroundingDINO settings, border defaults, …).
Change a value there to change behavior for the corresponding endpoint.

## Project structure

```
main.py            FastAPI app — the 5 endpoints
config.py          all tunables, grouped by pipeline
generate/          /generate  — prompt_expander, generate_presets, prompt .txt files
cutout/            /cutout    — bbox_detector, grounding_dino, sam_segmenter
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

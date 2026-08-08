"""
MiraNote POC -- Image Generation API
Imagen 4 sticker generation with rembg background removal.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import asyncio
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, ImageFilter
from pydantic import BaseModel
from rembg import remove, new_session

import config
from shared.vertex_client import _get_client
from generate import fallback, prompt_expander, generate_presets
from cutout import bbox_detector, sam_segmenter, grounding_dino
from stylize import stylizer, style_presets
from border import border, border_presets


_imagen_unavailable = False


def _call_model(prompt: str, aspect_ratio: str) -> list[bytes]:
    global _imagen_unavailable
    if not _imagen_unavailable:
        try:
            response = _get_client().models.generate_images(
                model=config.MODEL_ID,
                prompt=prompt,
                config={
                    "number_of_images": config.NUMBER_OF_IMAGES,
                    "aspect_ratio": aspect_ratio,
                },
            )
            return [img.image.image_bytes for img in response.generated_images]
        except Exception as error:
            if not fallback.is_model_unavailable(error):
                raise
            # Imagen is gated per project; remember and stop retrying it.
            _imagen_unavailable = True
            print(f"[generate] Imagen unavailable, using {config.FALLBACK_IMAGE_MODEL}: {str(error)[:100]}")
    client = _get_client()

    def _one() -> bytes | None:
        response = client.models.generate_content(
            model=config.FALLBACK_IMAGE_MODEL,
            contents=fallback.build_prompt(prompt, aspect_ratio),
        )
        parts = fallback.image_parts(response)
        return parts[0] if parts else None

    # The images are independent; generate them concurrently so the whole
    # request stays comfortably inside client timeouts.
    with ThreadPoolExecutor(max_workers=config.NUMBER_OF_IMAGES) as pool:
        results = list(pool.map(lambda _: _one(), range(config.NUMBER_OF_IMAGES)))
    images = [image for image in results if image]
    if not images:
        raise HTTPException(status_code=502, detail="image generation returned no image")
    return images


_rembg_session = None


class _NotFound(Exception):
    pass


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    if iy2 <= iy1 or ix2 <= ix1:
        return 0.0
    inter = (iy2 - iy1) * (ix2 - ix1)
    area_a = (ay2 - ay1) * (ax2 - ax1)
    area_b = (by2 - by1) * (bx2 - bx1)
    return inter / (area_a + area_b - inter)


def _union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Smallest box covering both. Boxes in 0-1000 (y_min, x_min, y_max, x_max)."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _normalized_to_pixels(image_bytes: bytes, bbox_norm: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    y_min, x_min, y_max, x_max = bbox_norm
    return ((x_min / 1000) * w, (y_min / 1000) * h, (x_max / 1000) * w, (y_max / 1000) * h)


def _flatten_on_bg(rgba_png_bytes: bytes, color: tuple[int, int, int]) -> bytes:
    """Composite an RGBA cutout onto a solid background color, return RGB PNG.

    Downstream detectors (DINO/SAM) convert("RGB"), which turns rembg's transparent
    pixels black; flatten onto white/gray instead so the background is clean but neutral.
    """
    img = Image.open(io.BytesIO(rgba_png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, color + (255,))
    flat = Image.alpha_composite(bg, img).convert("RGB")
    buf = io.BytesIO()
    flat.save(buf, format="PNG")
    return buf.getvalue()


async def _cutout_via_hybrid_prebg(image_bytes: bytes, prompt: str):
    """Full-image rembg -> flatten onto a solid gray bg -> hybrid_sam_union.

    Removing the background first stops the union step from spanning competing
    subjects in cluttered/multi-subject scenes -- where plain hybrid_sam_union
    tends to grab the whole frame. Downstream detectors convert("RGB"), so we
    flatten onto a neutral solid color instead of leaving rembg's transparency
    (which would otherwise become black).
    """
    removed = await asyncio.to_thread(remove, image_bytes, session=_rembg_session)
    flattened = _flatten_on_bg(removed, (128, 128, 128))
    out, bbox, extras = await _cutout_via_hybrid_sam(flattened, prompt)
    extras = extras or {}
    extras["prebg"] = "gray"
    return out, bbox, extras


def _apply_fullsize_mask(image_bytes: bytes, mask_png: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_png)).convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
    img.putalpha(mask)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _cutout_via_hybrid_sam(image_bytes: bytes, prompt: str):
    dino_task = asyncio.to_thread(
        grounding_dino.detect_all_boxes, image_bytes, prompt, config.HYBRID_DINO_THRESHOLD
    )
    gemini_task = asyncio.to_thread(
        bbox_detector.detect_bbox, image_bytes, prompt, config.BBOX_DETECTOR_MODEL
    )
    dino_candidates, gemini_bbox = await asyncio.gather(
        dino_task, gemini_task, return_exceptions=False
    )

    if not dino_candidates and gemini_bbox is None:
        raise _NotFound(f"hybrid_sam: '{prompt}' not found")

    if gemini_bbox is None:
        chosen_bbox = dino_candidates[0][0]
        used_path = "dino-only"
    elif not dino_candidates:
        chosen_bbox = gemini_bbox
        used_path = "gemini-only"
    else:
        rated = [(box, score, _iou(gemini_bbox, box)) for box, score in dino_candidates]
        passing = [(b, s, m) for b, s, m in rated if m >= config.HYBRID_IOU_THRESHOLD]
        if passing:
            box, score, m = max(passing, key=lambda t: t[2])   # highest IoU wins
            chosen_bbox = _union(box, gemini_bbox)
            used_path = f"dino(iou={m:.2f}, score={score:.2f})+union"
        else:
            chosen_bbox = gemini_bbox
            best = max((m for _b, _s, m in rated), default=0.0)
            used_path = f"gemini(best_iou={best:.2f})"

    print(f"[hybrid] {used_path} bbox={chosen_bbox}")

    bbox_pixels = _normalized_to_pixels(image_bytes, chosen_bbox)
    mask_png = await asyncio.to_thread(sam_segmenter.segment_with_bbox, image_bytes, bbox_pixels)
    out = _apply_fullsize_mask(image_bytes, mask_png)
    extras = {
        "dino_bboxes": [list(b) for b, _s in dino_candidates],
        "dino_scores": [round(s, 3) for _b, s in dino_candidates],
        "gemini_bbox": list(gemini_bbox) if gemini_bbox else None,
        "chosen_path": used_path,
    }
    return out, chosen_bbox, extras


PROMPT_CUTOUT_MODES = {
    "hybrid_sam_union": _cutout_via_hybrid_sam,             # DINO candidates disambiguated by Gemini box (IoU + union) -> SAM
    "hybrid_sam_prebg_gray": _cutout_via_hybrid_prebg,     # DEFAULT: rembg -> gray bg -> hybrid_sam_union
}


def _parse_hex(color: str) -> tuple[int, int, int, int]:
    """'#RRGGBB' or '#RRGGBBAA' -> (r, g, b, a). Raises HTTPException(400) on bad input."""
    s = color.lstrip("#")
    if len(s) not in (6, 8):
        raise HTTPException(status_code=400, detail=f"invalid hex color '{color}'")
    try:
        rgb = tuple(int(s[i:i + 2], 16) for i in range(0, 6, 2))
        a = int(s[6:8], 16) if len(s) == 8 else 255
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid hex color '{color}'")
    return rgb + (a,)


def _safe_debug_dir(debug_dir: str) -> str | None:
    """Confine /border debug frames under a fixed test_output/ root.

    debug_dir is a client-supplied request field and ai_outline_border writes PNG
    frames into it, so an unchecked value lets a caller create directories and
    write files anywhere on the server. Reject absolute paths and parent-directory
    traversal; nest any accepted relative path under test_output/.
    """
    if not debug_dir:
        return None
    norm = os.path.normpath(debug_dir)
    if os.path.isabs(norm) or norm == ".." or norm.startswith(".." + os.sep):
        raise HTTPException(
            status_code=400,
            detail="debug_dir must be a relative path under test_output/",
        )
    root = "test_output"
    if norm != root and not norm.startswith(root + os.sep):
        norm = os.path.join(root, norm)
    return norm


def _erode_alpha(png_bytes: bytes, radius: int) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    r, g, b, a = img.split()
    a = a.filter(ImageFilter.MinFilter(radius * 2 + 1))
    img = Image.merge("RGBA", (r, g, b, a))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rembg_session
    print("Connecting to Vertex AI (genai client)...")
    _get_client()
    print("Vertex AI client ready.")
    print(f"Loading rembg session ({config.REMBG_MODEL})...")
    _rembg_session = new_session(config.REMBG_MODEL)
    print("rembg session ready.")
    print("Loading SAM ONNX sessions...")
    sam_segmenter.preload()
    print("SAM ready.")
    print("Loading GroundingDINO...")
    grounding_dino.preload()
    print("GroundingDINO ready.")
    yield


app = FastAPI(title="MiraNote Image Generation", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    command: str        # "sticker" | "background"
    prompt: str = ""    # user-written prompt for sticker
    expand: bool = True   # if True, expand prompt via LLM before generation


@app.post("/generate")
async def generate_images(req: GenerateRequest):
    if req.command == "sticker":
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt is required for sticker")
        core = await asyncio.to_thread(
            prompt_expander.expand, req.prompt, config.PROMPT_EXPANDER_MODEL
        ) if req.expand else req.prompt
        prompt = generate_presets.build_sticker_prompt(core)
    elif req.command == "background":
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt is required for background")
        core = await asyncio.to_thread(
            prompt_expander.expand_background, req.prompt, config.PROMPT_EXPANDER_MODEL
        ) if req.expand else req.prompt
        prompt = generate_presets.build_background_prompt(core)
    elif req.command == "art":
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt is required for art")
        core = await asyncio.to_thread(
            prompt_expander.expand_art, req.prompt, config.PROMPT_EXPANDER_MODEL
        ) if req.expand else req.prompt
        prompt = generate_presets.build_art_prompt(core)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {req.command}")

    ratio = config.ASPECT_RATIOS.get(req.command, "1:1")
    images = await asyncio.to_thread(_call_model, prompt, ratio)

    remove_bg = config.REMOVE_BG and req.command == "sticker"
    encoded = []
    for raw in images:
        processed = remove(raw, session=_rembg_session) if remove_bg else raw
        if remove_bg and config.REMBG_ERODE_RADIUS > 0:
            processed = _erode_alpha(processed, config.REMBG_ERODE_RADIUS)
        encoded.append(base64.b64encode(processed).decode())

    return {"command": req.command, "prompt": prompt, "raw_input": req.prompt, "images": encoded, "count": len(encoded)}


@app.post("/cutout")
async def cutout_image(
    file: UploadFile,
    prompt: str = "",
    mode: str = "",
):
    raw = await file.read()

    if prompt:
        chosen = mode or config.DEFAULT_PROMPT_CUTOUT_MODE
        if chosen not in PROMPT_CUTOUT_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown mode '{chosen}'; valid: {list(PROMPT_CUTOUT_MODES)}",
            )
        try:
            processed, bbox, extras = await PROMPT_CUTOUT_MODES[chosen](raw, prompt)
            used = chosen
        except _NotFound as e:
            raise HTTPException(status_code=404, detail=str(e))
        response = {
            "image": base64.b64encode(processed).decode(),
            "mode_used": used,
            "prompt": prompt,
            "bbox": bbox,
        }
        if extras:
            response.update(extras)
        return response

    processed = await asyncio.to_thread(remove, raw, session=_rembg_session)
    if config.REMBG_ERODE_RADIUS > 0:
        processed = _erode_alpha(processed, config.REMBG_ERODE_RADIUS)
    return {
        "image": base64.b64encode(processed).decode(),
        "mode_used": "auto",
    }


@app.post("/stylize")
async def stylize_image(
    file: UploadFile,
    style: str = "",          # preset key, e.g. "impressionist"
    prompt: str = "",         # custom style description (used when no/unknown preset)
    temperature: float | None = None,  # 0 = faithful to original; higher = more creative
):
    raw = await file.read()
    try:
        instruction = style_presets.build_instruction(style=style, prompt=prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    temp = config.STYLE_TEMPERATURE if temperature is None else temperature
    processed = await asyncio.to_thread(
        stylizer.stylize, raw, instruction, config.STYLE_MODEL, temp
    )
    return {
        "image": base64.b64encode(processed).decode(),
        "style_used": style or "custom",
        "instruction": instruction,
        "temperature": temp,
    }


@app.post("/describe")
async def describe_image(file: UploadFile, prompt: str = None):
    """Vision over one image.

    Default: one warm sentence about the photo, for the app's page
    context -- so the journaling chat can "see" what sits on the page.
    Canvas mode passes its own `prompt` to ask what a whole page looks
    like instead."""
    raw = await file.read()
    question = config.describe_question(prompt)

    def _describe() -> str:
        from google.genai import types
        response = _get_client().models.generate_content(
            model=config.PROMPT_EXPANDER_MODEL,
            contents=[
                types.Part.from_bytes(data=raw, mime_type=file.content_type or "image/png"),
                question,
            ],
        )
        return (response.text or "").strip()

    description = await asyncio.to_thread(_describe)
    if not description:
        raise HTTPException(status_code=502, detail="the model returned no description")
    return {"description": description}


@app.post("/border")
async def border_image(
    file: UploadFile,
    mode: str = "outline",        # "outline" | "ai_outline"
    # outline mode
    color: str = config.BORDER_COLOR,
    width: int = config.BORDER_WIDTH,
    # ai_outline mode
    style: str = "",
    prompt: str = "",
    band_ratio: float = config.BORDER_BAND_RATIO,
    paste_back: bool = True,  # False: keep the model's own subject (skip cutout paste-back)
    white_edge: bool | None = None,  # None: ai_outline default (off)
    white_edge_width: int = config.WHITE_EDGE_WIDTH,
    shadow: bool = True,
    temperature: float | None = None,
    debug_dir: str = "",
):
    raw = await file.read()

    if mode == "outline":
        out = await asyncio.to_thread(border.outline_cutout, raw, _parse_hex(color), width)
    elif mode == "ai_outline":
        try:
            instruction = border_presets.build_outline_instruction(style=style, prompt=prompt)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        temp = config.BORDER_TEMPERATURE if temperature is None else temperature
        we = False if white_edge is None else white_edge  # ai_outline: white edge off by default
        out = await asyncio.to_thread(
            border.ai_outline_border, raw, instruction, config.BORDER_MODEL,
            band_ratio, (255, 255, 255, 255), border._DEFAULT_BG,
            paste_back, we, white_edge_width, shadow, temp,
            config.BORDER_WORK_SIZE, _safe_debug_dir(debug_dir), config.BORDER_GUIDE_MIN_RATIO,
        )
    else:
        raise HTTPException(status_code=400, detail=f"unknown mode '{mode}'; valid: outline, ai_outline")

    return {"image": base64.b64encode(out).decode(), "mode_used": mode}


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.MODEL_ID}

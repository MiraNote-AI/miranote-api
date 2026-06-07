"""
MiraNote POC -- Image Generation API
Imagen 4 sticker generation with rembg background removal.
"""

import asyncio
import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, ImageFilter
from pydantic import BaseModel
from rembg import remove, new_session

from dotenv import load_dotenv
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

import config
from prompts import prompt_builder

load_dotenv()
_PROJECT_ID = os.environ["PROJECT_ID"]
_LOCATION = os.environ["LOCATION"]


def _init_model() -> ImageGenerationModel:
    vertexai.init(project=_PROJECT_ID, location=_LOCATION)
    return ImageGenerationModel.from_pretrained(config.MODEL_ID)


def _call_model(model: ImageGenerationModel, prompt: str, aspect_ratio: str):
    return model.generate_images(
        prompt=prompt,
        number_of_images=config.NUMBER_OF_IMAGES,
        aspect_ratio=aspect_ratio,
    )


_model = None
_rembg_session = None
_rembg_session_human = None

_PROMPT_DIR = Path(__file__).parent / "prompts"
_STICKER_SUFFIX = (_PROMPT_DIR / "sticker_suffix.txt").read_text(encoding="utf-8").strip()
_BACKGROUND_RULE = (_PROMPT_DIR / "background_rule.txt").read_text(encoding="utf-8").strip()


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
    global _model, _rembg_session, _rembg_session_human
    print("Loading Imagen model...")
    _model = _init_model()
    print("Imagen model loaded.")
    print(f"Loading rembg session ({config.REMBG_MODEL})...")
    _rembg_session = new_session(config.REMBG_MODEL)
    print(f"Loading rembg human session ({config.REMBG_HUMAN_MODEL})...")
    _rembg_session_human = new_session(config.REMBG_HUMAN_MODEL)
    print("rembg sessions ready.")
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
        if req.expand:
            prompt = await asyncio.to_thread(
                prompt_builder.expand, req.prompt, config.PROMPT_BUILDER_MODEL
            )
        else:
            prompt = req.prompt
        prompt = f"{prompt}, {_STICKER_SUFFIX}"
    elif req.command == "background":
        if not req.prompt:
            raise HTTPException(status_code=400, detail="prompt is required for background")
        if req.expand:
            refined = await asyncio.to_thread(
                prompt_builder.expand_background, req.prompt, config.PROMPT_BUILDER_MODEL
            )
            prompt = refined + _BACKGROUND_RULE
        else:
            prompt = req.prompt + _BACKGROUND_RULE
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {req.command}")

    ratio = config.ASPECT_RATIOS.get(req.command, "1:1")
    images = await asyncio.to_thread(_call_model, _model, prompt, ratio)

    remove_bg = config.REMOVE_BG and req.command == "sticker"
    encoded = []
    for img in images:
        raw = img._image_bytes  # no public bytes accessor in this SDK version
        processed = remove(raw, session=_rembg_session) if remove_bg else raw
        if remove_bg and config.REMBG_ERODE_RADIUS > 0:
            processed = _erode_alpha(processed, config.REMBG_ERODE_RADIUS)
        encoded.append(base64.b64encode(processed).decode())

    return {"command": req.command, "prompt": prompt, "raw_input": req.prompt, "images": encoded, "count": len(encoded)}


@app.post("/cutout")
async def cutout_image(file: UploadFile, type: str = "general"):
    raw = await file.read()
    session = _rembg_session_human if type == "person" else _rembg_session
    processed = await asyncio.to_thread(remove, raw, session=session)
    if config.REMBG_ERODE_RADIUS > 0:
        processed = _erode_alpha(processed, config.REMBG_ERODE_RADIUS)
    return {"image": base64.b64encode(processed).decode(), "type": type}


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.MODEL_ID}

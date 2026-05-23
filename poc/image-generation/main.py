"""
MiraNote POC -- Image Generation API
Imagen 3 image generation with optional rembg background removal for stickers and dividers.
"""

import asyncio
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rembg import remove

import config
from generate import _call_model, init_model
from prompts import journal_styles

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print("Loading Imagen model...")
    _model = init_model()
    print("Imagen model loaded.")
    yield


app = FastAPI(title="MiraNote Image Generation", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    command: str          # "sticker" | "background" | "divider"
    subject: str = ""     # for sticker
    palette: str = "pastel"
    season: str = "spring"
    mood: str = "cozy"
    style: str = "floral"


@app.post("/generate")
async def generate_images(req: GenerateRequest):
    if req.command == "sticker":
        prompt = journal_styles.sticker(req.subject, req.palette)
    elif req.command == "background":
        prompt = journal_styles.background(req.season, req.mood)
    elif req.command == "divider":
        prompt = journal_styles.divider(req.style)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {req.command}")

    ratio = config.ASPECT_RATIOS.get(req.command, "1:1")
    images = await asyncio.to_thread(_call_model, _model, prompt, ratio)

    remove_bg = req.command in ("sticker", "divider")
    encoded = []
    for img in images:
        raw = img._image_bytes  # no public bytes accessor in this SDK version
        processed = remove(raw) if remove_bg else raw
        encoded.append(base64.b64encode(processed).decode())

    return {"command": req.command, "prompt": prompt, "images": encoded, "count": len(encoded)}


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.MODEL_ID}

"""
MiraNote POC -- Text Clean & Expand API
Clean: fix typos/punctuation/grammar, preserve original meaning.
Expand: give user ideas to continue writing, concise and inspiring.
"""

from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY is required. Set it in .env")

client_kwargs = {"api_key": LLM_API_KEY}
if LLM_BASE_URL:
    client_kwargs["base_url"] = LLM_BASE_URL
client = OpenAI(**client_kwargs)

app = FastAPI(title="MiraNote Text Clean & Expand", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Prompts (loaded from external files to keep source CJK-free) --

_PROMPT_DIR = Path(__file__).parent / "prompts"
CLEAN_SYSTEM = (_PROMPT_DIR / "clean.txt").read_text(encoding="utf-8")
EXPAND_SYSTEM = (_PROMPT_DIR / "expand.txt").read_text(encoding="utf-8")
POLISH_SYSTEM = (_PROMPT_DIR / "polish.txt").read_text(encoding="utf-8")
SHORTEN_SYSTEM = (_PROMPT_DIR / "shorten.txt").read_text(encoding="utf-8")
KEYWORDS_SYSTEM = (_PROMPT_DIR / "keywords.txt").read_text(encoding="utf-8")
CAPTION_SYSTEM = (_PROMPT_DIR / "caption.txt").read_text(encoding="utf-8")


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User's raw text input")
    context: Optional[str] = Field(None, description="Optional surrounding context for better expansion")


class CleanResponse(BaseModel):
    original: str
    cleaned: str


class ExpandResponse(BaseModel):
    original: str
    expanded: str


class PolishResponse(BaseModel):
    original: str
    polished: str


class ShortenRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to shorten")
    target: Literal["30%", "50%", "tweet"] = Field(
        "50%", description="How aggressively to shorten"
    )


class ShortenResponse(BaseModel):
    original: str
    shortened: str
    target: str


class KeywordsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max: int = Field(10, ge=1, le=20, description="Maximum keywords to return")


class Keyword(BaseModel):
    term: str = Field(..., min_length=1, max_length=64)
    score: int = Field(..., ge=1, le=10)


class KeywordsResponse(BaseModel):
    original: str
    keywords: List[Keyword]


class CaptionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    style: Literal["instagram", "diary", "tweet"] = Field("instagram")


class CaptionResponse(BaseModel):
    original: str
    caption: str
    style: str


async def call_llm(system: str, user_text: str, max_tokens: int = 2048) -> str:
    for attempt in range(3):
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 45 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                await asyncio.sleep(wait)
            else:
                raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    raise HTTPException(status_code=502, detail="LLM call failed after retries")


@app.post("/clean", response_model=CleanResponse)
async def clean_text(req: TextRequest):
    """Clean up messy text: fix typos, punctuation, grammar. Minimal changes, preserve original meaning."""
    cleaned = await call_llm(CLEAN_SYSTEM, req.text, max_tokens=2048)
    return CleanResponse(original=req.text, cleaned=cleaned)


@app.post("/expand", response_model=ExpandResponse)
async def expand_text(req: TextRequest):
    """Expand on user's input: provide concise directions and ideas to continue writing."""
    user_msg = req.text
    if req.context:
        user_msg = f"Context:\n{req.context}\n\nCurrent input:\n{req.text}"
    expanded = await call_llm(EXPAND_SYSTEM, user_msg, max_tokens=2048)
    return ExpandResponse(original=req.text, expanded=expanded)


@app.post("/polish", response_model=PolishResponse)
async def polish_text(req: TextRequest):
    """Polish: final editing pass. Improve word choice and flow, preserve structure and meaning."""
    polished = await call_llm(POLISH_SYSTEM, req.text, max_tokens=2048)
    return PolishResponse(original=req.text, polished=polished)


@app.post("/shorten", response_model=ShortenResponse)
async def shorten_text(req: ShortenRequest):
    """Shorten: produce a shorter version preserving meaning. Target controls aggressiveness."""
    user_msg = f"Target: {req.target}\n\n{req.text}"
    shortened = await call_llm(SHORTEN_SYSTEM, user_msg, max_tokens=2048)
    return ShortenResponse(original=req.text, shortened=shortened, target=req.target)


@app.post("/keywords", response_model=KeywordsResponse)
async def keywords_endpoint(req: KeywordsRequest):
    """Extract 5-10 salient keywords as [{term, score}]. Score is 1-10 LLM-assigned salience."""
    user_msg = f"max={req.max}\n\n{req.text}"
    raw = await call_llm(KEYWORDS_SYSTEM, user_msg, max_tokens=1024)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail=f"LLM emitted invalid JSON: {raw[:200]}",
        )
    try:
        keywords_list = [Keyword(**k) for k in parsed][: req.max]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"LLM emitted unexpected schema: {raw[:200]} ({e})",
        )
    return KeywordsResponse(original=req.text, keywords=keywords_list)


@app.post("/caption", response_model=CaptionResponse)
async def caption_endpoint(req: CaptionRequest):
    """Generate a 1-2 sentence caption in the given style."""
    user_msg = f"style={req.style}\n\n{req.text}"
    caption = await call_llm(CAPTION_SYSTEM, user_msg, max_tokens=512)
    return CaptionResponse(original=req.text, caption=caption, style=req.style)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


# -- Static files (frontend) --
STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

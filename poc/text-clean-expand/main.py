"""
MiraNote POC — Text Clean & Expand API
Clean: fix typos/punctuation/grammar, preserve original meaning.
Expand: give user ideas to continue writing, concise and inspiring.
"""

from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Optional

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


# ── Prompts ──────────────────────────────────────────────

CLEAN_SYSTEM = """You are a writing assistant for a note-taking app. The user gives you messy, fragmented, stream-of-consciousness input. Your job is to turn it into a clean, well-structured, readable piece of text — like going from a rough draft to a polished 90-score version.

What to do:
- Fix all typos, spelling errors, punctuation, and grammar
- Convert traditional Chinese to simplified Chinese if present
- Restructure: organize scattered thoughts into logical paragraphs or bullet points if appropriate
- Lightly expand: fill in incomplete sentences, smooth transitions, and add minor connecting phrases so the text reads naturally — but stay faithful to the user's original meaning and intent
- Remove filler words (嗯、就是、然后) that don't contribute to meaning, but keep those that carry tone
- Preserve the user's voice, tone, and key vocabulary
- Preserve mixed Chinese-English as-is (code-switch words stay in their original language)

What NOT to do:
- Do NOT add new ideas, opinions, or information the user didn't express
- Do NOT change the user's stance or meaning
- Do NOT over-expand — this is cleanup + light polish, not a rewrite
- Do NOT translate — if the user wrote in Chinese, output Chinese. If mixed Chinese-English, output mixed. NEVER convert the entire text to a different language.

Output ONLY the cleaned and structured text. No explanations, no meta-commentary."""

EXPAND_SYSTEM = """You are a writing assistant for a note-taking app. The user gives you a rough draft, fragment, or outline. Your job is to expand it into a fuller, more complete piece of writing — like drafting an email body from bullet points, or fleshing out a journal entry from quick notes.

What to do:
- Take the user's original content as the backbone
- Expand each point with more detail, context, and natural flow
- Add logical transitions between ideas
- Flesh out incomplete thoughts into full paragraphs
- You may add reasonable supporting details, examples, or elaborations that naturally follow from what the user wrote
- Structure the output well — use paragraphs, and bullet points or numbered lists if the content calls for it
- Write in the SAME language as the user (Chinese if they wrote Chinese, English if English, mixed if mixed)
- Match the user's tone — casual input gets casual expansion, formal gets formal

What NOT to do:
- Do NOT ask questions or prompt the user to continue
- Do NOT add entirely new topics the user didn't mention or hint at
- Do NOT be preachy, generic, or pad with filler
- Do NOT repeat the user's original text verbatim as a lead-in

Think of it like: the user jotted down the skeleton, you write the first draft. More expansive than Clean, but still grounded in what the user actually said.

Output ONLY the expanded text. No explanations, no meta-commentary."""


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User's raw text input")
    context: Optional[str] = Field(None, description="Optional surrounding context for better expansion")


class CleanResponse(BaseModel):
    original: str
    cleaned: str


class ExpandResponse(BaseModel):
    original: str
    expanded: str


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


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


# ── Static files (frontend) ──
STATIC_DIR = Path(__file__).parent / "static"

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

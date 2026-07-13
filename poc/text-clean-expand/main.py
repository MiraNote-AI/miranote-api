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


def _contains_cjk(text: str) -> bool:
    # CJK Unified Ideographs, U+4E00..U+9FFF (chr() keeps this file ASCII-only)
    return any(chr(0x4E00) <= ch <= chr(0x9FFF) for ch in text)


# Common simplified/traditional variant pairs. Spot-checking these is enough to
# catch gross script drift (a fully traditional reply to a simplified note);
# texts that hit neither set are treated as indeterminate and never retried.
# Pairs: hui/shuo/ji(record)/jiang/kai/guan/men/wen/shi/dong/che/jian/chang/
# tou/dian/xian/fa/jing/ge/men(plural)/zhe/wei/xue/xie/yu/wan
_SIMPLIFIED_CHARS = frozenset(map(chr, [
    0x4F1A, 0x8BF4, 0x8BB0, 0x8BB2, 0x5F00, 0x5173, 0x95E8, 0x95EE, 0x65F6,
    0x4E1C, 0x8F66, 0x89C1, 0x957F, 0x5934, 0x70B9, 0x73B0, 0x53D1, 0x7ECF,
    0x4E2A, 0x4EEC, 0x8FD9, 0x4E3A, 0x5B66, 0x5199, 0x4E0E, 0x4E07,
]))
_TRADITIONAL_CHARS = frozenset(map(chr, [
    0x6703, 0x8AAA, 0x8A18, 0x8B1B, 0x958B, 0x95DC, 0x9580, 0x554F, 0x6642,
    0x6771, 0x8ECA, 0x898B, 0x9577, 0x982D, 0x9EDE, 0x73FE, 0x767C, 0x7D93,
    0x500B, 0x5011, 0x9019, 0x70BA, 0x5B78, 0x5BEB, 0x8207, 0x842C,
]))


def _chinese_script(text: str) -> Optional[str]:
    """'simplified' / 'traditional' when clearly one of them, else None."""
    has_simp = any(ch in _SIMPLIFIED_CHARS for ch in text)
    has_trad = any(ch in _TRADITIONAL_CHARS for ch in text)
    if has_simp and not has_trad:
        return "simplified"
    if has_trad and not has_simp:
        return "traditional"
    return None


def _language_correction(ref: str, reply: str) -> Optional[str]:
    """A corrective instruction when the reply's language does not match the
    reference text, or None when it matches (or is indeterminate)."""
    if _contains_cjk(ref) != _contains_cjk(reply):
        if _contains_cjk(ref):
            return (
                "Your reply is in the wrong language. The note contains Chinese, "
                "so rewrite your entire reply in Chinese. Keep English technical "
                "terms as-is. Do not translate the note itself. Every other rule "
                "in the system prompt still applies to the rewrite."
            )
        return (
            "Your reply is in the wrong language. The note is entirely in "
            "English, so rewrite your entire reply in English only, with "
            "no Chinese characters. Every other rule in the system prompt "
            "still applies to the rewrite."
        )
    if _contains_cjk(ref):
        ref_script = _chinese_script(ref)
        reply_script = _chinese_script(reply)
        if ref_script and reply_script and ref_script != reply_script:
            return (
                f"Your reply uses {reply_script} Chinese characters, but the "
                f"note is written in {ref_script} Chinese. Rewrite your entire "
                f"reply using {ref_script} Chinese characters. Every other rule "
                "in the system prompt still applies to the rewrite."
            )
    return None


async def _completion(messages: list[dict], max_tokens: int, temperature: float) -> str:
    for attempt in range(3):
        try:
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
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


_GROUNDING_CHECK_SYSTEM = (
    "You are a strict fact checker for a journal app. Compare NOTE and "
    "EXPANSION. List every concrete factual claim in EXPANSION that is not "
    "supported by NOTE: events, actions, people, places, times, numbers, "
    "objects, quoted dialogue. Feelings, generic reflections, intentions, and "
    "rephrasings of NOTE content are fine and must NOT be listed. "
    "Reply with a JSON array of short strings quoting the invented claims, "
    "[] if there are none. Reply with JSON only, no other text."
)


async def _grounding_violations(note: str, expansion: str) -> list[str]:
    """Concrete facts in `expansion` that `note` never stated ([] on checker
    failure -- the checker must never take the endpoint down)."""
    raw = await _completion(
        [
            {"role": "system", "content": _GROUNDING_CHECK_SYSTEM},
            {"role": "user", "content": f"NOTE:\n{note}\n\nEXPANSION:\n{expansion}"},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


async def call_llm(
    system: str,
    user_text: str,
    max_tokens: int = 2048,
    language_ref: Optional[str] = None,
    temperature: float = 1.0,
    max_output_chars: Optional[int] = None,
    grounding_check: bool = False,
) -> str:
    """Call the LLM; the reply must match the language of `language_ref`
    (defaults to `user_text`): Chinese stays Chinese (simplified stays
    simplified, traditional stays traditional), English stays English.
    `max_output_chars` additionally bounds the reply length (the model
    ignores the prompt's length rule and novelizes; over-length replies are
    where the invented details live).

    The model sometimes ignores these rules in the prompt, so violations are
    detected here and corrected with one explicit retry. If the retry
    violates too, the reply is returned as-is: an imperfect reply is still
    usable, a 502 is not.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]
    reply = await _completion(messages, max_tokens, temperature)

    ref = user_text if language_ref is None else language_ref
    notes = []
    if language_note := _language_correction(ref, reply):
        notes.append(language_note)
    if max_output_chars is not None and len(reply) > max_output_chars:
        notes.append(
            f"Your reply is {len(reply)} characters -- far too long for this "
            f"note. Rewrite it in at most {max_output_chars} characters. Cut "
            "invented details first: keep only content grounded in the note, "
            "in the user's voice. Do not summarize the note away; just stop "
            "adding things it never said."
        )
    if notes:
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "\n\n".join(notes)},
        ]
        reply = await _completion(messages, max_tokens, temperature)

    # Fabrication is checked last, on whatever reply survived the retry:
    # a fact-checker pass lists invented specifics, one rewrite removes them.
    if grounding_check:
        violations = await _grounding_violations(user_text, reply)
        if violations:
            listed = "\n".join(f"- {v}" for v in violations)
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": (
                        "These details in your reply are NOT in the note -- "
                        f"they are invented:\n{listed}\n"
                        "Rewrite your reply without them. Keep everything that "
                        "IS grounded in the note, same language, same voice"
                        + (
                            f", at most {max_output_chars} characters."
                            if max_output_chars is not None
                            else "."
                        )
                    ),
                },
            ]
            reply = await _completion(messages, max_tokens, temperature)
    return reply


@app.post("/clean", response_model=CleanResponse)
async def clean_text(req: TextRequest):
    """Clean up messy text: fix typos, punctuation, grammar. Minimal changes, preserve original meaning."""
    cleaned = await call_llm(CLEAN_SYSTEM, req.text, max_tokens=2048, temperature=0.3)
    return CleanResponse(original=req.text, cleaned=cleaned)


@app.post("/expand", response_model=ExpandResponse)
async def expand_text(req: TextRequest):
    """Expand on user's input: provide concise directions and ideas to continue writing."""
    user_msg = req.text
    if req.context:
        user_msg = f"Context:\n{req.context}\n\nCurrent input:\n{req.text}"
    # 0.3: some creative room, but low enough to curb invented specifics.
    # The char cap (2x + slack) is the real fence against novelizing; the
    # prompt's own length rule alone is ignored.
    expanded = await call_llm(
        EXPAND_SYSTEM,
        user_msg,
        max_tokens=2048,
        language_ref=req.text,
        temperature=0.3,
        max_output_chars=2 * len(req.text) + 150,
        grounding_check=True,
    )
    return ExpandResponse(original=req.text, expanded=expanded)


@app.post("/polish", response_model=PolishResponse)
async def polish_text(req: TextRequest):
    """Polish: final editing pass. Improve word choice and flow, preserve structure and meaning."""
    polished = await call_llm(POLISH_SYSTEM, req.text, max_tokens=2048, temperature=0.3)
    return PolishResponse(original=req.text, polished=polished)


@app.post("/shorten", response_model=ShortenResponse)
async def shorten_text(req: ShortenRequest):
    """Shorten: produce a shorter version preserving meaning. Target controls aggressiveness."""
    user_msg = f"Target: {req.target}\n\n{req.text}"
    shortened = await call_llm(SHORTEN_SYSTEM, user_msg, max_tokens=2048, temperature=0.3)
    return ShortenResponse(original=req.text, shortened=shortened, target=req.target)


@app.post("/keywords", response_model=KeywordsResponse)
async def keywords_endpoint(req: KeywordsRequest):
    """Extract 5-10 salient keywords as [{term, score}]. Score is 1-10 LLM-assigned salience."""
    user_msg = f"max={req.max}\n\n{req.text}"
    raw = await call_llm(KEYWORDS_SYSTEM, user_msg, max_tokens=1024, temperature=0.2)
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

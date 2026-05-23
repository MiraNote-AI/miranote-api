"""
MiraNote POC -- Voice-to-Text API
Whisper transcription + optional LLM correction (any OpenAI-compatible provider).
"""

import os
import tempfile
import asyncio

import whisper
from fastapi import FastAPI, UploadFile, File, Query
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------- Config ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# ---------- Load models ----------
print(f"Loading Whisper model: {WHISPER_MODEL} ...")
model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded.")

llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None

app = FastAPI(title="MiraNote Voice-to-Text", version="0.1.0")


# The correction prompt is loaded from a separate file to keep source code
# ASCII-only (org Rule 3).  The file ships as a runtime data asset.
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "correction.txt")
CORRECTION_PROMPT: str = ""
if os.path.exists(_PROMPT_PATH):
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        CORRECTION_PROMPT = f.read()


async def correct_with_ai(raw_text: str) -> str:
    """Use the configured LLM to correct Whisper transcription errors, with retry on rate limit."""
    if not llm or not CORRECTION_PROMPT:
        return raw_text
    for attempt in range(3):
        try:
            resp = await asyncio.to_thread(
                llm.chat.completions.create,
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": CORRECTION_PROMPT + "\n\n" + raw_text},
                ],
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 45 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                await asyncio.sleep(wait)
            else:
                print(f"AI correction failed: {e}")
                return raw_text
    return raw_text


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (mp3, wav, flac, m4a, ogg, webm)"),
    correct: bool = Query(True, description="Apply AI correction after Whisper transcription"),
):
    """
    Voice-to-text endpoint.
    - Accepts audio file upload
    - Returns Whisper transcription + optional AI-corrected version
    """
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = await asyncio.to_thread(
            model.transcribe, tmp_path, verbose=False
        )
        raw_text = result["text"]
        language = result.get("language", "unknown")
        segments = [
            {
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]

        corrected_text = None
        if correct and raw_text.strip():
            corrected_text = await correct_with_ai(raw_text)

        return {
            "language": language,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "segments": segments,
        }
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_model": WHISPER_MODEL,
        "llm_model": LLM_MODEL if llm else None,
    }

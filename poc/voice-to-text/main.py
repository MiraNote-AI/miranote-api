"""
MiraNote POC -- Voice-to-Text API
Whisper transcription + optional LLM correction (any OpenAI-compatible provider).
"""

import os
import tempfile
import asyncio
from typing import Any, Dict, Literal, Optional, Tuple

import whisper
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI
from emotion import analyze_emotion

load_dotenv()

# ---------- Config ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# ---------- Lazy model loading ----------
_whisper_model = None


def get_whisper_model():
    """Lazy-load Whisper on first call so import is cheap (tests, /health)."""
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _whisper_model

llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None

app = FastAPI(title="MiraNote Voice-to-Text", version="0.1.0")


# The correction prompt is loaded from a separate file to keep source code
# ASCII-only (org Rule 3).  The file ships as a runtime data asset.
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "correction.txt")
CORRECTION_PROMPT: str = ""
if os.path.exists(_PROMPT_PATH):
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        CORRECTION_PROMPT = f.read()


async def correct_with_ai(raw_text: str) -> Tuple[Optional[str], str]:
    """Use the configured LLM to correct Whisper transcription errors.

    Returns (corrected_text, status) where status is one of:
      "ok"      -- corrected_text is the LLM response
      "skipped" -- no LLM configured; corrected_text is None
      "failed"  -- LLM call errored after retries; corrected_text is None
    """
    if not llm or not CORRECTION_PROMPT:
        return None, "skipped"
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
            return resp.choices[0].message.content, "ok"
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 45 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/3)")
                await asyncio.sleep(wait)
            else:
                print(f"AI correction failed: {e}")
                return None, "failed"
    return None, "failed"


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (mp3, wav, flac, m4a, ogg, webm)"),
    correct: bool = Query(True, description="Apply AI correction after Whisper transcription"),
    lang: Literal["zh", "en"] = Query(
        "zh",
        description=(
            "Audio language. `zh` (default) is the right choice for Chinese, "
            "Chinese + English code-switching, or any audio dominated by Mandarin -- "
            "the multilingual Whisper model handles inline English tokens. "
            "`en` is for pure English audio. We disable Whisper's auto-detect because "
            "on short or noisy clips it misfires (e.g. classifies Mandarin as Javanese)."
        ),
    ),
    with_emotion: bool = Query(
        True,
        description="Run acoustic emotion classifier on the audio (adds ~1 sec).",
    ),
):
    """
    Voice-to-text endpoint.
    - Accepts audio file upload
    - Returns Whisper transcription + optional AI-corrected version
    """
    raw_bytes = await file.read()
    print(f"/transcribe: filename={file.filename!r} size={len(raw_bytes)} bytes")
    # MediaRecorder can emit empty/headers-only blobs if the user stops before
    # any audio chunks arrive. ffmpeg then fails to parse the container and
    # Whisper bubbles a 500. Cheap pre-check turns this into a clean 422.
    if len(raw_bytes) < 1024:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Audio too small ({len(raw_bytes)} bytes). "
                "Recording may be empty or truncated -- try again and record for at least 1 second."
            ),
        )

    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        try:
            result = await asyncio.to_thread(
                get_whisper_model().transcribe, tmp_path, language=lang, verbose=False
            )
        except Exception as e:
            print(f"Whisper/ffmpeg failed on {file.filename!r}: {e}")
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Could not decode audio ({type(e).__name__}). "
                    "Check that the file is a real audio recording in a supported format. See server logs for details."
                ),
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

        corrected_text: Optional[str] = None
        correction_status = "skipped"
        if correct and raw_text.strip():
            corrected_text, correction_status = await correct_with_ai(raw_text)

        emotion_result: Optional[Dict[str, Any]] = None
        emotion_status = "skipped"
        if with_emotion:
            try:
                emotion_result = await asyncio.to_thread(analyze_emotion, tmp_path)
                emotion_status = "ok"
            except Exception as e:  # noqa: BLE001 -- surface to caller as status
                print(f"Emotion analysis failed: {e}")
                emotion_status = "failed"

        return {
            "language": language,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "correction_status": correction_status,
            "segments": segments,
            "emotion": emotion_result,
            "emotion_status": emotion_status,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/emotion")
async def emotion_endpoint(
    file: UploadFile = File(..., description="Audio file"),
):
    """Run only the acoustic emotion classifier on an uploaded audio file."""
    raw_bytes = await file.read()
    if len(raw_bytes) < 1024:
        raise HTTPException(
            status_code=422,
            detail=f"Audio too small ({len(raw_bytes)} bytes). Record at least 1 second.",
        )
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        return await asyncio.to_thread(analyze_emotion, tmp_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Emotion analysis failed: {e}")
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_model": WHISPER_MODEL,
        "llm_model": LLM_MODEL if llm else None,
    }


# Mount static UI at "/" last so explicit API routes above take precedence.
# Visit http://localhost:8000/ in a browser.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

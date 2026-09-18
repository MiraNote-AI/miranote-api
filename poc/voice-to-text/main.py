"""
MiraNote POC -- Voice-to-Text API
Whisper transcription + optional LLM correction (any OpenAI-compatible provider).
"""

import os
import tempfile
import asyncio
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Dict, Literal, Optional, Tuple

import whisper
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import beta_auth
from openai import OpenAI
from yanyi_client import YanYiError, health as yanyi_health, transcribe as yanyi_transcribe
import emotion
from emotion import analyze_emotion

load_dotenv()

# ---------- Config ----------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")

# Which engine answers /transcribe. "whisper" runs the local model; "yanyi"
# calls the hosted YanYi API. Deliberately a restart-time choice rather than a
# per-request one: the two engines have different operational costs, and the
# point of selecting one is that the other stops being paid for.
TRANSCRIBE_ENGINE = os.getenv("TRANSCRIBE_ENGINE", "whisper").strip().lower()
ENGINES = ("whisper", "yanyi")
if TRANSCRIBE_ENGINE not in ENGINES:
    raise ValueError(
        f"TRANSCRIBE_ENGINE must be one of {ENGINES}, got {TRANSCRIBE_ENGINE!r}. "
        "Refusing to start rather than quietly serving the other engine."
    )
YANYI_API_URL = os.getenv("YANYI_API_URL", "https://shujv.synology.me/v1/transcribe")
YANYI_API_KEY = os.getenv("YANYI_API_KEY", "")
YANYI_MODE = os.getenv("YANYI_MODE", "medium")
# The iOS client gives up at 110s and Cloudflare's free plan at 125s. A YanYi
# timeout above either fires after the caller has stopped listening, so this
# sits below both. Measured round trips are 1.4s to 6.4s, so it is generous.
YANYI_TIMEOUT = float(os.getenv("YANYI_TIMEOUT", "90"))
if TRANSCRIBE_ENGINE == "yanyi" and not YANYI_API_KEY:
    raise ValueError(
        "TRANSCRIBE_ENGINE=yanyi needs YANYI_API_KEY. Without it every "
        "transcription would fail with 401 one caller at a time."
    )

# What the startup probe found, reported by /health next to the LLM status.
YANYI_UNCONFIGURED = "unconfigured"   # a different engine is selected
YANYI_OK = "ok"                       # YanYi answered its health check
YANYI_UNREACHABLE = "unreachable"     # it did not
_yanyi_status = YANYI_UNCONFIGURED

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# ---------- Lazy model loading ----------
_whisper_model = None
_whisper_lock = Lock()


def get_whisper_model():
    """Lazy-load Whisper on first call so import is cheap (tests, /health).

    Double-checked locking mirrors emotion._get_pipeline(): under concurrent
    asyncio.to_thread() calls at startup, only one thread loads the model
    instead of several racing to load ~2 GB each.
    """
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                print(f"Loading Whisper model: {WHISPER_MODEL} ...")
                _whisper_model = whisper.load_model(WHISPER_MODEL)
                print("Whisper model loaded.")
    return _whisper_model


def _mean_logprob(result: Dict[str, Any]) -> float:
    """Whisper's own confidence for a decode: the mean of the segments'
    avg_logprob, worst possible when a decode produced no segments."""
    segments = result.get("segments") or []
    scores = [seg["avg_logprob"] for seg in segments if "avg_logprob" in seg]
    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


def _transcribe_picking_language(tmp_path: str) -> Dict[str, Any]:
    """Closed-set selection behind `lang=auto`: decode the clip as zh and
    as en, keep whichever Whisper itself scored higher. Open-set detection
    stays off (it misfires on short clips -- the original reason it was
    disabled); a two-way choice cannot wander off to a third language."""
    model = get_whisper_model()
    candidates = [
        model.transcribe(tmp_path, language=candidate, verbose=False)
        for candidate in ("zh", "en")
    ]
    return max(candidates, key=_mean_logprob)


def drop_no_speech_segments(result: Dict[str, Any]) -> Dict[str, Any]:
    """Silence reaches Whisper as hallucinated stock phrases ("Thanks for
    watching!", subtitle credits). Drop segments Whisper itself distrusts
    -- the reference heuristic: likely non-speech AND low decode
    confidence -- and rebuild `text` from what remains. An all-dropped
    decode yields empty text: the honest answer for a silent clip."""
    segments = result.get("segments") or []
    kept = [
        seg for seg in segments
        if not (
            seg.get("no_speech_prob", 0.0) > 0.6
            and seg.get("avg_logprob", 0.0) < -1.0
        )
    ]
    if len(kept) == len(segments):
        return result
    filtered = dict(result)
    filtered["segments"] = kept
    filtered["text"] = "".join(seg.get("text", "") for seg in kept).strip()
    return filtered


llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None

# What the startup probe found. Reported by /health so a misconfigured provider
# is visible without reading the log -- the failure mode that hid #73 for weeks
# was that a rejected key looked exactly like a working one from outside.
LLM_UNCONFIGURED = "unconfigured"   # no LLM_API_KEY; correction deliberately off
LLM_OK = "ok"                       # the provider answered
LLM_UNREACHABLE = "unreachable"     # configured, and it did not
_llm_status = LLM_UNCONFIGURED


def _redact(text: str) -> str:
    """Never let the key out, whatever the provider echoed back at us."""
    if LLM_API_KEY:
        text = text.replace(LLM_API_KEY, "<LLM_API_KEY>")
    return text


def _check_llm() -> None:
    """Ask the provider one cheap question so a misconfiguration surfaces now.

    A real round-trip rather than a key-shape check: the shapes are only a
    heuristic, and what matters is whether this key works against this base URL
    for this model -- the exact triple that was wrong in #73.

    Never raises. A broken corrector is a degraded service, not a dead one:
    raw Whisper output is still worth serving, so this reports and returns.
    """
    global _llm_status
    if llm is None:
        _llm_status = LLM_UNCONFIGURED
        print("[voice] LLM correction is off: no LLM_API_KEY set. "
              "/transcribe will return raw Whisper output.")
        return
    try:
        llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as error:
        _llm_status = LLM_UNREACHABLE
        print("[voice] LLM CORRECTION IS NOT WORKING. Every transcript will be "
              "raw Whisper output, and /transcribe will still answer 200.")
        print(f"[voice]   model={LLM_MODEL} base_url={LLM_BASE_URL}")
        print(f"[voice]   provider said: {_redact(str(error))[:300]}")
        print("[voice]   LLM_API_KEY, LLM_BASE_URL and LLM_MODEL must all "
              "belong to the same provider.")
        return
    _llm_status = LLM_OK
    print(f"[voice] LLM correction ready: {LLM_MODEL} at {LLM_BASE_URL}")


def _check_yanyi() -> None:
    """Ask YanYi whether it is up, so a dead dependency is visible in /health.

    Never raises: a blip at startup must not leave a dead process, and the
    engine may recover by the first real request. What it must not do is stay
    invisible -- that is what let a broken corrector run unnoticed for weeks
    (#73).
    """
    global _yanyi_status
    if TRANSCRIBE_ENGINE != "yanyi":
        _yanyi_status = YANYI_UNCONFIGURED
        return
    try:
        yanyi_health(api_url=YANYI_API_URL)
    except Exception as error:  # noqa: BLE001 -- degraded, not dead
        _yanyi_status = YANYI_UNREACHABLE
        print("[voice] YANYI IS NOT ANSWERING. Every /transcribe call will fail "
              "until it recovers or TRANSCRIBE_ENGINE is set back to whisper.")
        print(f"[voice]   url={YANYI_API_URL} error={str(error)[:200]}")
        return
    _yanyi_status = YANYI_OK
    print(f"[voice] YanYi ready at {YANYI_API_URL} (mode={YANYI_MODE})")


def _preload_models() -> None:
    """Load what this engine needs, then probe the services it depends on.

    Blocking, called from startup. Whisper loads only when it is the selected
    engine. The probes go last so a slow or dead third party cannot delay the
    models the endpoint actually needs.
    """
    # Whisper's weights are the cost TRANSCRIBE_ENGINE=yanyi exists to avoid.
    # Preloading them on that path would leave the memory on the host anyway.
    if TRANSCRIBE_ENGINE == "whisper":
        get_whisper_model()
    emotion.preload()
    _check_yanyi()
    _check_llm()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pay for the model weights before the service accepts any traffic.

    Measured 2026-09-15 through the tunnel: with both loaded inside the first
    request, that request took 63.4s for a 10s clip while every later one took
    6.6s. The app gives up at 110s, so a cold call carrying a longer recording
    could exceed it -- and /health answered "ok" throughout the minute the
    service could not yet serve. Loading here makes healthy mean ready.

    start_backends.sh already allows 300s per service for exactly this.
    """
    await asyncio.to_thread(_preload_models)
    yield


app = FastAPI(title="MiraNote Voice-to-Text", version="0.1.0", lifespan=lifespan)

# Reachable from the public internet through the Cloudflare tunnel, so every
# request needs a beta token. Installed before CORSMiddleware on purpose: the
# most recently added middleware is the outermost, and CORS must stay outside
# the gate to answer a browser preflight, which carries no Authorization
# header.
beta_auth.install(app)

# POC default is permissive so the unified local UI can call across ports.
# Set CORS_ALLOW_ORIGIN (comma-separated) to scope this in any real
# deployment, e.g. CORS_ALLOW_ORIGIN=https://app.miranote.ai.
_CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGIN", "*").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


async def _transcribe_with_yanyi(tmp_path: str, filename: Optional[str]) -> Dict[str, Any]:
    """Call YanYi off the event loop and return its decoded body.

    A YanYi failure is answered with YanYi's own status and a readable reason.
    Collapsing these into a 500 is what makes an exhausted budget look like a
    server bug; the request id is logged so a support question can quote it.
    """
    try:
        return await asyncio.to_thread(
            yanyi_transcribe,
            tmp_path,
            api_url=YANYI_API_URL,
            api_key=YANYI_API_KEY,
            mode=YANYI_MODE,
            timeout=YANYI_TIMEOUT,
        )
    except YanYiError as error:
        print(
            f"[voice] YanYi failed on {filename!r}: "
            f"status={error.status} code={error.code} detail={error.detail}"
        )
        raise HTTPException(status_code=error.status, detail=error.detail)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (mp3, wav, flac, m4a, ogg, webm)"),
    correct: bool = Query(True, description="Apply AI correction after Whisper transcription"),
    lang: Literal["zh", "en", "auto"] = Query(
        "zh",
        description=(
            "Audio language. `zh` (default) is the right choice for Chinese, "
            "Chinese + English code-switching, or any audio dominated by Mandarin -- "
            "the multilingual Whisper model handles inline English tokens. "
            "`en` is for pure English audio. `auto` decodes as both and keeps "
            "whichever Whisper scored higher -- use it when the speaker could be "
            "in either language (dictation). Whisper's own open-set auto-detect "
            "stays disabled: on short or noisy clips it misfires (e.g. classifies "
            "Mandarin as Javanese); a two-way choice cannot wander like that."
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
    - Returns the transcript from whichever engine TRANSCRIBE_ENGINE selects,
      named in the `engine` field
    - Whisper adds the detected language, per-segment timings, and an optional
      AI-corrected version; `correct` and `lang` apply to that engine only,
      since YanYi takes neither and returns a finished transcript
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
        corrected_text: Optional[str] = None
        correction_status = "skipped"
        truncated: Optional[bool] = None

        if TRANSCRIBE_ENGINE == "yanyi":
            engine = "yanyi"
            payload = await _transcribe_with_yanyi(tmp_path, file.filename)
            raw_text = payload.get("text") or ""
            truncated = payload.get("truncated")
            # YanYi returns a finished transcript and no timings. It reports
            # no language, and running the LLM corrector on top would trade
            # the whole reason for choosing it -- a measured 1.4s round trip
            # against a corrector whose worst case is about 60s.
            language = "unknown"
            segments: list = []
        else:
            engine = "whisper"
            try:
                if lang == "auto":
                    result = await asyncio.to_thread(_transcribe_picking_language, tmp_path)
                else:
                    result = await asyncio.to_thread(
                        get_whisper_model().transcribe, tmp_path, language=lang, verbose=False
                    )
                result = drop_no_speech_segments(result)
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
            "engine": engine,
            "language": language,
            "raw_text": raw_text,
            "truncated": truncated,
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
    except Exception as e:  # noqa: BLE001 -- log internally, return generic detail
        print(f"/emotion analysis failed: {e}")
        raise HTTPException(
            status_code=502,
            detail="Emotion analysis failed. See server logs.",
        )
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": TRANSCRIBE_ENGINE,
        "yanyi": _yanyi_status,
        "whisper_model": WHISPER_MODEL,
        "llm_model": LLM_MODEL if llm else None,
        "llm": _llm_status,
    }


# Mount static UI at "/" last so explicit API routes above take precedence.
# Visit http://localhost:8005/ in a browser.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

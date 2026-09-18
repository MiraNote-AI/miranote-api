"""Client for the YanYi transcription API.

YanYi is an external, hosted transcription service. It is one of the two
engines behind /transcribe; the other is local Whisper. Which one runs is a
deployment choice (TRANSCRIBE_ENGINE), not a per-request one.

The wire contract, read off the service's own OpenAPI document and confirmed
against a live call on 2026-09-17:

    POST <api_url>?mode=<mode>
    Authorization: Bearer <key>
    Content-Type: audio/m4a
    body: the raw audio bytes (not multipart)

    200 -> {"text": str, "mode": str, "request_id": str, "truncated": bool|None}
    non-200 -> {"detail": "<machine readable code>"}

Only stdlib is used here: neither httpx nor requests is declared in
requirements.txt, and both are present only as transitive dependencies.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")

# YanYi caps a request body at 2 MB. The iOS recorder writes 44.1 kHz AAC and
# caps nothing, so a long note would be refused on size alone. Re-encoding to
# the shape YanYi's own recorder uses keeps roughly eight minutes of speech
# inside that ceiling and normalises the container at the same time.
UPLOAD_SAMPLE_RATE = "16000"
UPLOAD_BITRATE = "32k"

# YanYi's machine-readable failure codes, turned into something a person can
# act on. An unmapped code still reaches the caller verbatim -- better an
# unfamiliar string than a swallowed one.
_REASONS = {
    "invalid_api_key": "YanYi rejected the API key. Check YANYI_API_KEY.",
    "invalid_mode": "YanYi rejected the transcription mode. Check YANYI_MODE.",
    "budget_exhausted": "YanYi monthly budget is exhausted.",
    "request_too_large": "Recording is too large for YanYi (2 MB limit).",
    "unsupported_audio": "YanYi could not read this audio format.",
    "upload_timeout": "Upload to YanYi timed out.",
    "rate_limited": "YanYi is rate limiting this key. Retry shortly.",
    "too_many_inflight": "Too many transcriptions in flight at YanYi. Retry shortly.",
    "upstream_error": "YanYi's transcription model failed.",
    "empty_response": "YanYi returned no content.",
    "upstream_timeout": "YanYi's transcription model timed out.",
}


class YanYiError(Exception):
    """A YanYi call that did not return a transcript.

    `status` is the HTTP status to answer the caller with, and `detail` the
    human-readable reason. `code` keeps the service's own machine-readable
    string for the log, so a support question has something to quote.
    """

    def __init__(self, status: int, detail: str, code: Optional[str] = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.code = code


def encode_for_upload(audio_path: str) -> str:
    """Re-encode a recording to 16 kHz mono 32 kbps m4a in a new temp file.

    The caller owns the returned path and must delete it.
    """
    handle, out_path = tempfile.mkstemp(suffix=".m4a")
    os.close(handle)
    try:
        subprocess.run(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-i", audio_path,
                "-ac", "1", "-ar", UPLOAD_SAMPLE_RATE,
                "-c:a", "aac", "-b:a", UPLOAD_BITRATE,
                out_path,
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as error:
        os.unlink(out_path)
        stderr = (error.stderr or b"").decode("utf-8", "replace")[-300:]
        print(f"[voice] ffmpeg could not re-encode the recording: {stderr}")
        raise YanYiError(
            422,
            "Could not decode the recording. Check that it is a real audio file.",
            code="encode_failed",
        )
    return out_path


def transcribe(
    audio_path: str,
    *,
    api_url: str,
    api_key: str,
    mode: str = "medium",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """POST one recording to YanYi and return its decoded JSON body."""
    upload_path = encode_for_upload(audio_path)
    try:
        with open(upload_path, "rb") as handle:
            body = handle.read()

        request = urllib.request.Request(
            f"{api_url}?mode={mode}",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "audio/m4a",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            try:
                return json.loads(raw.decode("utf-8"))
            except ValueError:
                # An edge or proxy in front of YanYi can answer 200 with an
                # HTML error page. That is a bad gateway, not a bug here.
                raise YanYiError(
                    502,
                    "YanYi returned a body that is not JSON.",
                    code="malformed_response",
                )
        except urllib.error.HTTPError as error:
            raise _from_http_error(error)
        except TimeoutError:
            raise YanYiError(504, "YanYi did not answer in time.", code="timeout")
        except urllib.error.URLError as error:
            raise YanYiError(
                502, f"YanYi is unreachable: {error.reason}", code="network_error"
            )
    finally:
        try:
            os.unlink(upload_path)
        except OSError:
            pass


def _from_http_error(error: urllib.error.HTTPError) -> YanYiError:
    """Turn YanYi's {"detail": "<code>"} body into a readable failure."""
    code = None
    try:
        code = json.loads(error.read().decode("utf-8")).get("detail")
    except Exception:  # noqa: BLE001 -- a body we cannot parse must not mask the status
        pass
    reason = _REASONS.get(code) or f"YanYi failed: HTTP {error.code} {code}"
    return YanYiError(error.code, reason, code=code)


def health(*, api_url: str, timeout: float = 10.0) -> bool:
    """True when YanYi answers its health check.

    YanYi serves /health at the origin while the configured URL points at
    /v1/transcribe, so the path is replaced rather than appended to.
    """
    parts = urllib.parse.urlsplit(api_url)
    probe = urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
    with urllib.request.urlopen(probe, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8")).get("status") == "ok"

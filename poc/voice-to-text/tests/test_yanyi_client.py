"""Unit tests for the YanYi wire client.

The HTTP tests stub urlopen -- the real service cannot be called from a test
suite. The encoder test does not stub ffmpeg: what matters there is the shape
of the file that actually comes out, which a mock cannot tell us.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import urllib.error
import urllib.request

import pytest

import yanyi_client


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(status: int, detail: str) -> urllib.error.HTTPError:
    body = json.dumps({"detail": detail}).encode("utf-8")
    return urllib.error.HTTPError(
        "https://yanyi.example/v1/transcribe", status, detail, {}, io.BytesIO(body)
    )


@pytest.fixture
def clip(tmp_path):
    """A real, tiny m4a so the client has something decodable to send."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "clip.m4a"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "128k",
            str(path),
        ],
        check=True,
    )
    return str(path)


def _call(monkeypatch, clip, responder):
    sent = {}

    def fake_urlopen(request, timeout=None):
        sent["url"] = request.full_url
        sent["headers"] = dict(request.headers)
        sent["body"] = request.data
        sent["timeout"] = timeout
        return responder()

    monkeypatch.setattr(yanyi_client.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_a_transcript_comes_back_decoded(monkeypatch, clip):
    _call(monkeypatch, clip, lambda: _FakeResponse({"text": "hello", "truncated": None}))
    result = yanyi_client.transcribe(
        clip, api_url="https://yanyi.example/v1/transcribe", api_key="k"
    )
    assert result["text"] == "hello"


def test_the_request_carries_the_key_and_the_mode(monkeypatch, clip):
    sent = _call(monkeypatch, clip, lambda: _FakeResponse({"text": "hello"}))
    yanyi_client.transcribe(
        clip, api_url="https://yanyi.example/v1/transcribe", api_key="secret", mode="fast"
    )
    assert sent["url"] == "https://yanyi.example/v1/transcribe?mode=fast"
    # urllib title-cases header names.
    assert sent["headers"]["Authorization"] == "Bearer secret"
    assert sent["headers"]["Content-type"] == "audio/m4a"
    assert isinstance(sent["body"], bytes) and len(sent["body"]) > 0


def test_a_budget_failure_is_readable_and_keeps_its_status(monkeypatch, clip):
    def boom():
        raise _http_error(402, "budget_exhausted")

    _call(monkeypatch, clip, boom)
    with pytest.raises(yanyi_client.YanYiError) as caught:
        yanyi_client.transcribe(
            clip, api_url="https://yanyi.example/v1/transcribe", api_key="k"
        )
    assert caught.value.status == 402
    assert caught.value.code == "budget_exhausted"
    assert "budget" in caught.value.detail.lower()


def test_an_unmapped_code_still_reports_status_and_code(monkeypatch, clip):
    def boom():
        raise _http_error(418, "some_new_code")

    _call(monkeypatch, clip, boom)
    with pytest.raises(yanyi_client.YanYiError) as caught:
        yanyi_client.transcribe(
            clip, api_url="https://yanyi.example/v1/transcribe", api_key="k"
        )
    assert caught.value.status == 418
    assert "some_new_code" in caught.value.detail


def test_an_unreachable_service_is_a_bad_gateway(monkeypatch, clip):
    def boom():
        raise urllib.error.URLError("connection refused")

    _call(monkeypatch, clip, boom)
    with pytest.raises(yanyi_client.YanYiError) as caught:
        yanyi_client.transcribe(
            clip, api_url="https://yanyi.example/v1/transcribe", api_key="k"
        )
    assert caught.value.status == 502


def test_a_timeout_is_reported_as_a_timeout(monkeypatch, clip):
    def boom():
        raise TimeoutError("timed out")

    _call(monkeypatch, clip, boom)
    with pytest.raises(yanyi_client.YanYiError) as caught:
        yanyi_client.transcribe(
            clip, api_url="https://yanyi.example/v1/transcribe", api_key="k"
        )
    assert caught.value.status == 504


def test_upload_audio_is_re_encoded_small_enough_to_fit(clip):
    """YanYi rejects a body over 2 MB, and the iOS recorder has no length cap.
    Re-encoding to the shape YanYi's own recorder uses -- 16 kHz mono 32 kbps --
    buys about eight minutes of speech inside that ceiling."""
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")
    encoded = yanyi_client.encode_for_upload(clip)
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json", encoded,
        ],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1

    import os
    # 32 kbps means a 2 MB body holds about 500 seconds of audio.
    assert os.path.getsize(encoded) < os.path.getsize(clip)


def test_undecodable_audio_fails_before_any_upload(monkeypatch, tmp_path):
    """ffmpeg refusing the file means the recording is broken. Saying so beats
    letting YanYi answer 415 about a file we never should have sent."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    junk = tmp_path / "not-audio.m4a"
    junk.write_bytes(b"this is not audio at all" * 100)

    called = []
    monkeypatch.setattr(
        yanyi_client.urllib.request,
        "urlopen",
        lambda *a, **k: called.append(1),
    )
    with pytest.raises(yanyi_client.YanYiError) as caught:
        yanyi_client.transcribe(
            str(junk), api_url="https://yanyi.example/v1/transcribe", api_key="k"
        )
    assert caught.value.status == 422
    assert called == []


def test_health_is_probed_at_the_service_root_not_the_transcribe_path(monkeypatch):
    """YanYi serves /health at the origin, while the configured URL points at
    /v1/transcribe. Probing the configured path would report a 404 as down."""
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url if hasattr(request, "full_url") else request
        return _FakeResponse({"status": "ok"})

    monkeypatch.setattr(yanyi_client.urllib.request, "urlopen", fake_urlopen)
    assert yanyi_client.health(api_url="https://yanyi.example/v1/transcribe") is True
    assert seen["url"] == "https://yanyi.example/health"


def test_a_non_json_body_behind_a_200_is_a_bad_gateway(monkeypatch, clip):
    """An edge or proxy can answer 200 with an HTML error page. Letting that
    surface as a 500 would blame this service for someone else's failure."""
    class _Html:
        def read(self):
            return b"<html>503 Service Unavailable</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    _call(monkeypatch, clip, _Html)
    with pytest.raises(yanyi_client.YanYiError) as caught:
        yanyi_client.transcribe(
            clip, api_url="https://yanyi.example/v1/transcribe", api_key="k"
        )
    assert caught.value.status == 502

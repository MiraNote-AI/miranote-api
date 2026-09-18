"""TRANSCRIBE_ENGINE=yanyi: /transcribe answers from the YanYi API instead of
local Whisper, keeping the response shape every client already decodes."""
from __future__ import annotations
import io

import pytest
from fastapi.testclient import TestClient

from tests.conftest import load_voice_main


def _fake_audio_bytes(size: int = 4096) -> bytes:
    # Opaque bytes; the transcription engine is stubbed so format does not matter.
    return b"\x00\x01" * (size // 2)


def _files():
    return {"file": ("clip.m4a", io.BytesIO(_fake_audio_bytes()), "audio/m4a")}


class _YanYiStub:
    """Stands in for the network call to YanYi, recording what it was asked."""

    def __init__(self, payload=None, error=None):
        self.payload = payload or {
            "text": "ni hao from yanyi",
            "truncated": None,
            "request_id": "req-1",
        }
        self.error = error
        self.calls = []

    def __call__(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        if self.error:
            raise self.error
        return self.payload


@pytest.fixture
def yanyi_app(stub_whisper, stub_emotion, monkeypatch):
    """App loaded with the YanYi engine selected and the network call stubbed."""
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setenv("WHISPER_MODEL", "tiny")
    monkeypatch.setenv("TRANSCRIBE_ENGINE", "yanyi")
    monkeypatch.setenv("YANYI_API_URL", "https://yanyi.example/v1/transcribe")
    monkeypatch.setenv("YANYI_API_KEY", "test-yanyi-key")

    main = load_voice_main()
    stub = _YanYiStub()
    monkeypatch.setattr(main, "yanyi_transcribe", stub, raising=False)
    monkeypatch.setattr(main, "get_whisper_model", lambda: stub_whisper)
    monkeypatch.setattr(main, "analyze_emotion", stub_emotion, raising=False)

    from fastapi.testclient import TestClient
    client = TestClient(main.app, headers={"Authorization": "Bearer test-token"})
    return client, main, stub


def test_yanyi_text_becomes_raw_text(yanyi_app):
    test_client, _, stub = yanyi_app
    r = test_client.post("/transcribe", files=_files(), params={"with_emotion": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["raw_text"] == "ni hao from yanyi"
    assert body["engine"] == "yanyi"
    assert len(stub.calls) == 1


def test_whisper_stays_the_default_engine(monkeypatch):
    """An unset TRANSCRIBE_ENGINE must not change what existing deployments do.

    load_dotenv is neutralised so this reads the code's own default rather than
    whichever engine the developer has selected in their local .env.
    """
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("TRANSCRIBE_ENGINE", raising=False)
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    assert load_voice_main().TRANSCRIBE_ENGINE == "whisper"


def test_the_whisper_path_reports_itself_as_the_engine(voice_client):
    test_client, _, _ = voice_client
    r = test_client.post(
        "/transcribe", files=_files(), params={"correct": "false", "with_emotion": "false"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "whisper"
    assert body["raw_text"] == "hello world"


def test_yanyi_never_calls_the_llm_corrector(yanyi_app, monkeypatch):
    """Correction costs up to ~60s and would erase the reason to use YanYi."""
    test_client, main, _ = yanyi_app
    calls = []

    async def spy(text):
        calls.append(text)
        return "corrected", "ok"

    monkeypatch.setattr(main, "correct_with_ai", spy)

    r = test_client.post(
        "/transcribe", files=_files(), params={"correct": "true", "with_emotion": "false"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == []
    assert body["corrected_text"] is None
    assert body["correction_status"] == "skipped"


def test_yanyi_truncation_flag_reaches_the_caller(yanyi_app):
    """YanYi reports a clipped recording; a caller that cannot see it would
    silently keep a partial note."""
    test_client, main, stub = yanyi_app
    stub.payload = {"text": "partial", "truncated": True, "request_id": "req-2"}
    r = test_client.post("/transcribe", files=_files(), params={"with_emotion": "false"})
    assert r.status_code == 200, r.text
    assert r.json()["truncated"] is True


def test_yanyi_budget_exhaustion_is_reported_as_itself(yanyi_app):
    """The one failure a caller can act on: the account is out of budget.
    A generic 500 here would send someone debugging the wrong thing."""
    from yanyi_client import YanYiError

    test_client, _, stub = yanyi_app
    stub.error = YanYiError(402, "YanYi budget exhausted.", code="budget_exhausted")
    r = test_client.post("/transcribe", files=_files(), params={"with_emotion": "false"})
    assert r.status_code == 402
    assert "budget" in r.json()["detail"].lower()


def test_yanyi_being_unreachable_is_a_bad_gateway(yanyi_app):
    from yanyi_client import YanYiError

    test_client, _, stub = yanyi_app
    stub.error = YanYiError(502, "YanYi is unreachable.", code="network_error")
    r = test_client.post("/transcribe", files=_files(), params={"with_emotion": "false"})
    assert r.status_code == 502
    assert "yanyi" in r.json()["detail"].lower()


def test_emotion_still_runs_on_the_yanyi_path(yanyi_app):
    """Emotion is acoustic and local; it does not depend on which engine
    produced the words."""
    test_client, _, _ = yanyi_app
    r = test_client.post("/transcribe", files=_files(), params={"with_emotion": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["emotion"]["label"] == "happy"
    assert body["emotion_status"] == "ok"


def test_an_unknown_engine_name_stops_the_service(monkeypatch):
    """A typo in TRANSCRIBE_ENGINE must not quietly serve the other engine.
    Issue #73 was exactly this shape: a misconfiguration that still answered
    200 and went unnoticed for weeks."""
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.setenv("TRANSCRIBE_ENGINE", "yanyy")
    with pytest.raises(ValueError, match="TRANSCRIBE_ENGINE"):
        load_voice_main()


def test_health_reports_the_engine_in_use(yanyi_app):
    test_client, _, _ = yanyi_app
    body = test_client.get("/health").json()
    assert body["engine"] == "yanyi"


def test_the_yanyi_engine_refuses_to_start_without_a_key(monkeypatch):
    """An empty key means every transcription 401s. Failing at startup beats
    discovering it one tester at a time."""
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.setenv("TRANSCRIBE_ENGINE", "yanyi")
    monkeypatch.setenv("YANYI_API_KEY", "")
    with pytest.raises(ValueError, match="YANYI_API_KEY"):
        load_voice_main()


def test_health_says_yanyi_is_reachable_once_probed(yanyi_app, monkeypatch):
    test_client, main, _ = yanyi_app
    monkeypatch.setattr(main, "yanyi_health", lambda **kwargs: True)
    with TestClient(main.app, headers={"Authorization": "Bearer test-token"}) as started:
        assert started.get("/health").json()["yanyi"] == "ok"


def test_health_says_yanyi_is_unreachable_without_killing_the_service(
    yanyi_app, monkeypatch
):
    """A transient network blip at startup must not leave a dead process, but
    it must be visible -- the failure mode that hid #73 was invisibility."""
    test_client, main, _ = yanyi_app

    def boom(**kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(main, "yanyi_health", boom)
    with TestClient(main.app, headers={"Authorization": "Bearer test-token"}) as started:
        body = started.get("/health").json()
    assert body["status"] == "ok"
    assert body["yanyi"] == "unreachable"


def test_the_upload_timeout_leaves_the_ios_client_room(yanyi_app):
    """The iOS client gives up at 110s and Cloudflare's edge at 125s. A YanYi
    timeout above those fires after the caller has already stopped listening."""
    test_client, _, stub = yanyi_app
    test_client.post("/transcribe", files=_files(), params={"with_emotion": "false"})
    _, kwargs = stub.calls[0]
    assert kwargs["timeout"] < 110

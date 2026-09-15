"""A misconfigured correction provider is visible without reading the log.

The beta ran with a DeepSeek-shaped key against the Gemini base URL until
2026-09-15. Every correction failed, /transcribe still answered 200 with
correction_status "failed", and the app does `correctedText ?? rawText`
without decoding that field -- so no tester and no operator could tell that no
transcript had ever been corrected (#73).

Startup now probes the provider once and records the result where someone can
see it. It deliberately does NOT refuse to start: raw Whisper output is still
worth serving, and taking voice offline entirely because its cleanup step is
misconfigured trades a degraded feature for no feature.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REAL_KEY = "sk-0123456789abcdef0123456789abcdef"


def _load_main(monkeypatch, *, api_key, probe):
    """main.py imported fresh with the heavy loaders and the LLM stubbed.

    `probe` is called in place of the provider round-trip: return to pass,
    raise to fail.
    """
    os.environ["BETA_TOKENS"] = "test-token"
    os.environ["WHISPER_MODEL"] = "tiny"
    if api_key is None:
        os.environ.pop("LLM_API_KEY", None)
    else:
        os.environ["LLM_API_KEY"] = api_key
    os.environ["LLM_BASE_URL"] = "https://api.example.com/v1"
    os.environ["LLM_MODEL"] = "example-model"

    import dotenv
    import whisper
    import emotion as emotion_module

    # main.py calls load_dotenv() at import. Left alone it reads the developer's
    # real poc/voice-to-text/.env and puts a live key back into the environment
    # this fixture just cleared, so "no provider configured" becomes untestable
    # and every case here silently depends on whoever ran it.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

    emotion_module._PIPELINE = None
    monkeypatch.setattr(whisper, "load_model", lambda name: object())
    monkeypatch.setattr(emotion_module, "_build_pipeline", lambda: (lambda path: {}))

    sys.modules.pop("voice_to_text_main", None)
    spec = importlib.util.spec_from_file_location(
        "voice_to_text_main", Path(__file__).parent.parent / "main.py"
    )
    main = importlib.util.module_from_spec(spec)
    sys.modules["voice_to_text_main"] = main
    spec.loader.exec_module(main)

    if main.llm is not None:
        class _Completions:
            def create(self, **kwargs):
                return probe()

        class _Chat:
            completions = _Completions()

        monkeypatch.setattr(main.llm, "chat", _Chat(), raising=False)
    return main


@pytest.fixture
def working(monkeypatch):
    return _load_main(monkeypatch, api_key=REAL_KEY, probe=lambda: "pong")


@pytest.fixture
def broken(monkeypatch):
    def _raise():
        # Shaped after what DeepSeek actually returns for a bad key, observed
        # 2026-09-15: it echoes the key back in the message. A provider that
        # does not echo it would leave redaction untested, and this test is
        # the only thing standing between that echo and the log.
        raise Exception(
            f"Error code: 401 - {{'error': {{'message': 'Authentication Fails, "
            f"Your api key: {REAL_KEY} is invalid', "
            "'type': 'authentication_error'}}"
        )

    return _load_main(monkeypatch, api_key=REAL_KEY, probe=_raise)


@pytest.fixture
def unconfigured(monkeypatch):
    return _load_main(monkeypatch, api_key=None, probe=lambda: "unused")


def test_a_working_provider_is_reported_as_ok(working):
    with TestClient(working.app) as client:
        assert client.get("/health").json()["llm"] == "ok"


def test_a_broken_provider_is_visible_in_health(broken):
    with TestClient(broken.app) as client:
        body = client.get("/health").json()
    assert body["llm"] == "unreachable", (
        "a rejected key still reported as usable; this is the bug #73 was about"
    )


def test_no_provider_is_distinct_from_a_broken_one(unconfigured):
    """'I chose not to configure it' must not read the same as 'it is broken'."""
    with TestClient(unconfigured.app) as client:
        assert client.get("/health").json()["llm"] == "unconfigured"


def test_a_broken_provider_does_not_stop_the_service(broken):
    """Raw transcripts are still worth serving."""
    with TestClient(broken.app) as client:
        assert client.get("/health").status_code == 200


def test_the_failure_is_loud_in_the_log(broken, capsys):
    with TestClient(broken.app):
        pass
    out = capsys.readouterr().out
    assert "Authentication Fails" in out, "the provider's own reason was dropped"
    assert "example-model" in out and "api.example.com" in out, (
        "the log must name the model and base URL that were rejected"
    )


def test_the_api_key_never_reaches_the_log_or_health(broken, capsys):
    with TestClient(broken.app) as client:
        body = client.get("/health").json()
    out = capsys.readouterr().out
    assert "Authentication Fails" in out, (
        "precondition: the provider echoed the key, so redaction has work to do"
    )
    assert REAL_KEY not in out, "the key was printed to the log"
    assert "<LLM_API_KEY>" in out, "the key was dropped rather than redacted"
    assert REAL_KEY not in str(body), "the key was served from /health"


def test_an_unconfigured_provider_makes_no_call(unconfigured, capsys):
    """No key means no round-trip to anywhere at startup."""
    with TestClient(unconfigured.app):
        pass
    assert "unreachable" not in capsys.readouterr().out

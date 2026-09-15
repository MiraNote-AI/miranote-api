"""Both models load before the service accepts traffic.

Measured 2026-09-15 through the tunnel: the first /transcribe after a restart
took 63.4s for a 10s clip while every later call took 6.6s, because Whisper and
the emotion classifier were loaded inside that first request. The app gives up
at 110s, so a cold call carrying a longer recording could exceed it -- and
/health reported healthy the whole time it was not yet able to serve (#74).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def loaded_main(monkeypatch):
    """main.py imported fresh, with both loaders counted rather than run."""
    os.environ["BETA_TOKENS"] = "test-token"
    os.environ.setdefault("LLM_API_KEY", "fake")
    os.environ["WHISPER_MODEL"] = "tiny"

    calls = {"whisper": 0, "emotion": 0}

    import dotenv
    import whisper
    import emotion as emotion_module

    # Startup probes the correction provider (see test_llm_startup_check.py).
    # Without these two lines this fixture reaches the real provider over the
    # network once per test: load_dotenv() restores whatever key the developer
    # has configured, and _check_llm() then spends it. These tests are about
    # model preloading and must not depend on a network or a third party.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    os.environ.pop("LLM_API_KEY", None)

    # emotion caches its pipeline in a module global that outlives a re-import
    # of main.py, so without this the second test in the file sees a warm cache
    # and counts zero loads. main.py's own _whisper_model needs no reset: the
    # fixture re-executes that module from source each time.
    emotion_module._PIPELINE = None

    monkeypatch.setattr(
        whisper, "load_model",
        lambda name: calls.__setitem__("whisper", calls["whisper"] + 1) or object(),
    )
    monkeypatch.setattr(
        emotion_module, "_build_pipeline",
        lambda: calls.__setitem__("emotion", calls["emotion"] + 1) or (lambda path: {}),
        raising=False,
    )

    sys.modules.pop("voice_to_text_main", None)
    spec = importlib.util.spec_from_file_location(
        "voice_to_text_main", Path(__file__).parent.parent / "main.py"
    )
    main = importlib.util.module_from_spec(spec)
    sys.modules["voice_to_text_main"] = main
    spec.loader.exec_module(main)
    return main, calls


def test_importing_the_module_loads_nothing(loaded_main):
    """Import stays cheap: tests and tooling must not pull 2 GB of weights."""
    _, calls = loaded_main
    assert calls == {"whisper": 0, "emotion": 0}


def test_startup_loads_both_models(loaded_main):
    main, calls = loaded_main
    with TestClient(main.app):
        pass
    assert calls["whisper"] == 1, "Whisper was not loaded during startup"
    assert calls["emotion"] == 1, "the emotion classifier was not loaded during startup"


def test_health_is_only_answered_once_both_are_loaded(loaded_main):
    """The point of the change: healthy must not mean 'a minute from working'."""
    main, calls = loaded_main
    with TestClient(main.app) as client:
        assert calls["whisper"] == 1 and calls["emotion"] == 1
        assert client.get("/health").status_code == 200


def test_a_request_after_startup_does_not_load_again(loaded_main):
    """Startup replaces the lazy path, it does not add a second one."""
    main, calls = loaded_main
    with TestClient(main.app):
        main.get_whisper_model()
        main.get_whisper_model()
    assert calls["whisper"] == 1

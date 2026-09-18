from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


class _WhisperStub:
    def __init__(self):
        self.result: Dict[str, Any] = {
            "text": "hello world",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.5, "text": "hello world"}],
        }

    def transcribe(self, path, **kwargs):
        return self.result


class _EmotionStub:
    def __init__(self):
        self.result: Dict[str, Any] = {
            "label": "happy",
            "confidence": 0.83,
            "all_scores": [
                {"label": "happy", "score": 0.83},
                {"label": "neutral", "score": 0.10},
                {"label": "sad", "score": 0.05},
                {"label": "angry", "score": 0.02},
            ],
        }

    def __call__(self, path):
        return self.result


def load_voice_main():
    """Import main.py fresh under a stable module name.

    A fresh load is what lets a test set environment variables first and have
    the module read them at import time, the way the real process does.
    """
    main_path = Path(__file__).parent.parent / "main.py"

    # Clear any cached import so monkeypatching takes effect on fresh load
    if "voice_to_text_main" in sys.modules:
        del sys.modules["voice_to_text_main"]

    spec = importlib.util.spec_from_file_location("voice_to_text_main", main_path)
    main = importlib.util.module_from_spec(spec)
    sys.modules["voice_to_text_main"] = main
    spec.loader.exec_module(main)
    return main


@pytest.fixture
def stub_whisper():
    return _WhisperStub()


@pytest.fixture
def stub_emotion():
    return _EmotionStub()


@pytest.fixture
def voice_client(stub_whisper, stub_emotion, monkeypatch):
    """FastAPI TestClient with Whisper and emotion both stubbed."""
    # The services require a beta token on every request except /health, so the
    # client carries one. Without it these tests would exercise the gate rather
    # than the endpoint they are about. The gate's own behaviour, including
    # rejection, is covered in tests/test_beta_auth.py and test_beta_gate.py.
    os.environ["BETA_TOKENS"] = "test-token"
    os.environ.setdefault("LLM_API_KEY", "fake")
    os.environ.setdefault("WHISPER_MODEL", "tiny")
    # Pinned, not inherited. main.py calls load_dotenv(), so without this a
    # deployment .env that selects the YanYi engine silently redirects every
    # test in this file to a different code path.
    monkeypatch.setenv("TRANSCRIBE_ENGINE", "whisper")

    main = load_voice_main()

    # Inject Whisper stub by replacing get_whisper_model
    monkeypatch.setattr(main, "get_whisper_model", lambda: stub_whisper)

    # Inject emotion stub if emotion module exists (later tasks will create it)
    emotion_path = Path(__file__).parent.parent / "emotion.py"
    if emotion_path.exists():
        # Replace analyze_emotion symbol that main.py imports
        monkeypatch.setattr(main, "analyze_emotion", stub_emotion, raising=False)

    from fastapi.testclient import TestClient
    client = TestClient(
        main.app, headers={"Authorization": "Bearer test-token"}
    )
    return client, stub_whisper, stub_emotion

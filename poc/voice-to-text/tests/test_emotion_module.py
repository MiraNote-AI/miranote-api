from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_emotion():
    """Fresh import each call so module-level cache `_PIPELINE` is reset between tests."""
    if "voice_to_text_emotion_isolated" in sys.modules:
        del sys.modules["voice_to_text_emotion_isolated"]
    p = Path(__file__).parent.parent / "emotion.py"
    spec = importlib.util.spec_from_file_location("voice_to_text_emotion_isolated", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["voice_to_text_emotion_isolated"] = m
    spec.loader.exec_module(m)
    return m


def test_analyze_emotion_returns_expected_shape(monkeypatch):
    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipe(audio_path, top_k=None):
            return [
                {"label": "happy", "score": 0.83},
                {"label": "neutral", "score": 0.10},
                {"label": "sad", "score": 0.05},
                {"label": "angry", "score": 0.02},
            ]
        return fake_pipe

    em = _load_emotion()
    monkeypatch.setattr(em, "_PIPELINE", None)
    monkeypatch.setattr(em.transformers, "pipeline", fake_pipeline_factory)

    out = em.analyze_emotion("/fake/path.wav")

    assert out["label"] == "happy"
    assert out["confidence"] == 0.83
    assert len(out["all_scores"]) == 4
    assert out["all_scores"][0]["label"] == "happy"
    assert out["all_scores"][-1]["label"] == "angry"


def test_analyze_emotion_sorts_unsorted_input(monkeypatch):
    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipe(audio_path, top_k=None):
            return [
                {"label": "sad", "score": 0.1},
                {"label": "happy", "score": 0.7},
                {"label": "angry", "score": 0.2},
            ]
        return fake_pipe

    em = _load_emotion()
    monkeypatch.setattr(em, "_PIPELINE", None)
    monkeypatch.setattr(em.transformers, "pipeline", fake_pipeline_factory)

    out = em.analyze_emotion("/fake.wav")
    assert out["label"] == "happy"
    assert [s["label"] for s in out["all_scores"]] == ["happy", "angry", "sad"]


def test_analyze_emotion_pipeline_caches(monkeypatch):
    call_count = {"n": 0}

    def fake_pipeline_factory(*args, **kwargs):
        call_count["n"] += 1
        def fake_pipe(p, top_k=None):
            return [{"label": "happy", "score": 1.0}]
        return fake_pipe

    em = _load_emotion()
    monkeypatch.setattr(em, "_PIPELINE", None)
    monkeypatch.setattr(em.transformers, "pipeline", fake_pipeline_factory)

    em.analyze_emotion("/a.wav")
    em.analyze_emotion("/b.wav")
    em.analyze_emotion("/c.wav")
    assert call_count["n"] == 1, "pipeline factory should only run once (cached)"

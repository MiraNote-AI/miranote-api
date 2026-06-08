"""Acoustic speech emotion classifier wrapping a HuggingFace pipeline.

Default model: hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0
- Backbone: facebook/wav2vec2-large-xlsr-53 (53-language pretrained)
- Fine-tuned on: CREMA + RAVDESS + SAVEE + TESS (English)
- 7 classes: angry, disgust, fear, happy, neutral, sad, surprise
- Cross-lingual claim (Chinese, French) is empirical from the model author,
  not a benchmark; treat confidence as advisory on non-English audio.

Model lazily loads on first analyze_emotion() call. ~1.3 GB download to
~/.cache/huggingface/ on first use. Subsequent calls reuse the cached
in-memory pipeline.
"""
from __future__ import annotations

import os
from threading import Lock
from typing import Any, Dict

import transformers

_PIPELINE = None
_LOCK = Lock()
_MODEL = os.getenv(
    "EMOTION_MODEL",
    "hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0",
)


def _get_pipeline():
    """Lazy-load and cache the HuggingFace audio-classification pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                _PIPELINE = transformers.pipeline("audio-classification", model=_MODEL)
    return _PIPELINE


def analyze_emotion(audio_path: str) -> Dict[str, Any]:
    """Classify the speaker's emotion from an audio file.

    Returns:
        {
            "label": "<top emotion>",
            "confidence": <float, 0-1, the top score rounded to 3 dp>,
            "all_scores": [{"label": "...", "score": <float>}, ...] sorted descending
        }
    """
    pipe = _get_pipeline()
    raw = pipe(audio_path, top_k=None)
    raw_sorted = sorted(raw, key=lambda r: r["score"], reverse=True)
    return {
        "label": raw_sorted[0]["label"],
        "confidence": round(float(raw_sorted[0]["score"]), 3),
        "all_scores": [
            {"label": r["label"], "score": round(float(r["score"]), 3)}
            for r in raw_sorted
        ],
    }

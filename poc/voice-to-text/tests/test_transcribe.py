from __future__ import annotations
import io


def _fake_audio_bytes(size: int = 4096) -> bytes:
    # Just opaque bytes; Whisper is stubbed so format doesn't matter.
    return b"\x00\x01" * (size // 2)


def test_transcribe_with_emotion_default_true(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post("/transcribe", files=files, params={"correct": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["raw_text"] == "hello world"
    assert "emotion" in body
    assert body["emotion"]["label"] == "happy"
    assert body["emotion"]["confidence"] == 0.83
    assert body.get("emotion_status") == "ok"


def test_transcribe_with_emotion_false_omits_emotion(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post(
        "/transcribe",
        files=files,
        params={"correct": "false", "with_emotion": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("emotion") is None
    assert body.get("emotion_status") in (None, "skipped")


def test_transcribe_emotion_failure_returns_status(voice_client, monkeypatch):
    test_client, _, _ = voice_client

    def explode(path):
        raise RuntimeError("emotion model failed")

    import voice_to_text_main as main_mod
    monkeypatch.setattr(main_mod, "analyze_emotion", explode, raising=False)

    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post("/transcribe", files=files, params={"correct": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["raw_text"] == "hello world"
    assert body.get("emotion") is None
    assert body.get("emotion_status") == "failed"


class _TwoLanguageStub:
    """Whisper double whose confidence differs by requested language."""

    def __init__(self, better: str):
        self.better = better

    def transcribe(self, path, language=None, **kwargs):
        score = -0.2 if language == self.better else -1.4
        return {
            "text": f"{language} transcript",
            "language": language,
            "segments": [
                {"start": 0.0, "end": 1.0, "text": f"{language} transcript", "avg_logprob": score}
            ],
        }


def _post_auto(test_client):
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    return test_client.post(
        "/transcribe",
        files=files,
        params={"correct": "false", "with_emotion": "false", "lang": "auto"},
    )


def test_auto_keeps_the_decode_whisper_scored_higher(voice_client, monkeypatch):
    test_client, _, _ = voice_client
    import voice_to_text_main as main_mod

    monkeypatch.setattr(main_mod, "get_whisper_model", lambda: _TwoLanguageStub(better="zh"))
    body = _post_auto(test_client).json()
    assert body["language"] == "zh"
    assert body["raw_text"] == "zh transcript"

    monkeypatch.setattr(main_mod, "get_whisper_model", lambda: _TwoLanguageStub(better="en"))
    body = _post_auto(test_client).json()
    assert body["language"] == "en"
    assert body["raw_text"] == "en transcript"


def test_auto_survives_a_decode_with_no_segments(voice_client, monkeypatch):
    test_client, _, _ = voice_client
    import voice_to_text_main as main_mod

    class _OneSidedStub:
        def transcribe(self, path, language=None, **kwargs):
            if language == "zh":
                return {"text": "", "language": "zh", "segments": []}
            return {
                "text": "steady words",
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "steady words", "avg_logprob": -0.9}
                ],
            }

    monkeypatch.setattr(main_mod, "get_whisper_model", lambda: _OneSidedStub())
    body = _post_auto(test_client).json()
    assert body["language"] == "en"
    assert body["raw_text"] == "steady words"


def test_explicit_languages_still_decode_once(voice_client):
    test_client, stub_whisper, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post(
        "/transcribe",
        files=files,
        params={"correct": "false", "with_emotion": "false", "lang": "en"},
    )
    assert r.status_code == 200
    assert r.json()["raw_text"] == "hello world"


def test_silence_hallucination_is_dropped(voice_client):
    # A silent clip decodes to Whisper's stock hallucination with high
    # no_speech_prob and rotten confidence: the endpoint returns no words.
    test_client, stub_whisper, _ = voice_client
    stub_whisper.result = {
        "text": " Thanks for watching!",
        "language": "en",
        "segments": [
            {
                "start": 0.0, "end": 2.0, "text": " Thanks for watching!",
                "no_speech_prob": 0.92, "avg_logprob": -1.6,
            }
        ],
    }
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post(
        "/transcribe",
        files=files,
        params={"correct": "false", "with_emotion": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["raw_text"] == ""
    assert body["segments"] == []


def test_mixed_decode_keeps_confident_segments(voice_client):
    test_client, stub_whisper, _ = voice_client
    stub_whisper.result = {
        "text": " real words Thanks for watching!",
        "language": "en",
        "segments": [
            {
                "start": 0.0, "end": 1.2, "text": " real words",
                "no_speech_prob": 0.05, "avg_logprob": -0.4,
            },
            {
                "start": 1.2, "end": 3.0, "text": " Thanks for watching!",
                "no_speech_prob": 0.88, "avg_logprob": -1.4,
            },
        ],
    }
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post(
        "/transcribe",
        files=files,
        params={"correct": "false", "with_emotion": "false"},
    )
    body = r.json()
    assert body["raw_text"] == "real words"
    assert len(body["segments"]) == 1


def test_clean_decode_passes_through_untouched(voice_client):
    # Segments without the probability fields (older stubs, clean speech)
    # are never dropped.
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post(
        "/transcribe",
        files=files,
        params={"correct": "false", "with_emotion": "false"},
    )
    assert r.json()["raw_text"] == "hello world"

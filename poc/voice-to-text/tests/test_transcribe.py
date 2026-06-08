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

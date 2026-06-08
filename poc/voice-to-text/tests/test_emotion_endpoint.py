from __future__ import annotations
import io


def _fake_audio_bytes(size: int = 4096) -> bytes:
    return b"\x00\x01" * (size // 2)


def test_emotion_endpoint_returns_shape(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post("/emotion", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "happy"
    assert body["confidence"] == 0.83
    assert len(body["all_scores"]) == 4


def test_emotion_endpoint_rejects_tiny_file(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(b"\x00"), "audio/wav")}
    r = test_client.post("/emotion", files=files)
    assert r.status_code == 422

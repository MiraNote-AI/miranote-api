from __future__ import annotations
from types import SimpleNamespace

import pytest

from poc.chatbot.text_client import TextClient


@pytest.fixture
def captured_post(monkeypatch):
    """Capture httpx.post calls and let tests script responses."""
    captured = {"url": None, "json": None}
    scripted = {"response": SimpleNamespace(status_code=200, json=lambda: {"ok": True})}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return scripted["response"]

    monkeypatch.setattr("httpx.post", fake_post)
    return captured, scripted


def test_polish_posts_to_polish_url(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.polish("hello world")
    assert captured["url"] == "http://localhost:8001/polish"
    assert captured["json"] == {"text": "hello world"}


def test_polish_includes_context_when_given(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.polish("hi", context="diary entry")
    assert captured["json"] == {"text": "hi", "context": "diary entry"}


def test_shorten_sends_target(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.shorten("long text", target="tweet")
    assert captured["url"].endswith("/shorten")
    assert captured["json"] == {"text": "long text", "target": "tweet"}


def test_keywords_sends_max(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.keywords("x", max_hits=5)
    assert captured["json"] == {"text": "x", "max": 5}


def test_caption_sends_style(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.caption("entry text", style="diary")
    assert captured["json"] == {"text": "entry text", "style": "diary"}


def test_clean_and_expand_share_shape(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.clean("msg")
    assert captured["url"].endswith("/clean")
    c.expand("msg2")
    assert captured["url"].endswith("/expand")


def test_base_url_trailing_slash_stripped(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001/")
    c.polish("hi")
    assert captured["url"] == "http://localhost:8001/polish"


def test_non_200_raises_runtimeerror(captured_post):
    captured, scripted = captured_post
    scripted["response"] = SimpleNamespace(
        status_code=502, text="bad", json=lambda: {"detail": "upstream failed"}
    )
    c = TextClient("http://localhost:8001")
    with pytest.raises(RuntimeError, match="502"):
        c.polish("x")


def test_connection_error_raises_runtimeerror(monkeypatch):
    import httpx
    def boom(url, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr("httpx.post", boom)
    c = TextClient("http://localhost:9999")
    with pytest.raises(RuntimeError, match="unreachable"):
        c.polish("x")

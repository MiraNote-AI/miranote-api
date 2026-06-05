from __future__ import annotations
from types import SimpleNamespace

import pytest

from poc.chatbot.retrieval_client import RetrievalClient


@pytest.fixture
def captured_post(monkeypatch):
    captured = {"url": None, "json": None}
    scripted = {"response": SimpleNamespace(status_code=200, json=lambda: {"matches": []})}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return scripted["response"]

    monkeypatch.setattr("httpx.post", fake_post)
    return captured, scripted


def test_quotes_posts_to_quotes_url(captured_post):
    captured, _ = captured_post
    c = RetrievalClient("http://localhost:8004")
    c.quotes("I feel tired")
    assert captured["url"] == "http://localhost:8004/quotes"
    assert captured["json"]["text"] == "I feel tired"


def test_quotes_default_max_is_3(captured_post):
    captured, _ = captured_post
    RetrievalClient("http://localhost:8004").quotes("x")
    assert captured["json"]["max"] == 3


def test_quotes_passes_lang_when_set(captured_post):
    captured, _ = captured_post
    RetrievalClient("http://localhost:8004").quotes("x", lang="zh")
    assert captured["json"]["lang"] == "zh"


def test_trailing_slash_stripped(captured_post):
    captured, _ = captured_post
    RetrievalClient("http://localhost:8004/").quotes("x")
    assert captured["url"] == "http://localhost:8004/quotes"


def test_non_200_raises_runtimeerror(captured_post):
    captured, scripted = captured_post
    scripted["response"] = SimpleNamespace(
        status_code=502, text="bad", json=lambda: {"detail": "upstream"}
    )
    with pytest.raises(RuntimeError, match="502"):
        RetrievalClient("http://localhost:8004").quotes("x")


def test_connection_error_raises_runtimeerror(monkeypatch):
    import httpx
    def boom(url, json=None, timeout=None):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr("httpx.post", boom)
    with pytest.raises(RuntimeError, match="unreachable"):
        RetrievalClient("http://localhost:9999").quotes("x")

from __future__ import annotations

import httpx
import pytest

from poc.chatbot.image_client import ImageClient


def test_describe_posts_the_image_and_returns_the_description(monkeypatch):
    captured = {}

    def fake_post(url, files=None, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["filename"] = files["file"][0]
        captured["timeout"] = timeout
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"description": "a calm page"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    client = ImageClient("http://localhost:8002")
    assert client.describe(b"jpegbytes", "how does this page look?") == "a calm page"
    assert captured["url"].endswith("/describe")
    assert captured["params"]["prompt"] == "how does this page look?"
    assert captured["filename"] == "page.jpg"


def test_describe_times_out_well_under_the_apps_turn_budget(monkeypatch):
    captured = {}

    def fake_post(url, files=None, params=None, timeout=None):
        captured["timeout"] = timeout
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"description": "ok"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    ImageClient("http://localhost:8002").describe(b"x", "p")
    # The app gives a canvas turn 60s; a stuck look must leave room for
    # the model to still answer from the page map.
    assert captured["timeout"] <= 30


def test_describe_raises_when_the_body_has_no_description(monkeypatch):
    def fake_post(url, files=None, params=None, timeout=None):
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ValueError):
        ImageClient("http://localhost:8002").describe(b"x", "p")


def test_describe_raises_on_a_server_error(monkeypatch):
    def fake_post(url, files=None, params=None, timeout=None):
        request = httpx.Request("POST", url)
        return httpx.Response(502, json={"detail": "down"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        ImageClient("http://localhost:8002").describe(b"x", "p")

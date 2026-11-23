from __future__ import annotations

import uuid

import httpx
import pytest

from poc.chatbot.image_client import ImageClient


def test_describe_posts_the_image_and_returns_the_description(monkeypatch):
    captured = {}

    def fake_post(url, files=None, params=None, headers=None, timeout=None):
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

    def fake_post(url, files=None, params=None, headers=None, timeout=None):
        captured["timeout"] = timeout
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"description": "ok"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    ImageClient("http://localhost:8002").describe(b"x", "p")
    # The app gives a canvas turn 60s; a stuck look must leave room for
    # the model to still answer from the page map.
    assert captured["timeout"] <= 30


def test_describe_raises_when_the_body_has_no_description(monkeypatch):
    def fake_post(url, files=None, params=None, headers=None, timeout=None):
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ValueError):
        ImageClient("http://localhost:8002").describe(b"x", "p")


def test_describe_raises_on_a_server_error(monkeypatch):
    def fake_post(url, files=None, params=None, headers=None, timeout=None):
        request = httpx.Request("POST", url)
        return httpx.Response(502, json={"detail": "down"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        ImageClient("http://localhost:8002").describe(b"x", "p")


# The image service's gate answers 401 to a call without a token (#136):
# canvas mode's look_at_page failed on every call. /chat binds its own
# request's Authorization header to the client for that turn; the shared
# client in main never holds one.

@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake_post(url, files=None, params=None, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"description": "a calm page"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def test_a_client_for_a_caller_sends_that_callers_authorization(sent):
    ImageClient("http://localhost:8002").for_caller("Bearer writer-token").describe(b"x", "p")
    assert sent[0]["headers"]["Authorization"] == "Bearer writer-token"


def test_a_client_for_a_caller_keeps_the_base_url_and_timeout(sent):
    ImageClient("http://localhost:8002/", timeout=12.0).for_caller("Bearer writer-token").describe(b"x", "p")
    assert sent[0]["url"] == "http://localhost:8002/describe"
    assert sent[0]["timeout"] == 12.0


def test_binding_a_caller_leaves_the_shared_client_without_a_token(sent):
    shared = ImageClient("http://localhost:8002")
    shared.for_caller("Bearer writer-token")
    shared.describe(b"x", "p")
    assert "Authorization" not in sent[0]["headers"]


def test_each_describe_sends_its_own_request_id(sent):
    client = ImageClient("http://localhost:8002").for_caller("Bearer writer-token")
    client.describe(b"x", "p")
    client.describe(b"y", "p")
    first, second = (uuid.UUID(c["headers"]["X-Request-ID"]) for c in sent)
    assert first != second

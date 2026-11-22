"""POST /chat's text tools reach the text service as the caller (#136).

The text service sits behind the same ai_auth gate as /chat and answers
401 missing_token to a call without a token. Each test drives the real
/chat endpoint and run_turn with a scripted fake model; the text tools go
to a fake text service with the real gate in front of it, through a fake
accounts service, so nothing leaves the process.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

os.environ.setdefault("BETA_TOKENS", "test-token")

import ai_auth  # noqa: E402
from fake_accounts import FakeAccounts  # noqa: E402
from poc.chatbot import main, text_client  # noqa: E402
from poc.chatbot.text_client import TextClient  # noqa: E402

BETA = {"Authorization": "Bearer test-token"}
TEXT_URL = "http://text.test"

PAGE = {
    "width": 390,
    "height": 844,
    "elements": [{"handle": "t1", "kind": "text", "x": 20, "y": 40, "w": 300, "h": 40, "says": "Beach day"}],
    "image": base64.b64encode(b"jpeg").decode(),
}


def _call(call_id, name, args):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _calls(*tool_calls):
    return SimpleNamespace(content=None, tool_calls=list(tool_calls))


def _says(text):
    return SimpleNamespace(content=text, tool_calls=None)


class ScriptedModel:
    def __init__(self, script):
        self.script = list(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])


def _text_service(reached):
    """The text routes the chat calls, behind the gate with the metered map
    poc/text-clean-expand/main.py installs (/expand and /polish meter
    `expand`; /clean is not metered)."""
    app = FastAPI()
    ai_auth.install(app, metered={"/expand": ai_auth.Metered("expand"), "/polish": ai_auth.Metered("expand")})

    def route(path, field):
        async def handler(request: Request):
            reached.append({"path": path, **{k: request.headers.get(k) for k in ("Authorization", "X-Request-ID")}})
            text = (await request.json())["text"]
            return {"original": text, field: text + " (done)"}
        app.post(path)(handler)

    route("/clean", "cleaned")
    route("/expand", "expanded")
    route("/polish", "polished")
    return app


@pytest.fixture
def svc(monkeypatch):
    import beta_auth

    beta_auth._limiter._hits.clear()
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    accounts = FakeAccounts()
    accounts.install(monkeypatch)

    reached = []
    text_http = TestClient(_text_service(reached))

    def post(url, json=None, headers=None, timeout=None):
        assert url.startswith(TEXT_URL), url
        return text_http.post(url[len(TEXT_URL):], json=json, headers=headers)

    monkeypatch.setattr(text_client.httpx, "post", post)
    monkeypatch.setattr(main.config, "text_client", TextClient(TEXT_URL))

    def chat(script, headers, **body):
        monkeypatch.setattr(main, "client", ScriptedModel(script))
        r = TestClient(main.app).post("/chat", json=dict(message="tidy this", **body), headers=headers)
        assert r.status_code == 200, r.text
        return r.json()

    return chat, reached, accounts


@pytest.mark.parametrize("mode", [{"notes": []}, {"page": PAGE}, {}], ids=["journal", "canvas", "plain"])
def test_a_text_tool_reaches_the_text_route_with_the_callers_header(svc, mode):
    chat, reached, _ = svc
    body = chat([_calls(_call("c1", "clean_text", {"text": "mesy note"})), _says("Cleaned.")], BETA, **mode)
    assert [(r["path"], r["Authorization"]) for r in reached] == [("/clean", BETA["Authorization"])]
    assert json.loads(body["tool_trace"][0]["result_preview"]) == {"original": "mesy note", "cleaned": "mesy note (done)"}


def test_a_signed_in_writers_token_reaches_the_text_route(svc):
    chat, reached, accounts = svc
    headers = accounts.headers()
    chat([_calls(_call("c1", "clean_text", {"text": "mesy note"})), _says("Cleaned.")], headers, notes=[])
    assert [r["Authorization"] for r in reached] == [headers["Authorization"]]


def test_two_tool_calls_in_one_turn_carry_different_request_ids(svc):
    chat, reached, accounts = svc
    headers = accounts.headers()
    chat([
        _calls(_call("c1", "clean_text", {"text": "mesy note"}), _call("c2", "expand_text", {"text": "a note"})),
        _says("Done."),
    ], headers, notes=[])
    ids = [r["X-Request-ID"] for r in reached]
    assert [r["path"] for r in reached] == ["/clean", "/expand"]
    assert len({uuid.UUID(i) for i in ids}) == 2
    # Never the chat turn's own id: the gate reserved `chat` under it.
    assert headers["X-Request-ID"] not in ids


def test_a_writers_expand_and_polish_are_each_metered_as_expand(svc):
    chat, reached, accounts = svc
    headers = accounts.headers()
    chat([
        _calls(_call("c1", "expand_text", {"text": "a note"})),
        _calls(_call("c2", "polish_text", {"text": "a note"})),
        _says("Done."),
    ], headers, notes=[])
    assert accounts.rows[headers["X-Request-ID"]] == {"feature": "chat", "status": "ok"}
    assert accounts.statuses("expand") == ["ok", "ok"]
    assert {accounts.rows[r["X-Request-ID"]]["feature"] for r in reached} == {"expand"}


def test_a_writer_over_the_expand_cap_gets_the_quota_as_a_tool_error(svc):
    chat, reached, accounts = svc
    accounts.caps = {"expand": 0}
    headers = accounts.headers()
    body = chat([_calls(_call("c1", "expand_text", {"text": "a note"})), _says("I could not finish it.")], headers, notes=[])
    assert reached == []
    error = json.loads(body["tool_trace"][0]["result_preview"])["error"]
    assert error.startswith("text service returned 429") and '"quota"' in error
    # The chat turn still answered, so its own unit is charged.
    assert accounts.statuses("chat") == ["ok"]

"""POST /chat's tools reach their services as the caller (#136).

The text and image services sit behind the same ai_auth gate as /chat and
answer 401 missing_token to a call without a token. Each test drives the
real /chat endpoint and run_turn with a scripted fake model; the text tools
and canvas mode's look_at_page go to fake services with the real gate in
front of them, through a fake accounts service, so nothing leaves the
process.
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
from poc.chatbot.image_client import ImageClient  # noqa: E402
from poc.chatbot.text_client import TextClient  # noqa: E402

BETA = {"Authorization": "Bearer test-token"}
TEXT_URL = "http://text.test"
IMAGE_URL = "http://image.test"

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


def _image_service(reached):
    """The route canvas mode's look_at_page calls, behind the gate with the
    metered map poc/image-generation/main.py installs (/generate, /stylize
    and /border's ai_outline mode meter; /describe is not metered)."""
    app = FastAPI()
    ai_auth.install(app, metered={
        "/generate": ai_auth.Metered("image_generate"),
        "/stylize": ai_auth.Metered("image_edit"),
        "/border": ai_auth.Metered("image_edit", applies=lambda r: r.query_params.get("mode") == "ai_outline"),
    })

    @app.post("/describe")
    async def describe(request: Request):
        reached.append({"path": "/describe", **{k: request.headers.get(k) for k in ("Authorization", "X-Request-ID")}})
        return {"description": "a calm page with room at the bottom"}

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
    services = {TEXT_URL: TestClient(_text_service(reached)), IMAGE_URL: TestClient(_image_service(reached))}

    # text_client and image_client both call httpx.post.
    def post(url, json=None, files=None, params=None, headers=None, timeout=None):
        base = next(b for b in services if url.startswith(b))
        return services[base].post(url[len(base):], json=json, files=files, params=params, headers=headers)

    monkeypatch.setattr(text_client.httpx, "post", post)
    monkeypatch.setattr(main.config, "text_client", TextClient(TEXT_URL))
    monkeypatch.setattr(main, "image_client", ImageClient(IMAGE_URL))

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


def _look_then_edit():
    return [
        _calls(_call("c1", "look_at_page", {})),
        _calls(_call("c2", "edit_page", {"changes": [{"id": "t1", "size": 36}]})),
        _says("Made the title bigger."),
    ]


def test_look_at_page_reaches_describe_with_the_callers_header(svc):
    chat, reached, _ = svc
    body = chat(_look_then_edit(), BETA, page=PAGE)
    look = body["tool_trace"][0]
    assert look["name"] == "look_at_page"
    # The description, not {"status": "could not look at the page"}.
    assert look["result_preview"].startswith('{"description": "a calm page with room at the bottom')
    assert [(r["path"], r["Authorization"]) for r in reached] == [("/describe", BETA["Authorization"])]


def test_a_signed_in_writers_look_reaches_describe_unmetered(svc):
    chat, reached, accounts = svc
    headers = accounts.headers()
    chat(_look_then_edit(), headers, page=PAGE)
    assert [r["Authorization"] for r in reached] == [headers["Authorization"]]
    assert uuid.UUID(reached[0]["X-Request-ID"]) != uuid.UUID(headers["X-Request-ID"])
    # /describe is not metered: the turn's own chat unit is the only row.
    assert accounts.rows == {headers["X-Request-ID"]: {"feature": "chat", "status": "ok"}}

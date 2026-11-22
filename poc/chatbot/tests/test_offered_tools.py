"""POST /chat runs only the tools the turn offered the model (#130).

Journal and canvas mode withhold the docs tools, but the dispatcher used
to run any name the model emitted: a set_docs_root moved docs_root for
every later user of the service. Each test drives the real endpoint and
run_turn with a scripted fake model; the image and text clients are
stubs, so nothing leaves the process.
"""

from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BETA_TOKENS", "test-token")

from poc.chatbot import main  # noqa: E402

BETA = {"Authorization": "Bearer test-token"}

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
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])

    def offered(self):
        return {t["function"]["name"] for t in self.requests[0]["tools"]}


class _Looks:
    def describe(self, image, prompt):
        return "a calm page with room at the bottom"


@pytest.fixture
def svc(monkeypatch, tmp_path):
    import beta_auth

    beta_auth._limiter._hits.clear()
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    # Put docs_root back after each test, so a failing run cannot leak
    # a moved root into the rest of the suite.
    monkeypatch.setattr(main.config, "docs_root", main.config.docs_root)
    monkeypatch.setattr(main, "image_client", _Looks())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def chat(script, **body):
        model = ScriptedModel(script)
        monkeypatch.setattr(main, "client", model)
        r = TestClient(main.app).post("/chat", json=dict(message="hello", **body), headers=BETA)
        assert r.status_code == 200, r.text
        return r.json(), model

    return chat, str(elsewhere)


def _unknown(name):
    return json.dumps({"error": "unknown tool: " + name})


@pytest.mark.parametrize("mode", [{"notes": []}, {"page": PAGE}])
def test_set_docs_root_is_refused_where_it_was_not_offered(svc, mode):
    chat, elsewhere = svc
    root = main.config.docs_root
    body, model = chat([
        _calls(_call("c1", "set_docs_root", {"path": elsewhere})),
        _says("Done."),
    ], **mode)
    assert "set_docs_root" not in model.offered()
    assert main.config.docs_root == root
    assert body["tool_trace"] == [
        {"name": "set_docs_root", "args": {"path": elsewhere}, "result_preview": _unknown("set_docs_root")},
    ]
    # The model is told, so it can answer without the tool.
    told = [m["content"] for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert told == [_unknown("set_docs_root")]


@pytest.mark.parametrize("mode, name, args", [
    ({"notes": []}, "read_doc", {"path": "team.md"}),
    ({"page": PAGE}, "list_docs", {"subdir": "."}),
    ({"page": PAGE}, "search_docs", {"query": "MiraNote"}),
    ({}, "create_note", {"title": "T", "body": "B"}),
])
def test_any_tool_the_turn_did_not_offer_is_refused(svc, mode, name, args):
    chat, _ = svc
    body, model = chat([_calls(_call("c1", name, args)), _says("Done.")], **mode)
    assert name not in model.offered()
    assert body["tool_trace"] == [{"name": name, "args": args, "result_preview": _unknown(name)}]


def test_set_docs_root_after_the_canvas_nudge_does_not_run(svc):
    # The model looks, answers in prose, is nudged, calls set_docs_root,
    # then answers KEEP: the first reply stands and docs_root stays put.
    chat, elsewhere = svc
    root = main.config.docs_root
    body, model = chat([
        _calls(_call("c1", "look_at_page", {})),
        _says("It looks calm to me."),
        _calls(_call("c2", "set_docs_root", {"path": elsewhere})),
        _says("KEEP"),
    ], page=PAGE)
    assert body["reply"] == "It looks calm to me."
    assert main.config.docs_root == root
    assert model.script == []


def test_offered_tools_still_run_in_every_mode(svc):
    chat, _ = svc
    body, _ = chat([_calls(_call("c1", "list_docs", {"subdir": "."})), _says("Here.")])
    # A listing of the demo docs (the preview is cut at 300 characters).
    assert body["tool_trace"][0]["result_preview"].startswith('[{"path": ')

    body, _ = chat([
        _calls(_call("c1", "create_note", {"title": "Beach", "body": "Sun and salt."})),
        _says("I drafted a page."),
    ], notes=[])
    assert json.loads(body["tool_trace"][0]["result_preview"]) == {
        "status": "draft handed to the app for the user to shape",
    }

    body, _ = chat([
        _calls(_call("c1", "look_at_page", {})),
        _calls(_call("c2", "edit_page", {"changes": [{"id": "t1", "size": 36}]})),
        _says("Made the title bigger."),
    ], page=PAGE)
    assert body["reply"] == "Made the title bigger."
    look, edit = body["tool_trace"]
    assert look["name"] == "look_at_page"
    assert look["result_preview"].startswith('{"description": "a calm page')
    assert (edit["name"], edit["result_preview"]) == ("edit_page", '{"status": "handed to the app"}')

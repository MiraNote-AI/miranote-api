from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List
import json

import pytest

from poc.chatbot.chat_loop import run_turn, ChatTurnResult
from poc.chatbot.session import SessionStore


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _resp(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeChatCompletions:
    def __init__(self, scripted_responses):
        self._scripted = list(scripted_responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted_responses):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(scripted_responses))


def test_no_tool_call_returns_text(tmp_path):
    client = FakeClient([_resp(_msg(content="hello back"))])
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client,
        session_store=store,
        session_id=None,
        user_message="hi",
        model="fake-model",
        tools=[],
        tool_dispatcher=lambda name, args: {"error": "no tools"},
        max_iterations=6,
        max_history=40,
        system_prompt="you are a helper",
    )
    assert isinstance(result, ChatTurnResult)
    assert result.reply == "hello back"
    assert result.tool_trace == []
    history = store.get(result.session_id)
    assert history[0] == {"role": "system", "content": "you are a helper"}
    assert history[1] == {"role": "user", "content": "hi"}
    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == "hello back"


def _tool_call(call_id, name, args_dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args_dict)),
    )


def test_single_tool_call_then_text():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("call_1", "search_docs", {"query": "apples"})])),
        _resp(_msg(content="found apples in alpha.md")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    captured = []

    def dispatcher(name, args):
        captured.append((name, args))
        return [{"path": "alpha.md", "line": 3, "snippet": "Apples and oranges."}]

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="find apples", model="fake", tools=[{"type": "function", "function": {"name": "search_docs"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "found apples in alpha.md"
    assert captured == [("search_docs", {"query": "apples"})]
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["name"] == "search_docs"
    assert result.tool_trace[0]["args"] == {"query": "apples"}
    history = store.get(result.session_id)
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"


def test_multi_step_tool_use():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "list_docs", {"subdir": "."})])),
        _resp(_msg(tool_calls=[_tool_call("c2", "read_doc", {"path": "alpha.md"})])),
        _resp(_msg(content="alpha says hello")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    def dispatcher(name, args):
        return {"ok": True, "name": name}

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="look around", model="fake", tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "alpha says hello"
    assert [t["name"] for t in result.tool_trace] == ["list_docs", "read_doc"]


def test_dispatcher_errors_are_wrapped_for_model():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "read_doc", {"path": "../escape"})])),
        _resp(_msg(content="oh that was rejected")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    def dispatcher(name, args):
        raise ValueError("path outside DOCS_ROOT")

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="read escape", model="fake", tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "oh that was rejected"
    history = store.get(result.session_id)
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert "error" in tool_msgs[0]["content"]
    assert "outside" in tool_msgs[0]["content"]

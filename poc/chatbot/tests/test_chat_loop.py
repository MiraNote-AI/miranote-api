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


def test_iteration_cap_returns_synthetic_reply():
    # 8 scripted responses, but cap = 3 -> loop should give up at 3.
    scripted = [_resp(_msg(tool_calls=[_tool_call(f"c{i}", "list_docs", {"subdir": "."})])) for i in range(8)]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="loop forever", model="fake",
        tools=[{"type": "function", "function": {"name": "list_docs"}}],
        tool_dispatcher=lambda n, a: [],
        max_iterations=3, max_history=100, system_prompt="sys",
    )
    assert "stopped" in result.reply.lower()
    assert "max_tool_iterations" in result.reply.lower()
    # The fake should have been called exactly 3 times.
    assert len(client.chat.completions.calls) == 3
    assert len(result.tool_trace) == 3


def test_thinking_mode_reasoning_content_preserved():
    # DeepSeek v4-flash returns reasoning_content alongside tool_calls and
    # 400s the follow-up turn if we drop it. The assistant entry written to
    # history must round-trip it.
    msg_with_reasoning = SimpleNamespace(
        content=None,
        tool_calls=[_tool_call("c1", "list_docs", {"subdir": "."})],
        reasoning_content="I should list the docs first.",
    )
    scripted = [_resp(msg_with_reasoning), _resp(_msg(content="all done"))]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="x", model="fake",
        tools=[{"type": "function", "function": {"name": "list_docs"}}],
        tool_dispatcher=lambda n, a: [],
        max_iterations=6, max_history=40, system_prompt="sys",
    )
    history = store.get(result.session_id)
    assistant_with_tools = [m for m in history if m.get("role") == "assistant" and m.get("tool_calls")][0]
    assert assistant_with_tools["reasoning_content"] == "I should list the docs first."


def test_empty_content_retries_then_lands():
    scripted = [
        _resp(_msg(content="")),
        _resp(_msg(content=None)),
        _resp(_msg(content="here at last")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client,
        session_store=store,
        session_id=None,
        user_message="hi",
        model="fake-model",
        tools=[],
        tool_dispatcher=lambda name, args: {},
        max_iterations=6,
        max_history=40,
        system_prompt="helper",
    )
    assert result.reply == "here at last"
    history = store.get(result.session_id)
    assert [m["role"] for m in history] == ["system", "user", "assistant"], "empty tries append nothing"


def test_drop_unrenderable_strips_pua_and_replacement_chars():
    from poc.chatbot.chat_loop import drop_unrenderable

    dirty = "Hi! " + chr(0xE5D4) + chr(0xFFFD) + chr(0x0007) + "there"
    assert drop_unrenderable(dirty) == "Hi! there"


def test_drop_unrenderable_keeps_emoji_cjk_and_newlines():
    from poc.chatbot.chat_loop import drop_unrenderable

    # wave emoji + sun-with-VS16 + Chinese + newline all render fine on
    # clients and must pass through untouched
    clean = (
        "Hi " + chr(0x1F44B) + "\n"
        + chr(0x2600) + chr(0xFE0F) + " "
        + chr(0x4F60) + chr(0x597D)
    )
    assert drop_unrenderable(clean) == clean

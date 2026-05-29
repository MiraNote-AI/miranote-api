from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List

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

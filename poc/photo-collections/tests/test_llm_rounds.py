"""llm.complete: the repeat-theme round call kinds (fake SDK client, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import config
import llm

MSGS = [{"role": "user", "content": "x"}]


class _Completions:
    def __init__(self, usage=None):
        self.kwargs = None
        self.usage = usage

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="{}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self.usage)


@pytest.fixture
def comp(monkeypatch):
    completions = _Completions(SimpleNamespace(prompt_tokens=1978, completion_tokens=248))
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm, "_client", fake)
    yield completions
    monkeypatch.setattr(llm, "_client", None)


@pytest.mark.parametrize("kind", ["r1", "r2", "rs"])
def test_round_calls_disable_thinking_and_cap_tokens(comp, kind):
    llm.complete(MSGS, call=kind, timeout=config.ROUND_TIMEOUT)
    kwargs = comp.kwargs
    assert kwargs["model"] == config.LLM_MODEL
    assert kwargs["max_tokens"] == 1000
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"enable_thinking": False}
    assert kwargs["temperature"] == config.TEMPERATURE_ROUND
    assert "stream" not in kwargs


def test_round_constants():
    assert config.ROUND_MAX_TOKENS == 1000
    assert config.TEMPERATURE_ROUND == 0.2
    assert config.ROUND_TIMEOUT == 45.0


def test_round_reply_carries_usage(comp):
    reply = llm.complete(MSGS, call="r1", timeout=config.ROUND_TIMEOUT)
    assert reply.prompt_tokens == 1978
    assert reply.completion_tokens == 248
    assert reply.cost_cny == pytest.approx(1978 / 1e6 * 0.2 + 248 / 1e6 * 0.8)


def test_unknown_call_kind_still_raises(comp):
    with pytest.raises(ValueError):
        llm.complete(MSGS, call="x", timeout=10)


def test_abc_requests_unchanged(monkeypatch):
    monkeypatch.setattr(config, "CALL_A_MODE", "thinking_stream")
    monkeypatch.setattr(config, "CALL_B_MODE", "json_schema")
    monkeypatch.setattr(config, "CALL_C_MODE", "json_object")
    assert llm._request("a", MSGS, 7.0) == {
        "model": config.LLM_MODEL,
        "messages": MSGS,
        "temperature": 0.2,
        "timeout": 7.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_body": {"enable_thinking": True, "thinking_budget": config.THINKING_BUDGET},
    }
    assert llm._request("b", MSGS, 7.0) == {
        "model": config.CALL_B_MODEL,
        "messages": MSGS,
        "temperature": 0.7,
        "timeout": 7.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "collect_b", "strict": True, "schema": config.SCHEMA_B},
        },
        "extra_body": {"enable_thinking": False},
    }
    assert llm._request("c", MSGS, 7.0) == {
        "model": config.LLM_MODEL,
        "messages": MSGS,
        "temperature": 0.2,
        "timeout": 7.0,
        "response_format": {"type": "json_object"},
        "extra_body": {"enable_thinking": False},
    }

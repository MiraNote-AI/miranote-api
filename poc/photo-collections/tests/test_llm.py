"""llm.complete: the single model seam (fake SDK client, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

import config
import llm


def _chunk(content=None, reasoning=None, usage=None):
    choices = []
    if content is not None or reasoning is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content, reasoning_content=reasoning))]
    return SimpleNamespace(choices=choices, usage=usage)


class FakeCompletions:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result


def _install(monkeypatch, completions):
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    return completions


def test_stream_reply_is_assembled_and_reasoning_dropped(monkeypatch):
    monkeypatch.setattr(config, "CALL_A_MODE", "thinking_stream")
    chunks = [
        _chunk(reasoning="让我想想"),
        _chunk(content='{"a": '),
        _chunk(reasoning="还在想", content="1}"),
        _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20)),
    ]
    comp = _install(monkeypatch, FakeCompletions(result=iter(chunks)))
    reply = llm.complete([{"role": "user", "content": "x"}], call="a", timeout=120)
    assert reply.text == '{"a": 1}'
    assert reply.prompt_tokens == 10 and reply.completion_tokens == 20
    assert comp.kwargs["stream"] is True
    assert comp.kwargs["extra_body"]["enable_thinking"] is True
    assert "response_format" not in comp.kwargs


def test_json_schema_mode_for_call_b(monkeypatch):
    monkeypatch.setattr(config, "CALL_B_MODE", "json_schema")
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    comp = _install(monkeypatch, FakeCompletions(result=resp))
    reply = llm.complete([{"role": "user", "content": "x"}], call="b", timeout=30)
    assert reply.text == "{}"
    assert comp.kwargs["response_format"]["type"] == "json_schema"
    assert comp.kwargs["response_format"]["json_schema"]["strict"] is True
    assert comp.kwargs["extra_body"]["enable_thinking"] is False


def test_usage_becomes_cost(monkeypatch):
    monkeypatch.setattr(config, "CALL_B_MODE", "json_object")
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=1_000_000),
    )
    _install(monkeypatch, FakeCompletions(result=resp))
    reply = llm.complete([{"role": "user", "content": "x"}], call="b", timeout=30)
    assert reply.cost_cny == pytest.approx(1.0)


def test_sdk_timeout_maps_to_model_timeout(monkeypatch):
    error = openai.APITimeoutError(request=httpx.Request("POST", "http://example.invalid"))
    _install(monkeypatch, FakeCompletions(error=error))
    with pytest.raises(llm.ModelTimeout):
        llm.complete([{"role": "user", "content": "x"}], call="a", timeout=120)


def test_sdk_error_maps_to_model_error(monkeypatch):
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://example.invalid"))
    _install(monkeypatch, FakeCompletions(error=error))
    with pytest.raises(llm.ModelError):
        llm.complete([{"role": "user", "content": "x"}], call="b", timeout=30)


def test_stream_over_deadline_is_timeout(monkeypatch):
    monkeypatch.setattr(config, "CALL_A_MODE", "thinking_stream")
    ticks = iter([0.0, 0.0, 200.0, 200.0, 200.0])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(ticks))
    chunks = [_chunk(content="{"), _chunk(content="}")]
    _install(monkeypatch, FakeCompletions(result=iter(chunks)))
    with pytest.raises(llm.ModelTimeout):
        llm.complete([{"role": "user", "content": "x"}], call="a", timeout=120)


def test_slow_non_stream_reply_is_cut_at_wall_clock_timeout(monkeypatch):
    # Seen on the real model: call B answered after 35 s with timeout=30.
    # The SDK timeout is per network read, so the wall clock is enforced here.
    import time as _time

    class Slow(FakeCompletions):
        def create(self, **kwargs):
            _time.sleep(0.5)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    monkeypatch.setattr(config, "CALL_B_MODE", "json_schema")
    _install(monkeypatch, Slow())
    started = _time.monotonic()
    with pytest.raises(llm.ModelTimeout):
        llm.complete([{"role": "user", "content": "x"}], call="b", timeout=0.1)
    assert _time.monotonic() - started < 0.4

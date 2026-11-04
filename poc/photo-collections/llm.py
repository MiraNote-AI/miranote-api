"""The single seam between this service and the model.

Every model call goes through complete(), so tests replace one function.
The request shape per call (thinking / stream / response_format) comes from
config.CALL_A_MODE and config.CALL_B_MODE, settled by tools/probe_model.py.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import List, Optional

import openai
from openai import OpenAI

import config


class ModelTimeout(Exception):
    """The call did not finish inside its time limit."""


class ModelError(Exception):
    """The provider failed, refused, or answered with nothing usable."""


@dataclass
class LlmReply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_cny: float


_client: Optional[OpenAI] = None
_client_lock = threading.Lock()

# The SDK's timeout bounds each network read, not the whole call: a reply
# trickling in can run well past it (call B took 35 s against 30 on the real
# model). Requests run here so the caller waits on the wall clock instead.
# An abandoned request finishes or fails on its own read timeout.
_calls = ThreadPoolExecutor(max_workers=config.LLM_WORKERS, thread_name_prefix="llm")


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not config.LLM_API_KEY:
                    raise ModelError("LLM_API_KEY is not configured")
                # max_retries=0: the pipeline owns retries (one per call), and
                # the SDK's own retries would silently stretch the time limits.
                _client = OpenAI(
                    api_key=config.LLM_API_KEY,
                    base_url=config.LLM_BASE_URL,
                    max_retries=0,
                )
    return _client


def _redact(text: str) -> str:
    if config.LLM_API_KEY:
        text = text.replace(config.LLM_API_KEY, "<LLM_API_KEY>")
    return text


def _usage_pair(usage) -> tuple:
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _request(call: str, messages: List[dict], timeout: float) -> dict:
    mode = config.CALL_A_MODE if call == "a" else config.CALL_B_MODE
    if mode not in config.MODES:
        raise ModelError("unknown call mode " + repr(mode))
    kwargs = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE_A if call == "a" else config.TEMPERATURE_B,
        "timeout": timeout,
    }
    if mode == "thinking_stream":
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        kwargs["extra_body"] = {
            "enable_thinking": True,
            "thinking_budget": config.THINKING_BUDGET,
        }
    elif mode == "json_object":
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = {"enable_thinking": False}
    else:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "collect_b", "strict": True, "schema": config.SCHEMA_B},
        }
        kwargs["extra_body"] = {"enable_thinking": False}
    return kwargs


def complete(messages: List[dict], *, call: str, timeout: float) -> LlmReply:
    """One model call. Raises ModelTimeout or ModelError, never anything else
    from the SDK. Reasoning text from a thinking model is dropped."""
    if call not in ("a", "b"):
        raise ValueError("call must be 'a' or 'b'")
    if timeout <= 0:
        raise ModelTimeout("no time left for the call")
    kwargs = _request(call, messages, timeout)
    started = time.monotonic()

    def _do() -> tuple:
        client = _get_client()
        result = client.chat.completions.create(**kwargs)
        if not kwargs.get("stream"):
            return result.choices[0].message.content or "", getattr(result, "usage", None)
        parts: List[str] = []
        usage = None
        for chunk in result:
            if time.monotonic() - started > timeout:
                raise ModelTimeout("stream ran past {:.0f}s".format(timeout))
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                content = getattr(choice.delta, "content", None)
                if content:
                    parts.append(content)
        return "".join(parts), usage

    try:
        future = _calls.submit(_do)
        try:
            text, usage = future.result(timeout=timeout)
        except FutureTimeout:
            # Stops a request still queued; one already sent runs on until its
            # own read timeout. Its tokens are never seen, so the call is
            # logged at cost 0 although the provider may bill it.
            future.cancel()
            raise ModelTimeout("no complete reply within {:.0f}s".format(timeout)) from None
    except ModelTimeout:
        raise
    except ModelError:
        raise
    except openai.APITimeoutError as error:
        raise ModelTimeout(_redact(str(error))[:300]) from None
    except Exception as error:  # noqa: BLE001 -- every SDK failure is a ModelError
        raise ModelError(_redact("{}: {}".format(type(error).__name__, error))[:300]) from None
    prompt_tokens, completion_tokens = _usage_pair(usage)
    return LlmReply(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=int((time.monotonic() - started) * 1000),
        cost_cny=config.cost_cny(prompt_tokens, completion_tokens),
    )


LLM_UNCONFIGURED = "unconfigured"
LLM_OK = "ok"
LLM_UNREACHABLE = "unreachable"


def probe() -> str:
    """One tiny real text call at startup, so a wrong key, URL or model name
    shows up in /health instead of in the first user's job. Never raises."""
    if not config.LLM_API_KEY:
        print("[collections] LLM_API_KEY is not set -- every compose job will fail")
        return LLM_UNCONFIGURED
    try:
        _get_client().chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            extra_body={"enable_thinking": False},
            timeout=20,
        )
    except Exception as error:  # noqa: BLE001 -- degraded, not dead
        print("[collections] MODEL PROBE FAILED: model={} base_url={}".format(
            config.LLM_MODEL, config.LLM_BASE_URL))
        print("[collections]   provider said: " + _redact(str(error))[:300])
        return LLM_UNREACHABLE
    print("[collections] model ready: {} at {}".format(config.LLM_MODEL, config.LLM_BASE_URL))
    return LLM_OK

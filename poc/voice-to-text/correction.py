"""One LLM correction call, isolated from the FastAPI app.

`main.py` owns *when* to correct and how to retry; this module owns *what*
a single correction request looks like -- the prompt, the message shape, the
max_tokens, and how to read text and token usage back off the response.

It exists so `bench_correction.py` can run the exact request production runs
against a different client and model, without importing `main.py` and
dragging in whisper + transformers. Keeping the retry loop out of here is
deliberate: that loop is async and backs off with `asyncio.sleep`, which
yields the event loop. A sync version would park a worker thread for 45s
instead.

ASCII only (org Rule 3) -- the prompt itself is Chinese and lives in
prompts/correction.txt as a runtime data asset.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_MAX_TOKENS = 4096

# Was `45 * (attempt + 1)` inline in main.py. Named here so the bench can
# reuse the same waits, and indexed by attempt number: attempt 0 waits 45s,
# attempt 1 waits 90s, attempt 2 is the last try and never waits.
RETRY_BACKOFF_SECONDS = (45, 90)

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "correction.txt")

CORRECTION_PROMPT: str = ""
if os.path.exists(_PROMPT_PATH):
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        CORRECTION_PROMPT = f.read()


@dataclass
class CorrectionResult:
    """Outcome of one attempt. `status` is "ok" or "failed" -- "skipped" is
    main.py's call to make, since only it knows whether an LLM is configured.
    On failure `text` is None and `error` carries the exception text, which
    the caller inspects with is_rate_limited() to decide whether to retry."""

    text: Optional[str]
    status: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


def default_extra_body(model: str) -> Optional[Dict[str, Any]]:
    """Provider options a model needs to do the job it was benchmarked doing.

    Qwen3-series models on DashScope reason by default. Correction is not a
    reasoning task, the whole benchmark ran with thinking off, and thinking
    tokens bill as output -- so leaving it on would put production on a
    different configuration than the one that was measured. Worse, some Qwen3
    models on DashScope reject a non-streaming request outright when thinking
    is enabled, which would take /transcribe down rather than just make it
    slower.

    Matching on a provider prefix inside otherwise provider-agnostic code is
    not elegant. The alternative -- an env var carrying JSON -- puts a
    correctness requirement behind configuration a deployer can forget, which
    is worse.
    """
    return {"enable_thinking": False} if model.lower().startswith("qwen") else None


def build_messages(raw_text: str) -> List[Dict[str, str]]:
    """The request body production has always sent: a single user turn, no
    system role, prompt and transcript separated by a blank line."""
    return [{"role": "user", "content": CORRECTION_PROMPT + "\n\n" + raw_text}]


def is_rate_limited(error: Any) -> bool:
    """Whether an error should be retried after a backoff. Matches the
    original substring test rather than an SDK exception type, because the
    POC talks to several OpenAI-compatible providers that raise differently."""
    return "429" in str(error)


def _usage(resp: Any) -> tuple:
    """Token counts, defensively. Providers behind the OpenAI-compatible
    protocol are inconsistent about `usage` -- some omit it on short
    completions -- and a benchmark must not crash on a missing field."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def correct_once(
    raw_text: str,
    client: Any,
    model: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    extra_body: Optional[Dict[str, Any]] = None,
) -> CorrectionResult:
    """Send one correction request. Never raises -- a failure comes back as a
    result with status "failed", so a benchmark records it as a data point
    and the service can fall through to its retry loop.

    `extra_body` carries provider-specific options the OpenAI SDK passes
    through untouched. The benchmark uses it to send DashScope's
    `enable_thinking: false`, so a Qwen3-series model does the same minimal
    correction the others do instead of reasoning first.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": build_messages(raw_text),
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:  # noqa: BLE001 -- reported to the caller as status
        return CorrectionResult(
            text=None,
            status="failed",
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(e).__name__}: {e}",
        )
    latency_ms = (time.perf_counter() - started) * 1000

    # A truncated or content-filtered response can carry no choices at all.
    # Treat that as a failure rather than an IndexError.
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return CorrectionResult(
            text=None,
            status="failed",
            latency_ms=latency_ms,
            error="no choices in response",
        )

    prompt_tokens, completion_tokens = _usage(resp)
    return CorrectionResult(
        text=choices[0].message.content,
        status="ok",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
    )

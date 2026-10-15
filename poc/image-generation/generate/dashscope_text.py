"""qwen prompt expansion, the Chinese counterpart to the Gemini call in
prompt_expander.

Same contract as the Gemini branch it sits beside: one prompt string in, the
model's text out, or None when the model returned no text (a safety block, an
empty candidate). prompt_expander turns None into the user's own words, so this
module never needs to know about that fallback.

Two endpoints, chosen by model id. The qwen3.x models are not served on the
native text-generation path -- it answers 400 "url error" for them -- only on
the OpenAI-compatible one, which is also how poc/voice-to-text calls
qwen3.5-flash. The older ids (qwen-turbo, qwen-plus) keep the native path.
"""

import config
from generate import http_client

KEY_ENV = "DASHSCOPE_API_KEY"
_PATH = "/services/aigc/text-generation/generation"
_COMPATIBLE_PATH = "/chat/completions"


def _uses_compatible_endpoint(model: str) -> bool:
    return model.startswith("qwen3")


def _compatible_base_url() -> str:
    """The OpenAI-compatible root for the configured region:
    https://dashscope.aliyuncs.com/api/v1 -> .../compatible-mode/v1."""
    return config.DASHSCOPE_BASE_URL.replace("/api/v1", "/compatible-mode/v1")


def complete(prompt: str, model: str) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    if _uses_compatible_endpoint(model):
        # Thinking off: expansion is a rewrite, not a reasoning task, and a
        # thinking qwen3 model rejects a non-streaming request outright.
        payload = {"model": model, "messages": messages, "enable_thinking": False}
        url = _compatible_base_url() + _COMPATIBLE_PATH
    else:
        payload = {
            "model": model,
            "input": {"messages": messages},
            # message format keeps the reply at output.choices[].message.content;
            # the legacy default puts it at output.text instead.
            "parameters": {"result_format": "message"},
        }
        url = config.DASHSCOPE_BASE_URL + _PATH
    data = http_client.post_json(url, payload, KEY_ENV, timeout=60.0)
    return text_of(data)


def text_of(data: dict) -> str | None:
    """The reply text, or None. Split out so it can be unit tested offline.

    Reads both shapes: the native endpoint nests choices under "output", the
    OpenAI-compatible one puts them at the top level."""
    output = data.get("output") or {}
    choices = output.get("choices") or data.get("choices") or []
    if choices:
        content = (choices[0].get("message") or {}).get("content")
        # Some qwen models answer with a list of parts rather than a string.
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
        return content or None
    return output.get("text") or None

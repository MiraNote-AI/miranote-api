"""qwen-turbo prompt expansion, the Chinese counterpart to the Gemini call in
prompt_expander.

Same contract as the Gemini branch it sits beside: one prompt string in, the
model's text out, or None when the model returned no text (a safety block, an
empty candidate). prompt_expander turns None into the user's own words, so this
module never needs to know about that fallback.
"""

import config
from generate import http_client

KEY_ENV = "DASHSCOPE_API_KEY"
_PATH = "/services/aigc/text-generation/generation"


def complete(prompt: str, model: str) -> str | None:
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        # message format keeps the reply at output.choices[].message.content;
        # the legacy default puts it at output.text instead.
        "parameters": {"result_format": "message"},
    }
    data = http_client.post_json(config.DASHSCOPE_BASE_URL + _PATH, payload,
                                 KEY_ENV, timeout=60.0)
    return text_of(data)


def text_of(data: dict) -> str | None:
    """The reply text, or None. Split out so it can be unit tested offline."""
    output = data.get("output") or {}
    choices = output.get("choices") or []
    if choices:
        content = (choices[0].get("message") or {}).get("content")
        # Some qwen models answer with a list of parts rather than a string.
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content
                              if isinstance(part, dict))
        return content or None
    return output.get("text") or None

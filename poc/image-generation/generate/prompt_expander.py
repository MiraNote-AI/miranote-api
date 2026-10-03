"""LLM prompt expansion: rewrite a user's short input into a richer image prompt
via Gemini. Used by /generate (expand, expand_background). Static template/preset
assembly lives in the *_presets modules, not here.

The three functions differ only in which system prompt they load; each takes the
model id from its caller, so pointing one at a different model is a config
change rather than a code change. _complete routes on that id.
"""

from pathlib import Path

from shared.vertex_client import _get_client
from generate import dashscope_text


def _complete(prompt: str, model: str) -> str | None:
    """One text completion, routed by model id. Gemini stays the default; the
    qwen-* ids go to DashScope. None when the provider returned no text."""
    if model.startswith("qwen"):
        return dashscope_text.complete(prompt, model)
    return _get_client().models.generate_content(model=model, contents=prompt).text


_PROMPT_DIR = Path(__file__).parent
SYSTEM_PROMPT = (_PROMPT_DIR / "sticker_system.txt").read_text(encoding="utf-8")


def expand(user_input: str, model: str) -> str:
    prompt = SYSTEM_PROMPT.replace("{{USER_INPUT}}", user_input)
    # _complete is None when the model returns no text part (e.g. a safety
    # block); fall back to the user's own words instead of crashing on .strip().
    return (_complete(prompt, model) or user_input).strip()


BACKGROUND_SYSTEM_PROMPT = (_PROMPT_DIR / "background_system.txt").read_text(encoding="utf-8")


def expand_background(user_input: str, model: str) -> str:
    prompt = BACKGROUND_SYSTEM_PROMPT.replace("{user_prompt}", user_input)
    # _complete is None when the model returns no text part (e.g. a safety
    # block); fall back to the user's own words instead of crashing on .strip().
    return (_complete(prompt, model) or user_input).strip()


ART_SYSTEM_PROMPT = (_PROMPT_DIR / "art_system.txt").read_text(encoding="utf-8")


def expand_art(user_input: str, model: str) -> str:
    prompt = ART_SYSTEM_PROMPT.replace("{user_prompt}", user_input)
    # Same safety-block fallback as expand_background.
    return (_complete(prompt, model) or user_input).strip()

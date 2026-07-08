"""LLM prompt expansion: rewrite a user's short input into a richer image prompt
via Gemini. Used by /generate (expand, expand_background). Static template/preset
assembly lives in the *_presets modules, not here.
"""

from pathlib import Path

from shared.vertex_client import _get_client


_PROMPT_DIR = Path(__file__).parent
SYSTEM_PROMPT = (_PROMPT_DIR / "sticker_system.txt").read_text(encoding="utf-8")


def expand(user_input: str, model: str) -> str:
    prompt = SYSTEM_PROMPT.replace("{{USER_INPUT}}", user_input)
    response = _get_client().models.generate_content(
        model=model,
        contents=prompt,
    )
    # response.text is None when the model returns no text part (e.g. a safety
    # block); fall back to the user's own words instead of crashing on .strip().
    return (response.text or user_input).strip()


BACKGROUND_SYSTEM_PROMPT = (_PROMPT_DIR / "background_system.txt").read_text(encoding="utf-8")


def expand_background(user_input: str, model: str) -> str:
    prompt = BACKGROUND_SYSTEM_PROMPT.replace("{user_prompt}", user_input)
    response = _get_client().models.generate_content(model=model, contents=prompt)
    # response.text is None when the model returns no text part (e.g. a safety
    # block); fall back to the user's own words instead of crashing on .strip().
    return (response.text or user_input).strip()

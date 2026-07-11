"""
Static prompt assembly for the /generate endpoint.

The (LLM-expanded or raw) user prompt is combined with a fixed, per-command
suffix/rule loaded from the prompt txt files. This mirrors style_presets /
border_presets: main.py only calls build_*_prompt(), and all static template
assembly lives here. LLM expansion itself lives in prompt_expander, not here.
"""

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent
_STICKER_SUFFIX = (_PROMPT_DIR / "sticker_suffix.txt").read_text(encoding="utf-8").strip()
# background_rule.txt intentionally starts with ", " so it appends cleanly.
_BACKGROUND_RULE = (_PROMPT_DIR / "background_rule.txt").read_text(encoding="utf-8").strip()
_ART_SUFFIX = (_PROMPT_DIR / "art_suffix.txt").read_text(encoding="utf-8").strip()


def build_sticker_prompt(core: str) -> str:
    """Append the fixed sticker suffix to the (expanded or raw) user prompt."""
    return f"{core}, {_STICKER_SUFFIX}"


def build_background_prompt(core: str) -> str:
    """Append the fixed background rule to the (expanded or raw) user prompt."""
    return core + _BACKGROUND_RULE


def build_art_prompt(core: str) -> str:
    """Append the fixed art suffix to the (expanded or raw) user prompt."""
    return f"{core}, {_ART_SUFFIX}"

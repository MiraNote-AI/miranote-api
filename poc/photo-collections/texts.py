"""Every piece of Chinese text the service uses, loaded from prompts/*.txt.

Source files stay ASCII-only (org Rule 3); prompts/ is the allowlisted
place for prompts, word lists, the question bank and user-facing messages.
Files are re-read when they change on disk, so a word list can be edited
without a restart.
"""

from __future__ import annotations

import json as _json
import threading
from typing import Any, Dict, Tuple

import config

_lock = threading.Lock()
_cache: Dict[str, Tuple[float, str]] = {}


def _path(name: str):
    return config.PROMPT_DIR / (name + ".txt")


def raw(name: str) -> str:
    """The file prompts/<name>.txt as text."""
    path = _path(name)
    mtime = path.stat().st_mtime
    with _lock:
        hit = _cache.get(name)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    text = path.read_text(encoding="utf-8")
    with _lock:
        _cache[name] = (mtime, text)
    return text


def json(name: str) -> Any:
    """prompts/<name>.txt parsed as JSON (bank, messages, names)."""
    return _json.loads(raw(name))


def render(name: str, **values: Any) -> str:
    """prompts/<name>.txt with {placeholders} filled.

    Literal braces in a prompt are written {{ and }}. Values are inserted
    as-is, so braces inside a value (a place name, say) are never parsed.
    """
    return raw(name).format_map(values)


def lines(name: str) -> list:
    """Non-empty, non-comment lines of prompts/<name>.txt, stripped."""
    out = []
    for line in raw(name).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def words(name: str) -> list:
    """Whitespace-separated words of prompts/<name>.txt (comments skipped)."""
    out = []
    for line in lines(name):
        out.extend(line.split())
    return out


def message(reason: str) -> str:
    """The Chinese message shown on the phone for a rejection/error reason."""
    table = json("messages")
    return table.get(reason, table["model_error"])

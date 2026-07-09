"""Journal mode: chat grounded in the user's own MiraNote pages.

The iOS app sends the pages that matched the user's message (selected
on-device -- notes never live on this server). A request that carries a
`notes` field runs as the journaling companion: docs tools are withheld
and the pages ride along with the user message as a context block.
"""
from __future__ import annotations

from typing import Any, Dict, List

# The documentation-assistant tools. In journal mode these would leak
# the demo docs corpus into answers about the user's own notes.
DOCS_TOOL_NAMES = {"list_docs", "read_doc", "search_docs", "set_docs_root"}

MAX_NOTES = 8
MAX_BODY_CHARS = 600


def journal_tools(all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Everything except the docs tools (text transforms, find_quote)."""
    return [t for t in all_tools if t["function"]["name"] not in DOCS_TOOL_NAMES]


def render_notes(notes: List[Dict[str, str]]) -> str:
    """The context block listing the user's matching pages."""
    lines = ["[Pages from the user's own MiraNote library that matched this message]"]
    for note in notes[:MAX_NOTES]:
        title = (note.get("title") or "Untitled").strip()
        date = (note.get("date") or "").strip()
        body = (note.get("body") or "").strip()[:MAX_BODY_CHARS]
        head = f'- "{title}"' + (f" ({date})" if date else "")
        lines.append(f"{head}: {body}" if body else head)
    lines.append("[End of pages]")
    return "\n".join(lines)


def compose_user_message(message: str, notes: List[Dict[str, str]]) -> str:
    """Prepend the context block; an empty match list is stated, not omitted,
    so the model never guesses that pages might exist."""
    if not notes:
        return (
            "[No pages in the user's MiraNote library matched this message]\n\n"
            + message
        )
    return render_notes(notes) + "\n\n" + message

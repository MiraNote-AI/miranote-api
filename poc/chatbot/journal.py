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

# Journal-only: the model drafts a page and the APP opens it for the
# user to shape. The server stores nothing -- the app reads the call's
# arguments out of tool_trace.
CREATE_NOTE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "create_note",
        "description": (
            "Create a draft MiraNote page for the user. Call this when the "
            "user asks to note something down, save a memory, or make a new "
            "page. The app opens the draft for the user to review and edit; "
            "nothing is stored on the server."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short, evocative page title.",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "A few warm sentences capturing the memory, in the "
                        "user's language."
                    ),
                },
            },
            "required": ["title", "body"],
        },
    },
}


def journal_tools(all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The docs tools stay behind; page drafting joins."""
    kept = [t for t in all_tools if t["function"]["name"] not in DOCS_TOOL_NAMES]
    return kept + [CREATE_NOTE_TOOL]


def render_notes(notes: List[Dict[str, str]]) -> str:
    """The context block listing the user's matching pages."""
    lines = ["[The user's own MiraNote pages for this conversation -- the first may be the page open in the editor]"]
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

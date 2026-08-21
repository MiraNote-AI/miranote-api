"""Canvas mode: chat grounded in the page the user is editing.

A request carrying `page` is a canvas turn. The page arrives as a
structured map -- every element with a short handle, its box, and what
it says -- and is rendered into a context block the model reads.

The map is the source of truth for facts (what is where, how big,
what overlaps). A rendered image of the same page rides along with the
request but is only spent when the model calls look_at_page.
"""
from __future__ import annotations

import pathlib
from typing import Any, Callable, Dict, List, Optional

from poc.chatbot import journal

MAX_ELEMENTS = 24
MAX_SAYS_CHARS = 200

# Text and sticker contents are quoted (they are the user's words);
# photo and sound descriptions are prose written by vision or by the
# user, and quoting them invites the model to "edit" them.
_QUOTED_KINDS = {"text", "sticker"}


def _fmt(value: float) -> str:
    """Whole numbers read cleanly; the app sends points, not fractions."""
    return str(int(round(float(value))))


def _element_line(element: Dict[str, Any]) -> str:
    handle = str(element.get("handle", "?"))
    kind = str(element.get("kind", "?"))
    box = "({},{})".format(_fmt(element.get("x", 0)), _fmt(element.get("y", 0)))
    size = "{}x{}".format(_fmt(element.get("w", 0)), _fmt(element.get("h", 0)))

    marks: List[str] = []
    if element.get("point_size"):
        marks.append("{}pt".format(_fmt(element["point_size"])))
    if element.get("color"):
        marks.append(str(element["color"]))
    if element.get("treatment"):
        marks.append(str(element["treatment"]))
    if element.get("rotation"):
        marks.append("tilted {}".format(_fmt(element["rotation"])))

    says = str(element.get("says", "")).strip()[:MAX_SAYS_CHARS]
    if says and kind in _QUOTED_KINDS:
        says = '"{}"'.format(says)

    parts = ["{:<3} {:<8}".format(handle, kind), "{:<10}".format(box), "{:<9}".format(size)]
    parts.extend(marks)
    if says:
        parts.append(says)
    return "  ".join(part for part in parts if part).rstrip()


def render_page(page: Dict[str, Any]) -> str:
    """The context block describing the page open in the editor."""
    header = "[The page open in the editor -- {} wide, {} tall, {} background]".format(
        _fmt(page.get("width", 0)),
        _fmt(page.get("height", 0)),
        str(page.get("background") or "default gradient"),
    )

    palette = list(page.get("palette") or [])
    footer: List[str] = []
    if palette:
        footer.append("Text colors: " + ", ".join(str(name) for name in palette))
    footer.append("[End of page]")

    elements = list(page.get("elements") or [])[:MAX_ELEMENTS]
    if not elements:
        return "\n".join([header, "The page is open but there is nothing on it yet."] + footer)

    lines = [header]
    lines.extend(_element_line(element) for element in elements)
    omitted = int(page.get("omitted") or 0)
    if omitted > 0:
        lines.append("({} more elements not listed)".format(omitted))
    lines.extend(footer)
    return "\n".join(lines)


CANVAS_PROMPT_PATH = pathlib.Path(__file__).parent / "prompts" / "canvas.txt"

CANVAS_TOOL_NAMES = {
    "edit_page", "set_background", "clear_background", "look_at_page",
    "restyle_photo",
}

# Descriptions (with their Chinese trigger phrases) live in prompts/,
# which is the CJK-allowlisted path; code stays ASCII.
_DESC_PATH = pathlib.Path(__file__).parent / "prompts" / "tool_descriptions.txt"
_DESCRIPTIONS: Dict[str, str] = {}
for _line in _DESC_PATH.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if not _line or _line.startswith("#"):
        continue
    _name, _, _desc = _line.partition("|")
    _DESCRIPTIONS[_name.strip()] = _desc.strip()


def _tool(name: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _DESCRIPTIONS[name],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_CHANGE_ITEM: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "The element handle from the page block, e.g. t1 or p1."},
        "x": {"type": "number", "description": "New left edge, in canvas points."},
        "y": {"type": "number", "description": "New top edge, in canvas points."},
        "w": {"type": "number", "description": "New width, in canvas points."},
        "h": {"type": "number", "description": "New height, in canvas points."},
        "size": {
            "type": "number", "minimum": 11, "maximum": 48,
            "description": "New text point size. Text elements only.",
        },
        "color": {"type": "string", "description": "New text color, a palette name from the page block."},
        "layer": {
            "type": "string", "enum": ["front", "back"],
            "description": "Bring the element in front of, or behind, the others.",
        },
    },
    "required": ["id"],
}

EDIT_PAGE_TOOL = _tool(
    "edit_page",
    {"changes": {"type": "array", "items": _CHANGE_ITEM, "description": "Every change this turn, together."}},
    ["changes"],
)

SET_BACKGROUND_TOOL = _tool(
    "set_background",
    {"prompt": {"type": "string", "description": "What the backdrop should show, in a short phrase."}},
    ["prompt"],
)

CLEAR_BACKGROUND_TOOL = _tool("clear_background", {}, [])

LOOK_AT_PAGE_TOOL = _tool("look_at_page", {}, [])

RESTYLE_PHOTO_TOOL = _tool(
    "restyle_photo",
    {
        "id": {"type": "string", "description": "The photo's handle from the page block, e.g. p1."},
        "instruction": {
            "type": "string",
            "description": (
                "What the photo should become, as a short phrase -- "
                "'warmer and softer', 'like a faded film photo'."
            ),
        },
    },
    ["id", "instruction"],
)


def canvas_tools(all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The journal set plus the page tools. Docs tools stay behind."""
    return list(journal.journal_tools(all_tools)) + [
        EDIT_PAGE_TOOL, SET_BACKGROUND_TOOL, CLEAR_BACKGROUND_TOOL, LOOK_AT_PAGE_TOOL,
        RESTYLE_PHOTO_TOOL,
    ]


# What look_at_page asks about a whole page. /describe's own default is
# tuned for photos ("what is in it, the mood"); a page needs a different
# question entirely.
LOOK_PROMPT = (
    "This is a page from a personal journaling app, exactly as the "
    "person editing it sees it. Describe how the page looks: how the "
    "pieces are arranged, what draws the eye, what feels crowded or "
    "empty, and how the colors sit together. If it has photos, say what "
    "is in them. Be concrete and brief."
)

# The weak model reads a tool RESULT far more reliably than it reads
# the system prompt -- and a look with no follow-through is exactly how
# "tidy up" becomes a chat bubble instead of a change. The look's
# result ends with the one imperative that matters.
LOOK_ACTION_NUDGE = (
    "\n\n[You have now seen the page. If the user asked you to CHANGE it, "
    "apply the change now in this same turn with the matching tool: "
    "edit_page with concrete values for every element you move or resize, "
    "restyle_photo for a photo's look, or set_background for the backdrop. "
    "If they only asked how it looks, answer that.]"
)

# The loop-level guard's nudge (chat_loop.ActionGuard): fired when the
# model looked at the page and then answered in prose instead of acting.
ACTION_NUDGE = (
    "You looked at the page but have not changed it. If the user asked "
    "you to change the page, apply the change now with the matching "
    "tool -- edit_page for layout, restyle_photo for a photo's look, "
    "set_background or clear_background for the backdrop. If they only "
    "asked how the page looks, answer now."
)


def build_dispatcher(
    image_bytes: Optional[bytes],
    image_client: Any,
    fallback: Callable[[str, Dict[str, Any]], Any],
) -> Callable[[str, Dict[str, Any]], Any]:
    """Tool dispatch for one canvas turn.

    look_at_page is the only canvas tool that does work, and only once
    per turn -- max_tool_iterations would otherwise let the model spend
    six vision calls looking at the same picture.

    The rest are pure handoffs: the app reads their arguments out of
    tool_trace and executes them itself, exactly as with create_note.
    The server stores nothing about the user's page.
    """
    state = {"looked": False}

    def dispatch(name: str, args: Dict[str, Any]) -> Any:
        if name == "look_at_page":
            if state["looked"]:
                return {"status": "already looked at this page this turn"}
            state["looked"] = True
            if not image_bytes:
                return {"status": "could not look at the page"}
            try:
                return {
                    "description": image_client.describe(image_bytes, LOOK_PROMPT)
                    + LOOK_ACTION_NUDGE
                }
            except Exception:  # noqa: BLE001 -- a failed look is not a failed turn
                return {"status": "could not look at the page"}
        if name in CANVAS_TOOL_NAMES:
            return {"status": "handed to the app"}
        return fallback(name, args)

    return dispatch

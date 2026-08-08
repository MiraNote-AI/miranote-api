"""Canvas mode: chat grounded in the page the user is editing.

A request carrying `page` is a canvas turn. The page arrives as a
structured map -- every element with a short handle, its box, and what
it says -- and is rendered into a context block the model reads.

The map is the source of truth for facts (what is where, how big,
what overlaps). A rendered image of the same page rides along with the
request but is only spent when the model calls look_at_page.
"""
from __future__ import annotations

from typing import Any, Dict, List

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

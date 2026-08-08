# Canvas mode for the chat backend -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `/chat` run a canvas turn -- reason over a structured map
of the page the user is editing, act on it through page-edit tools, and
look at a rendered image of it only when the question needs pixels.

**Architecture:** A new `poc/chatbot/canvas.py` mirrors `journal.py`:
it renders the page block, defines the canvas tools, and gates them on
the request carrying `page`. `run_turn` gains a transient-prefix
parameter so the map reaches the model on every call but never enters
session history. `look_at_page` is the only canvas tool that does work
-- it calls the image POC through a new `image_client.py`; the other
three are pure handoffs whose arguments the iOS app reads out of
`tool_trace`, exactly like `create_note`.

**Tech Stack:** FastAPI, Pydantic v2, pytest, httpx, OpenAI-compatible
client (DeepSeek), Google GenAI (vision, in the image POC).

Implements miranote-api#38. Design:
`miranote-ios/docs/specs/2026-08-08-canvas-vision-design.md`.

## Global Constraints

- **No CJK or emoji in code.** Chinese trigger phrases live only in
  `poc/chatbot/prompts/tool_descriptions.txt`, which is allowlisted
  (`**/prompts/*.txt`). Test files are exempt but do not need it here.
- **Python target is 3.9.** Use `Optional[X]` and `List[X]` from
  `typing`, never `X | None` or `list[X]`.
- **Every module starts with `from __future__ import annotations`,**
  matching every existing file in `poc/chatbot/`.
- **Tests run from the repo root:**
  `PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v`
- **The server stores nothing about the user's pages.** Canvas tools
  other than `look_at_page` are pure handoffs.
- **Element handle vocabulary:** `t` text, `p` photo, `s` sticker,
  `a` sound, each followed by a 1-based index.
- **Point size range is 11-48**; the app clamps, the tool description
  states it.

---

### Task 1: Render the page map block

**Files:**
- Create: `poc/chatbot/canvas.py`
- Test: `poc/chatbot/tests/test_canvas.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `canvas.render_page(page: Dict[str, Any]) -> str`,
  `canvas.MAX_ELEMENTS = 24`, `canvas.MAX_SAYS_CHARS = 200`.
  The `page` dict has keys `width`, `height`, `background`,
  `palette`, `elements`, `omitted`. Each element has `handle`, `kind`,
  `x`, `y`, `w`, `h`, `says`, and optional `point_size`, `color`,
  `rotation`, `treatment`.

  `palette` is the list of text color names the app will accept. It
  must be stated, not inferred: the element lines only show colors
  currently in use, so a page whose text is all default would leave
  the model with nothing to choose from and it would invent a name.

- [ ] **Step 1: Write the failing test**

Create `poc/chatbot/tests/test_canvas.py`:

```python
from __future__ import annotations

from poc.chatbot import canvas


def _page(**overrides):
    page = {
        "width": 393.0,
        "height": 812.0,
        "background": "default gradient",
        "palette": ["ink", "forest", "taupe", "tan", "sage"],
        "omitted": 0,
        "elements": [
            {
                "handle": "t1", "kind": "text",
                "x": 28.0, "y": 40.0, "w": 304.0, "h": 44.0,
                "says": "Noodle shop by the bridge", "point_size": 30.0,
            },
            {
                "handle": "p1", "kind": "photo",
                "x": 40.0, "y": 110.0, "w": 280.0, "h": 200.0,
                "says": "a steaming bowl on a wooden counter",
                "treatment": "film",
            },
        ],
    }
    page.update(overrides)
    return page


def test_render_page_states_canvas_size_and_background():
    block = canvas.render_page(_page())
    assert block.startswith("[The page open in the editor -- 393 wide, 812 tall")
    assert "default gradient" in block
    assert block.rstrip().endswith("[End of page]")


def test_render_page_lists_every_element_with_handle_and_box():
    block = canvas.render_page(_page())
    assert "t1  text" in block
    assert "(28,40)" in block
    assert "304x44" in block
    assert "30pt" in block
    assert '"Noodle shop by the bridge"' in block
    assert "p1  photo" in block
    assert "film" in block
    # Photo descriptions are prose, not a quoted string the model
    # might think it can edit.
    assert '"a steaming bowl' not in block


def test_render_page_omits_absent_optional_fields():
    block = canvas.render_page(_page())
    assert "tilted" not in block
    assert "None" not in block


def test_render_page_includes_rotation_when_present():
    page = _page()
    page["elements"][0]["rotation"] = 8.0
    assert "tilted 8" in canvas.render_page(page)


def test_render_page_states_the_omitted_count():
    block = canvas.render_page(_page(omitted=6))
    assert "6 more elements not listed" in block


def test_render_page_lists_the_colors_the_app_will_accept():
    # The element lines only show colors in use; a page of default text
    # would otherwise leave the model guessing at names.
    block = canvas.render_page(_page())
    assert "Text colors: ink, forest, taupe, tan, sage" in block


def test_render_page_omits_the_color_line_when_none_are_offered():
    assert "Text colors" not in canvas.render_page(_page(palette=[]))


def test_render_page_says_so_when_the_page_is_empty():
    block = canvas.render_page(_page(elements=[], omitted=0))
    assert "nothing on it yet" in block


def test_render_page_truncates_very_long_text():
    page = _page()
    page["elements"][0]["says"] = "x" * 1000
    block = canvas.render_page(page)
    assert "x" * (canvas.MAX_SAYS_CHARS + 1) not in block
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_canvas.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'poc.chatbot.canvas'`

- [ ] **Step 3: Write the minimal implementation**

Create `poc/chatbot/canvas.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_canvas.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/canvas.py poc/chatbot/tests/test_canvas.py
git commit -m "feat(api): render the editor page as a chat context block

Refs #38"
```

---

### Task 2: Define the canvas tool set

**Files:**
- Modify: `poc/chatbot/canvas.py`
- Modify: `poc/chatbot/prompts/tool_descriptions.txt`
- Create: `poc/chatbot/prompts/canvas.txt`
- Test: `poc/chatbot/tests/test_canvas.py` (append)

**Interfaces:**
- Consumes: `journal.journal_tools` from Task 1's neighbouring module.
- Produces: `canvas.canvas_tools(all_tools: List[Dict]) -> List[Dict]`,
  `canvas.CANVAS_TOOL_NAMES: Set[str]`, and the module constant
  `canvas.CANVAS_PROMPT_PATH`. Tool names are exactly `edit_page`,
  `set_background`, `clear_background`, `look_at_page`.

- [ ] **Step 1: Write the failing test**

Append to `poc/chatbot/tests/test_canvas.py`:

```python
from poc.chatbot import journal, tools


def test_canvas_tools_add_the_page_tools_to_the_journal_set():
    names = {t["function"]["name"] for t in canvas.canvas_tools(tools.TOOLS)}
    assert {"edit_page", "set_background", "clear_background", "look_at_page"} <= names
    # Journal keeps its own; docs tools still stay behind.
    assert "create_note" in names
    assert names.isdisjoint(journal.DOCS_TOOL_NAMES)


def test_edit_page_accepts_only_the_documented_fields():
    tool = next(
        t for t in canvas.canvas_tools(tools.TOOLS)
        if t["function"]["name"] == "edit_page"
    )
    entry = tool["function"]["parameters"]["properties"]["changes"]["items"]
    assert set(entry["properties"]) == {"id", "x", "y", "w", "h", "size", "color", "layer"}
    assert entry["required"] == ["id"]
    assert entry["properties"]["layer"]["enum"] == ["front", "back"]
    assert entry["properties"]["size"]["minimum"] == 11
    assert entry["properties"]["size"]["maximum"] == 48


def test_look_at_page_takes_no_arguments():
    tool = next(
        t for t in canvas.canvas_tools(tools.TOOLS)
        if t["function"]["name"] == "look_at_page"
    )
    assert tool["function"]["parameters"]["properties"] == {}


def test_canvas_tool_descriptions_carry_chinese_triggers():
    tool = next(
        t for t in canvas.canvas_tools(tools.TOOLS)
        if t["function"]["name"] == "edit_page"
    )
    # The trigger phrases live in prompts/, never in code.
    assert "挪" in tool["function"]["description"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_canvas.py -v`
Expected: FAIL, `AttributeError: module 'poc.chatbot.canvas' has no attribute 'canvas_tools'`

- [ ] **Step 3: Add the trigger phrases**

Append to `poc/chatbot/prompts/tool_descriptions.txt`:

```
edit_page|Move, resize, restyle or re-layer elements on the page the user is editing. Call when the user asks to move something ('move', 'nudge', '挪', '移到'), resize it ('bigger', 'smaller', '放大', '缩小'), change text size or color ('太小了', '换个颜色'), restack it ('放到后面', 'behind'), or rearrange the whole page ('tidy up', 'arrange', '排一排', '重排'). Reference elements by the handle shown in the page block (t1, p1, s1, a1). One call may carry many changes; send them together.
set_background|Give the page a new AI-generated background. Call when the user asks for a backdrop or scene ('give this page a sunset background', '换个星空背景', '背景换一个'). The app runs the generation and shows the user two candidates; you only make the request.
clear_background|Remove the page background, returning it to the default gradient. Call for 'remove the background', '去掉背景'.
look_at_page|Look at what the page actually looks like right now. Call ONLY when the answer needs the picture rather than the numbers -- how it looks, whether it feels crowded or balanced, whether the colors sit well, what is in a photo beyond its one-line description. Position, size and overlap are already in the page block; do not call this for those. Once per turn.
```

- [ ] **Step 4: Add the canvas system prompt**

Create `poc/chatbot/prompts/canvas.txt`:

```
You are Mira, working alongside someone editing a page in the
MiraNote app. Each message begins with a bracketed block describing
the page open in front of them: the canvas size, and every element
with a short handle (t1 t2 for text, p1 for photos, s1 for stickers,
a1 for sounds), its top-left corner, its width and height, and what it
says. That block is written by the app, not by the user, and it
describes the page as it is right now. The user may call it a canvas
or a page -- same thing.

- The block is the truth about WHERE things are and HOW BIG they are.
  Answer questions about position, size, and overlap from it directly.
- It does not tell you how the page LOOKS. When the question is about
  appearance -- is it balanced, is it crowded, do the colors work, what
  is really in that photo -- call look_at_page first, then answer.
- To change the page, call edit_page with the handles from the block.
  Coordinates are in the same space the block uses: top-left corner,
  in the canvas width the header states. Never assume a width.
- When you rearrange the whole page, send every element you are moving
  in a single edit_page call, and leave a comfortable gap between them
  using the heights in the block.
- Changing a text size changes how tall that block becomes. If you are
  both resizing text and placing things below it, leave room.
- After a change lands, the app shows the user its own short receipt.
  Your reply should be one warm line about what you did, not a list.
- If you cannot tell which element the user means, ask rather than
  guessing.
- Be warm and brief. Reply in the language the user wrote in.
```

- [ ] **Step 5: Write the implementation**

Append to `poc/chatbot/canvas.py`:

```python
import pathlib

from poc.chatbot import journal

CANVAS_PROMPT_PATH = pathlib.Path(__file__).parent / "prompts" / "canvas.txt"

CANVAS_TOOL_NAMES = {"edit_page", "set_background", "clear_background", "look_at_page"}

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
        "layer": {"type": "string", "enum": ["front", "back"], "description": "Bring the element in front of, or behind, the others."},
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


def canvas_tools(all_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The journal set plus the page tools. Docs tools stay behind."""
    return list(journal.journal_tools(all_tools)) + [
        EDIT_PAGE_TOOL, SET_BACKGROUND_TOOL, CLEAR_BACKGROUND_TOOL, LOOK_AT_PAGE_TOOL,
    ]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_canvas.py -v`
Expected: PASS, 11 passed

- [ ] **Step 7: Confirm no CJK leaked into code**

Run from the `.github` checkout:
`PYTHONPATH=. python3 -m checks.no_cjk_or_emoji <path-to>/miranote-api`
Expected: no lines mentioning `canvas.py`

- [ ] **Step 8: Commit**

```bash
git add poc/chatbot/canvas.py poc/chatbot/prompts/ poc/chatbot/tests/test_canvas.py
git commit -m "feat(api): define the canvas page-edit and look tools

Refs #38"
```

---

### Task 3: Keep the page map out of session history

**Files:**
- Modify: `poc/chatbot/chat_loop.py:36-60`
- Test: `poc/chatbot/tests/test_chat_loop.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `run_turn(..., transient_prefix: Optional[str] = None)`.
  When set, the prefix is prepended to the current user message for
  every model call in the turn, and is absent from the stored history.

**Why this task exists:** `run_turn` appends the composed message to
the session, 40 messages deep. Persisting page maps would leave the
model reading several contradicting maps, and handles are renumbered
by y every turn, so an old `t1` may name a different element. The
model would move the wrong thing with full confidence.

- [ ] **Step 1: Write the failing test**

Append to `poc/chatbot/tests/test_chat_loop.py`:

These use the doubles already at the top of that file (`_msg`,
`_resp`, `FakeClient`) -- do not add a second fake client.

```python
def test_transient_prefix_reaches_the_model_but_not_the_history():
    client = FakeClient([_resp(_msg(content="ok"))])
    store = SessionStore(ttl_seconds=3600)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="move the title up", model="fake",
        tools=[], tool_dispatcher=lambda n, a: None,
        max_iterations=3, max_history=40, system_prompt="sys",
        transient_prefix="[PAGE MAP]",
    )

    sent = client.chat.completions.calls[0]["messages"][-1]["content"]
    assert sent.startswith("[PAGE MAP]")
    assert "move the title up" in sent

    stored = store.get(result.session_id)
    user_messages = [m for m in stored if m.get("role") == "user"]
    assert user_messages[-1]["content"] == "move the title up"
    assert all("[PAGE MAP]" not in (m.get("content") or "") for m in stored)


def test_transient_prefix_survives_a_tool_round_trip():
    # The prefix must be on EVERY call in the turn, not just the first:
    # a turn that calls look_at_page and then answers would otherwise
    # lose the page map exactly when it needs it.
    tool_call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(name="look_at_page", arguments="{}"),
    )
    client = FakeClient([
        _resp(_msg(tool_calls=[tool_call])),
        _resp(_msg(content="it looks calm")),
    ])
    store = SessionStore(ttl_seconds=3600)
    run_turn(
        client=client, session_store=store, session_id=None,
        user_message="how does it look", model="fake",
        tools=[], tool_dispatcher=lambda n, a: {"description": "calm"},
        max_iterations=3, max_history=40, system_prompt="sys",
        transient_prefix="[PAGE MAP]",
    )
    for call in client.chat.completions.calls:
        assert any(
            "[PAGE MAP]" in (m.get("content") or "") for m in call["messages"]
        )


def test_no_transient_prefix_leaves_the_message_untouched():
    client = FakeClient([_resp(_msg(content="ok"))])
    store = SessionStore(ttl_seconds=3600)
    run_turn(
        client=client, session_store=store, session_id=None,
        user_message="hello", model="fake", tools=[],
        tool_dispatcher=lambda n, a: None, max_iterations=3,
        max_history=40, system_prompt="sys",
    )
    assert client.chat.completions.calls[0]["messages"][-1]["content"] == "hello"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_chat_loop.py -v -k transient`
Expected: FAIL, `TypeError: run_turn() got an unexpected keyword argument 'transient_prefix'`

- [ ] **Step 3: Write the implementation**

In `poc/chatbot/chat_loop.py`, add the pure helper above `run_turn`:

```python
def _messages_for_model(
    history: List[Dict[str, Any]],
    transient_prefix: Optional[str],
    user_index: int,
) -> List[Dict[str, Any]]:
    """History as the model should see it this turn.

    The prefix (the page map) is context the model needs on every call
    but must never persist: it goes stale the moment the page changes,
    and its element handles are renumbered each turn.
    """
    if not transient_prefix:
        return history
    view = list(history)
    original = view[user_index].get("content", "")
    view[user_index] = {"role": "user", "content": transient_prefix + "\n\n" + original}
    return view
```

Add the parameter to `run_turn`'s signature (keyword-only, after
`system_prompt`):

```python
    system_prompt: str,
    transient_prefix: Optional[str] = None,
) -> ChatTurnResult:
```

Record the index right after the user message is appended:

```python
    history = session_store.get(session_id)
    user_index = len(history)
    history.append({"role": "user", "content": user_message})
```

And inside the iteration loop, replace the messages kwarg:

```python
    for _ in range(max_iterations):
        kwargs = {"model": model, "messages": _messages_for_model(history, transient_prefix, user_index)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_chat_loop.py -v`
Expected: PASS, all existing tests plus the two new ones

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/chat_loop.py poc/chatbot/tests/test_chat_loop.py
git commit -m "feat(api): let a turn carry context the history does not keep

Refs #38"
```

---

### Task 4: Ask the image POC about a page

**Files:**
- Create: `poc/chatbot/image_client.py`
- Modify: `poc/image-generation/main.py:372-392`
- Test: `poc/chatbot/tests/test_image_client.py`
- Test: `poc/image-generation/tests/` (append to the existing
  `/describe` test if there is one; otherwise create
  `test_describe_prompt.py` there)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ImageClient(base_url: str, timeout: float = 25.0)` with
  `describe(self, image: bytes, prompt: str) -> str`, raising on
  transport or decode failure. `/describe` gains an optional `prompt`
  query parameter.

- [ ] **Step 1: Write the failing client test**

Create `poc/chatbot/tests/test_image_client.py`:

```python
from __future__ import annotations

import httpx
import pytest

from poc.chatbot.image_client import ImageClient


def test_describe_posts_the_image_and_returns_the_description(monkeypatch):
    captured = {}

    def fake_post(url, files=None, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["filename"] = files["file"][0]
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"description": "a calm page"}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    client = ImageClient("http://localhost:8002")
    assert client.describe(b"jpegbytes", "how does this page look?") == "a calm page"
    assert captured["url"].endswith("/describe")
    assert captured["params"]["prompt"] == "how does this page look?"
    assert captured["filename"] == "page.jpg"


def test_describe_raises_when_the_body_has_no_description(monkeypatch):
    def fake_post(url, files=None, params=None, timeout=None):
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={}, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(ValueError):
        ImageClient("http://localhost:8002").describe(b"x", "p")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_image_client.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'poc.chatbot.image_client'`

- [ ] **Step 3: Write the client**

Create `poc/chatbot/image_client.py`:

```python
"""HTTP client for the image-generation POC.

Canvas mode's look_at_page delegates to that service's /describe
endpoint rather than holding a second vision model here. Same shape as
text_client.py and retrieval_client.py: synchronous httpx, called from
the dispatcher inside run_turn's thread pool.

The default timeout is deliberately well under the app's turn budget:
a stuck look must fail fast enough that the model still has room to
answer from the page map.
"""
from __future__ import annotations

import httpx


class ImageClient:
    def __init__(self, base_url: str, timeout: float = 25.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def describe(self, image: bytes, prompt: str) -> str:
        response = httpx.post(
            self._base_url + "/describe",
            files={"file": ("page.jpg", image, "image/jpeg")},
            params={"prompt": prompt},
            timeout=self._timeout,
        )
        response.raise_for_status()
        description = (response.json() or {}).get("description", "")
        if not description:
            raise ValueError("the image service returned no description")
        return description
```

- [ ] **Step 4: Run it to verify it passes**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_image_client.py -v`
Expected: PASS, 2 passed

- [ ] **Step 5: Add the prompt parameter to `/describe`**

In `poc/image-generation/main.py`, replace the endpoint (currently at
line 372) with:

```python
DEFAULT_DESCRIBE_PROMPT = (
    "Describe this photo in one warm, concrete sentence "
    "(what is in it, the mood). Answer with the sentence only."
)


@app.post("/describe")
async def describe_image(file: UploadFile, prompt: Optional[str] = None):
    """Vision over one image. Default: one warm sentence about a photo,
    for the app's page context. Canvas mode passes its own prompt to ask
    what a whole page looks like."""
    raw = await file.read()
    question = (prompt or "").strip() or DEFAULT_DESCRIBE_PROMPT

    def _describe() -> str:
        from google.genai import types
        response = _get_client().models.generate_content(
            model=config.PROMPT_EXPANDER_MODEL,
            contents=[
                types.Part.from_bytes(data=raw, mime_type=file.content_type or "image/png"),
                question,
            ],
        )
        return (response.text or "").strip()

    description = await asyncio.to_thread(_describe)
    if not description:
        raise HTTPException(status_code=502, detail="the model returned no description")
    return {"description": description}
```

Confirm `Optional` is imported at the top of that file; add it to the
existing `from typing import ...` line if not.

- [ ] **Step 6: Cover the default, which nothing tests today**

That suite has `test_fallback.py` and `test_generate_presets.py` only
-- `/describe` is untested, so the photo-import path has no guard
against this change. Create
`poc/image-generation/tests/test_describe.py`:

```python
from __future__ import annotations

import main


def test_describe_defaults_to_the_photo_sentence():
    assert "one warm, concrete sentence" in main.DEFAULT_DESCRIBE_PROMPT


def test_a_caller_prompt_replaces_the_default(monkeypatch):
    asked = {}

    class FakeModels:
        def generate_content(self, model, contents):
            asked["question"] = contents[-1]
            return type("R", (), {"text": "a calm page"})()

    monkeypatch.setattr(main, "_get_client", lambda: type("C", (), {"models": FakeModels()})())

    import asyncio

    class Upload:
        content_type = "image/jpeg"

        async def read(self):
            return b"bytes"

    asyncio.run(main.describe_image(Upload(), prompt="how does this page look?"))
    assert asked["question"] == "how does this page look?"

    asyncio.run(main.describe_image(Upload()))
    assert asked["question"] == main.DEFAULT_DESCRIBE_PROMPT
```

Match that suite's import style -- check whether its existing tests
import `main` directly or as `poc.image_generation.main`, and follow
it.

Run: `PYTHONPATH=. python3 -m pytest poc/image-generation/tests -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add poc/chatbot/image_client.py poc/chatbot/tests/test_image_client.py poc/image-generation/main.py
git commit -m "feat(api): let /describe answer a caller-supplied question

Refs #38"
```

---

### Task 5: Run a canvas turn end to end

**Files:**
- Modify: `poc/chatbot/main.py:85-140`
- Modify: `poc/chatbot/config.py`
- Test: `poc/chatbot/tests/test_canvas_endpoint.py`

**Interfaces:**
- Consumes: `canvas.render_page`, `canvas.canvas_tools`,
  `canvas.CANVAS_PROMPT_PATH`, `canvas.CANVAS_TOOL_NAMES` (Tasks 1-2);
  `run_turn(..., transient_prefix=...)` (Task 3); `ImageClient`
  (Task 4).
- Produces: `/chat` accepting `page`, and `tool_trace` entries the iOS
  app reads.

- [ ] **Step 1: Write the failing test**

Create `poc/chatbot/tests/test_canvas_endpoint.py`:

```python
from __future__ import annotations

from poc.chatbot import canvas


class _StubImageClient:
    def __init__(self):
        self.calls = 0

    def describe(self, image, prompt):
        self.calls += 1
        return "a calm page with room at the bottom"


def _dispatcher(image_bytes, image_client):
    from poc.chatbot.main import build_canvas_dispatcher
    return build_canvas_dispatcher(image_bytes, image_client)


def test_page_edit_tools_are_pure_handoffs():
    dispatch = _dispatcher(b"jpeg", _StubImageClient())
    for name in ("edit_page", "set_background", "clear_background"):
        result = dispatch(name, {"changes": []})
        assert "handed" in result["status"]


def test_look_at_page_calls_vision_once_per_turn():
    client = _StubImageClient()
    dispatch = _dispatcher(b"jpeg", client)
    first = dispatch("look_at_page", {})
    second = dispatch("look_at_page", {})
    assert first["description"].startswith("a calm page")
    assert "already looked" in second["status"]
    assert client.calls == 1


def test_look_at_page_reports_it_could_not_look_without_an_image():
    dispatch = _dispatcher(None, _StubImageClient())
    assert "could not look" in dispatch("look_at_page", {})["status"]


def test_look_at_page_survives_a_failing_vision_call():
    class Broken:
        def describe(self, image, prompt):
            raise RuntimeError("gemini is down")

    dispatch = _dispatcher(b"jpeg", Broken())
    assert "could not look" in dispatch("look_at_page", {})["status"]


def test_canvas_mode_composes_the_map_as_transient_context():
    from poc.chatbot.main import canvas_context
    page = {"width": 393, "height": 800, "background": "",
            "palette": ["ink"], "elements": [], "omitted": 0}
    prefix = canvas_context(page)
    assert prefix == canvas.render_page(page)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_canvas_endpoint.py -v`
Expected: FAIL, `ImportError: cannot import name 'build_canvas_dispatcher'`

- [ ] **Step 3: Extend the request model**

In `poc/chatbot/main.py`, beside `NoteIn`:

```python
class ElementIn(BaseModel):
    handle: str
    kind: str
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    says: str = ""
    point_size: Optional[float] = None
    color: Optional[str] = None
    rotation: Optional[float] = None
    treatment: Optional[str] = None


class PageIn(BaseModel):
    width: float
    height: float
    background: str = ""
    # Text color names the app will accept; stated so the model picks
    # from the palette instead of inventing "warm beige".
    palette: List[str] = []
    elements: List[ElementIn] = []
    omitted: int = 0
    # base64 JPEG of the page as the user sees it. Present on every
    # canvas turn; spent only if the model calls look_at_page.
    image: Optional[str] = None
```

and add to `ChatRequest`:

```python
    # Present = canvas mode: the user is editing this page right now.
    page: Optional[PageIn] = None
```

- [ ] **Step 4: Add the dispatcher and context helpers**

In `poc/chatbot/main.py`, above the `/chat` handler:

```python
CANVAS_PROMPT = canvas.CANVAS_PROMPT_PATH.read_text(encoding="utf-8")
CANVAS_TOOLS = canvas.canvas_tools(tools.TOOLS)
IMAGE_API_URL = os.getenv("IMAGE_API_URL", "http://localhost:8002")
image_client = ImageClient(IMAGE_API_URL)

CANVAS_LOOK_PROMPT = (
    "This is a page from a personal journaling app, exactly as the "
    "person editing it sees it. Describe how it looks: how the pieces "
    "are arranged, what draws the eye, what feels crowded or empty, "
    "and how the colors sit together. If it has photos, say what is in "
    "them. Be concrete and brief."
)


def canvas_context(page: Dict[str, Any]) -> str:
    """The page block, injected per call and never stored."""
    return canvas.render_page(page)


def build_canvas_dispatcher(image_bytes: Optional[bytes], client: Any):
    """Canvas tools: look_at_page does work (once), the rest hand off.

    The app executes edit_page, set_background and clear_background by
    reading their arguments out of tool_trace -- the server keeps
    nothing, exactly as with create_note.
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
                return {"description": client.describe(image_bytes, CANVAS_LOOK_PROMPT)}
            except Exception:  # noqa: BLE001 -- a failed look is not a failed turn
                return {"status": "could not look at the page"}
        if name in canvas.CANVAS_TOOL_NAMES:
            return {"status": "handed to the app"}
        return _dispatcher(name, args)

    return dispatch
```

Add the imports at the top: `from poc.chatbot import canvas` and
`from poc.chatbot.image_client import ImageClient`, plus `base64`.

- [ ] **Step 5: Branch the handler**

Replace the body of `chat()` up to the `run_turn` call:

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    canvas_mode = req.page is not None
    journal_mode = req.notes is not None
    message = req.message
    transient_prefix = None
    dispatcher = _dispatcher
    turn_tools = tools.TOOLS
    prompt = SYSTEM_PROMPT

    if canvas_mode:
        page = req.page.model_dump()
        image_b64 = page.pop("image", None)
        try:
            image_bytes = base64.b64decode(image_b64) if image_b64 else None
        except Exception:  # noqa: BLE001 -- a bad image only costs the look
            image_bytes = None
        transient_prefix = canvas_context(page)
        dispatcher = build_canvas_dispatcher(image_bytes, image_client)
        turn_tools = CANVAS_TOOLS
        prompt = CANVAS_PROMPT
    elif journal_mode:
        message = journal.compose_user_message(
            req.message, [n.model_dump() for n in req.notes]
        )
        turn_tools = JOURNAL_TOOLS
        prompt = JOURNAL_PROMPT

    try:
        result: ChatTurnResult = await asyncio.to_thread(
            run_turn,
            client=client,
            session_store=sessions,
            session_id=req.session_id,
            user_message=message,
            model=config.model,
            tools=turn_tools,
            tool_dispatcher=dispatcher,
            max_iterations=config.max_tool_iterations,
            max_history=config.max_history_messages,
            system_prompt=prompt,
            transient_prefix=transient_prefix,
        )
```

Leave the rest of the handler (the exception mapping, the empty-reply
guard, the response) exactly as it is.

- [ ] **Step 6: Run the whole suite**

Run: `PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v`
Expected: PASS, everything including the pre-existing journal tests

- [ ] **Step 7: Smoke test against a running server**

```bash
# terminal 1
PYTHONPATH=. python3 -m uvicorn poc.chatbot.main:app --port 8003
# terminal 2
curl -s localhost:8003/chat -H 'Content-Type: application/json' -d '{
  "message": "move the title above the photo",
  "page": {"width": 393, "height": 812, "background": "default gradient",
    "elements": [
      {"handle":"t1","kind":"text","x":28,"y":300,"w":304,"h":44,"says":"Noodle shop","point_size":30},
      {"handle":"p1","kind":"photo","x":40,"y":60,"w":280,"h":200,"says":"a steaming bowl"}
    ], "omitted": 0}
}' | python3 -m json.tool
```

Expected: `tool_trace` contains an `edit_page` call whose `changes`
move `t1` to a y above 60.

- [ ] **Step 8: Verify no page map entered the session**

```bash
curl -s localhost:8003/sessions/<session_id_from_above> | python3 -m json.tool
```

Expected: the user message reads `move the title above the photo` with
no page block.

- [ ] **Step 9: Commit**

```bash
git add poc/chatbot/main.py poc/chatbot/tests/test_canvas_endpoint.py
git commit -m "feat(api): run canvas turns with page tools and on-demand vision

Refs #38"
```

---

### Task 6: Open the PR

- [ ] **Step 1: Run the full check set**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v
PYTHONPATH=. python3 -m pytest poc/image-generation/tests -v
```

Expected: all pass.

- [ ] **Step 2: Run the repo rule checks**

From the `.github` checkout:

```bash
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji <path-to>/miranote-api
```

Expected: no new violations from `poc/chatbot/` or
`poc/image-generation/main.py`.

- [ ] **Step 3: Open the PR**

Use the create-pr skill. Base `main`, never stacked. Title must be a
Conventional Commit under 72 chars with a whitelisted scope, e.g.
`feat(api): add canvas mode with page-edit tools to chat`. Body must
contain `Closes #38`.

---

## Notes for the reviewer

- **The one thing to check hardest** is that no page map reaches
  session history (Task 3, Task 5 Step 8). Everything else degrades
  gracefully; this one makes Mira confidently edit the wrong element
  several turns later.
- `look_at_page` failing is not a turn failure anywhere in this plan.
  If you find a path where it is, that is a bug.
- The three handoff tools must stay handoffs. If the server ever
  starts generating a background itself, the 150s image budget will
  blow the chat turn.

from __future__ import annotations

from poc.chatbot import canvas, journal, tools


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


def test_restyle_photo_joins_the_canvas_tools():
    names = {t["function"]["name"] for t in canvas.canvas_tools(tools.TOOLS)}
    assert "restyle_photo" in names
    assert "restyle_photo" in canvas.CANVAS_TOOL_NAMES


def test_restyle_photo_names_an_element_and_an_instruction():
    tool = next(
        t for t in canvas.canvas_tools(tools.TOOLS)
        if t["function"]["name"] == "restyle_photo"
    )
    params = tool["function"]["parameters"]
    assert set(params["properties"]) == {"id", "instruction"}
    assert params["required"] == ["id", "instruction"]


def test_restyle_photo_description_says_it_is_about_looks_not_placement():
    tool = next(
        t for t in canvas.canvas_tools(tools.TOOLS)
        if t["function"]["name"] == "restyle_photo"
    )
    described = tool["function"]["description"].lower()
    # The whole point of the tool: it changes PIXELS. Moving and resizing
    # belong to edit_page, and confusing the two is the bug class this
    # canvas work keeps running into.
    assert "edit_page" in described

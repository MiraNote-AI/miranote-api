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

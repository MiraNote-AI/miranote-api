from __future__ import annotations

from poc.chatbot import canvas


class _StubImageClient:
    def __init__(self):
        self.calls = 0
        self.asked = None

    def describe(self, image, prompt):
        self.calls += 1
        self.asked = prompt
        return "a calm page with room at the bottom"


def _refuse(name, args):
    raise AssertionError("canvas tools must not reach the docs dispatcher: " + name)


def _dispatcher(image_bytes, image_client, fallback=_refuse):
    # Lives in canvas.py, not main.py: main pulls fastapi, which the
    # test environment does not have -- every chatbot test avoids it.
    return canvas.build_dispatcher(image_bytes, image_client, fallback)


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


def test_look_at_page_asks_about_the_page_not_a_photo():
    client = _StubImageClient()
    _dispatcher(b"jpeg", client)("look_at_page", {})
    assert "page" in client.asked.lower()


def test_each_turn_gets_a_fresh_look_budget():
    client = _StubImageClient()
    _dispatcher(b"jpeg", client)("look_at_page", {})
    _dispatcher(b"jpeg", client)("look_at_page", {})
    assert client.calls == 2


def test_look_at_page_reports_it_could_not_look_without_an_image():
    dispatch = _dispatcher(None, _StubImageClient())
    assert "could not look" in dispatch("look_at_page", {})["status"]


def test_look_at_page_survives_a_failing_vision_call():
    class Broken:
        def describe(self, image, prompt):
            raise RuntimeError("gemini is down")

    dispatch = _dispatcher(b"jpeg", Broken())
    assert "could not look" in dispatch("look_at_page", {})["status"]


def test_non_canvas_tools_fall_through_to_the_ordinary_dispatcher():
    seen = []

    def fallback(name, args):
        seen.append(name)
        return {"ok": True}

    dispatch = _dispatcher(b"jpeg", _StubImageClient(), fallback)
    assert dispatch("polish_text", {"text": "hi"}) == {"ok": True}
    assert seen == ["polish_text"]


def test_restyle_photo_is_a_pure_handoff():
    dispatch = _dispatcher(b"jpeg", _StubImageClient())
    result = dispatch("restyle_photo", {"id": "p1", "instruction": "warmer"})
    assert "handed" in result["status"]

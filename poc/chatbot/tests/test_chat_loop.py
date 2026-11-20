from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List
import json

import pytest

from poc.chatbot.chat_loop import run_turn, ChatTurnResult, ActionGuard
from poc.chatbot.session import SessionStore


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _resp(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeChatCompletions:
    def __init__(self, scripted_responses):
        self._scripted = list(scripted_responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted_responses):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(scripted_responses))


def test_no_tool_call_returns_text(tmp_path):
    client = FakeClient([_resp(_msg(content="hello back"))])
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client,
        session_store=store,
        session_id=None,
        user_message="hi",
        model="fake-model",
        tools=[],
        tool_dispatcher=lambda name, args: {"error": "no tools"},
        max_iterations=6,
        max_history=40,
        system_prompt="you are a helper",
    )
    assert isinstance(result, ChatTurnResult)
    assert result.reply == "hello back"
    assert result.tool_trace == []
    history = store.get(result.session_id)
    assert history[0] == {"role": "system", "content": "you are a helper"}
    assert history[1] == {"role": "user", "content": "hi"}
    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == "hello back"


def _tool_call(call_id, name, args_dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args_dict)),
    )


def test_single_tool_call_then_text():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("call_1", "search_docs", {"query": "apples"})])),
        _resp(_msg(content="found apples in alpha.md")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    captured = []

    def dispatcher(name, args):
        captured.append((name, args))
        return [{"path": "alpha.md", "line": 3, "snippet": "Apples and oranges."}]

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="find apples", model="fake", tools=[{"type": "function", "function": {"name": "search_docs"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "found apples in alpha.md"
    assert captured == [("search_docs", {"query": "apples"})]
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["name"] == "search_docs"
    assert result.tool_trace[0]["args"] == {"query": "apples"}
    history = store.get(result.session_id)
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"


def test_multi_step_tool_use():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "list_docs", {"subdir": "."})])),
        _resp(_msg(tool_calls=[_tool_call("c2", "read_doc", {"path": "alpha.md"})])),
        _resp(_msg(content="alpha says hello")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    def dispatcher(name, args):
        return {"ok": True, "name": name}

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="look around", model="fake", tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "alpha says hello"
    assert [t["name"] for t in result.tool_trace] == ["list_docs", "read_doc"]


def test_dispatcher_errors_are_wrapped_for_model():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "read_doc", {"path": "../escape"})])),
        _resp(_msg(content="oh that was rejected")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    def dispatcher(name, args):
        raise ValueError("path outside DOCS_ROOT")

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="read escape", model="fake", tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "oh that was rejected"
    history = store.get(result.session_id)
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert "error" in tool_msgs[0]["content"]
    assert "outside" in tool_msgs[0]["content"]


def test_iteration_cap_returns_synthetic_reply():
    # 8 scripted responses, but cap = 3 -> loop should give up at 3.
    scripted = [_resp(_msg(tool_calls=[_tool_call(f"c{i}", "list_docs", {"subdir": "."})])) for i in range(8)]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="loop forever", model="fake",
        tools=[{"type": "function", "function": {"name": "list_docs"}}],
        tool_dispatcher=lambda n, a: [],
        max_iterations=3, max_history=100, system_prompt="sys",
    )
    assert "stopped" in result.reply.lower()
    assert "max_tool_iterations" in result.reply.lower()
    # The fake should have been called exactly 3 times.
    assert len(client.chat.completions.calls) == 3
    assert len(result.tool_trace) == 3


def test_thinking_mode_reasoning_content_preserved():
    # DeepSeek v4-flash returns reasoning_content alongside tool_calls and
    # 400s the follow-up turn if we drop it. The assistant entry written to
    # history must round-trip it.
    msg_with_reasoning = SimpleNamespace(
        content=None,
        tool_calls=[_tool_call("c1", "list_docs", {"subdir": "."})],
        reasoning_content="I should list the docs first.",
    )
    scripted = [_resp(msg_with_reasoning), _resp(_msg(content="all done"))]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="x", model="fake",
        tools=[{"type": "function", "function": {"name": "list_docs"}}],
        tool_dispatcher=lambda n, a: [],
        max_iterations=6, max_history=40, system_prompt="sys",
    )
    history = store.get(result.session_id)
    assistant_with_tools = [m for m in history if m.get("role") == "assistant" and m.get("tool_calls")][0]
    assert assistant_with_tools["reasoning_content"] == "I should list the docs first."


def test_empty_content_retries_then_lands():
    scripted = [
        _resp(_msg(content="")),
        _resp(_msg(content=None)),
        _resp(_msg(content="here at last")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client,
        session_store=store,
        session_id=None,
        user_message="hi",
        model="fake-model",
        tools=[],
        tool_dispatcher=lambda name, args: {},
        max_iterations=6,
        max_history=40,
        system_prompt="helper",
    )
    assert result.reply == "here at last"
    history = store.get(result.session_id)
    assert [m["role"] for m in history] == ["system", "user", "assistant"], "empty tries append nothing"


def test_drop_unrenderable_strips_pua_and_replacement_chars():
    from poc.chatbot.chat_loop import drop_unrenderable

    dirty = "Hi! " + chr(0xE5D4) + chr(0xFFFD) + chr(0x0007) + "there"
    assert drop_unrenderable(dirty) == "Hi! there"


def test_drop_unrenderable_keeps_emoji_cjk_and_newlines():
    from poc.chatbot.chat_loop import drop_unrenderable

    # wave emoji + sun-with-VS16 + Chinese + newline all render fine on
    # clients and must pass through untouched
    clean = (
        "Hi " + chr(0x1F44B) + "\n"
        + chr(0x2600) + chr(0xFE0F) + " "
        + chr(0x4F60) + chr(0x597D)
    )
    assert drop_unrenderable(clean) == clean


def _last_user_content(call):
    """The user message as the model saw it on this call.

    Positional indexing does not work here: the kwargs a fake records
    hold a REFERENCE to the live history list, which the loop keeps
    appending to, so [-1] is whatever landed last, not the user turn.
    """
    return [m for m in call["messages"] if m.get("role") == "user"][-1]["content"]


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

    sent = _last_user_content(client.chat.completions.calls[0])
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
    assert _last_user_content(client.chat.completions.calls[0]) == "hello"
def test_wrong_language_reply_gets_one_corrective_rewrite():
    zh = chr(0x6839) + chr(0x636E) + chr(0x4F60) + chr(0x7684)  # "based on your"
    scripted = [
        _resp(_msg(content=zh + " pages...")),
        _resp(_msg(content="Based on your pages, it was July 12.")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client,
        session_store=store,
        session_id=None,
        user_message="composed message with notes",
        model="fake-model",
        tools=[],
        tool_dispatcher=lambda name, args: {},
        max_iterations=6,
        max_history=40,
        system_prompt="helper",
        language_ref="when did we talk about the sea?",
    )
    assert result.reply == "Based on your pages, it was July 12."
    correction = client.chat.completions.calls[1]["messages"][-1]
    assert correction["role"] == "user"
    assert "wrong language" in correction["content"]


def test_matching_language_gets_no_rewrite():
    scripted = [_resp(_msg(content="A fine English reply."))]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client,
        session_store=store,
        session_id=None,
        user_message="hello",
        model="fake-model",
        tools=[],
        tool_dispatcher=lambda name, args: {},
        max_iterations=6,
        max_history=40,
        system_prompt="helper",
    )
    assert result.reply == "A fine English reply."
    assert len(client.chat.completions.calls) == 1


def _guard(**overrides):
    defaults = {
        "look_tool": "look_at_page",
        "action_tools": ["edit_page", "restyle_photo"],
        "nudge": "you looked but did not act -- act now",
        "keep": "KEEP",
    }
    defaults.update(overrides)
    return ActionGuard(**defaults)


def test_a_look_without_an_action_gets_one_corrective_nudge():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content="Tidied up!")),
        _resp(_msg(tool_calls=[_tool_call("c2", "edit_page", {"changes": [{"id": "t1"}]})])),
        _resp(_msg(content="Done.")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    captured = []

    def dispatcher(name, args):
        captured.append((name, args))
        return {"status": "ok"}

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="tidy this page up", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == "Done."
    assert [name for name, _ in captured] == ["look_at_page", "edit_page"]
    history = store.get(result.session_id)
    nudges = [m for m in history if m.get("role") == "user" and "did not act" in m.get("content", "")]
    assert len(nudges) == 1


def test_a_look_without_an_action_that_was_only_a_question_keeps_its_reply():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content="It looks calm to me.")),
        _resp(_msg(content="It looks calm to me.")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="how does this page look", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == "It looks calm to me."
    assert len(client.chat.completions.calls) == 3


def test_a_change_after_a_look_needs_no_nudge():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(tool_calls=[_tool_call("c2", "edit_page", {"changes": [{"id": "t1"}]})])),
        _resp(_msg(content="Rearranged.")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="tidy this page up", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == "Rearranged."
    history = store.get(result.session_id)
    nudges = [m for m in history if m.get("role") == "user" and "did not act" in m.get("content", "")]
    assert nudges == []
    assert len(client.chat.completions.calls) == 3


def test_words_asked_for_after_a_look_survive_the_nudge():
    # S1-b (#121): the user asked for words, the model looked, then
    # answered with them. The nudge's reply makes no change, so the
    # words -- not the reply to the nudge -- are the answer.
    words = "The tide came in slow. We stayed until the light went gold."
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content=words)),
        _resp(_msg(content="KEEP")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="write me two sentences for this page", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == words
    history = store.get(result.session_id)
    assert history[-1] == {"role": "assistant", "content": words}
    assert all("did not act" not in (m.get("content") or "") for m in history)


def test_a_keep_reply_to_the_nudge_is_never_shown():
    words = "Two quiet lines for the page."
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content=words)),
        _resp(_msg(content="KEEP -- no change was asked for.")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="give me a caption", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == words
    assert len(client.chat.completions.calls) == 3
    history = store.get(result.session_id)
    assert history[-1] == {"role": "assistant", "content": words}
    assert all("KEEP" not in (m.get("content") or "") for m in history)


def test_a_described_change_is_still_nudged_into_a_real_one():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content="I moved the title to the top for you.")),
        _resp(_msg(tool_calls=[_tool_call("c2", "edit_page", {"changes": [{"id": "t1", "y": 0}]})])),
        _resp(_msg(content="The title is at the top now.")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="move the title to the top", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == "The title is at the top now."
    assert [t["name"] for t in result.tool_trace] == ["look_at_page", "edit_page"]
    sent = [m["content"] for m in client.chat.completions.calls[2]["messages"] if m.get("role") == "user"]
    assert "did not act" in sent[-1]


def test_running_out_of_iterations_after_the_nudge_keeps_the_first_reply():
    words = "A line for the page."
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content=words)),
        _resp(_msg(tool_calls=[_tool_call("c2", "look_at_page", {})])),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="write a line for this page", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=3, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == words
    history = store.get(result.session_id)
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool", "assistant"]



def test_tool_calls_answering_a_nudge_that_changed_nothing_are_dropped():
    # create_note is not a page change; a turn that keeps its first
    # reply must not hand the app a note the session never mentions.
    words = "A line for the page."
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content=words)),
        _resp(_msg(tool_calls=[_tool_call("c2", "create_note", {"title": "x"})])),
        _resp(_msg(content="KEEP")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="write a line for this page", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == words
    assert [t["name"] for t in result.tool_trace] == ["look_at_page"]


def test_keep_after_a_change_returns_the_first_reply():
    first = "I moved the title to the top."
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content=first)),
        _resp(_msg(tool_calls=[_tool_call("c2", "edit_page", {"changes": [{"id": "t1"}]})])),
        _resp(_msg(content="KEEP.")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="move the title to the top", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == first
    assert [t["name"] for t in result.tool_trace] == ["look_at_page", "edit_page"]


def _edit_after_the_nudge(user_message, first, final):
    """Look, answer in prose, get nudged, edit the page, then `final`."""
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content=first)),
        _resp(_msg(tool_calls=[_tool_call("c2", "edit_page", {"changes": [{"id": "t1", "size": 36}]})])),
        _resp(_msg(content=final)),
    ]
    client = FakeClient(scripted)
    result = run_turn(
        client=client, session_store=SessionStore(ttl_seconds=60), session_id=None,
        user_message=user_message, model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert len(client.chat.completions.calls) == 4
    return result


@pytest.mark.parametrize("user_message, answer", [
    # Page text in capitals is the user's own words, not the keep word.
    ("make KEEP GOING bigger", "Done -- KEEP GOING is 36pt now and anchors the page."),
    ("make KEEP GOING bigger", "KEEP GOING is 36pt now."),
    ("make the KEEP sticker bigger", "I made the 'KEEP' sticker bigger."),
    ("make the title bigger", "Keep calm: the title is 36pt now."),
    ("make the title bigger", "The keeper photo sits behind the 36pt title now."),
    # zh: 'make KEEP bigger' / 'Made "KEEP" 36pt.'
    ("\u628aKEEP\u8c03\u5927", "\u5df2\u628a\u201cKEEP\u201d\u8c03\u5230 36pt\u3002"),
])
def test_a_real_answer_after_the_nudge_is_shown_even_when_it_says_keep(user_message, answer):
    result = _edit_after_the_nudge(user_message, "I'd make it 36pt so it anchors the page.", answer)
    assert result.reply == answer


@pytest.mark.parametrize("keep_reply", [
    "KEEP", "KEEP.", "Keep.", "keep", "**KEEP**", "`KEEP`",
    "KEEP -- no change was asked for.", "Made the change. KEEP",
    "\u597d\u7684\uff0cKEEP",  # zh: 'OK, KEEP'
])
def test_the_keep_word_after_an_edit_is_never_shown(keep_reply):
    first = "I'd make the title 36pt so it anchors the page."
    result = _edit_after_the_nudge("make the title bigger", first, keep_reply)
    assert result.reply == first
    assert [t["name"] for t in result.tool_trace] == ["look_at_page", "edit_page"]


def test_a_claimed_change_the_model_cannot_make_is_not_shown_as_done():
    # The first reply claims a change; after the nudge the model says it
    # cannot make it. Its honest answer stands, not the false claim.
    honest = "Sorry -- I can't change fonts. I can move, resize, recolor or layer the title."
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content="I changed the title to a serif font.")),
        _resp(_msg(content=honest)),
    ]
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=FakeClient(scripted), session_store=store, session_id=None,
        user_message="change the title font to serif", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == honest
    assert store.get(result.session_id)[-1] == {"role": "assistant", "content": honest}


def test_a_note_made_in_answer_to_the_nudge_reaches_the_app():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "look_at_page", {})])),
        _resp(_msg(content="Here is a summary of your page: a beach day.")),
        _resp(_msg(tool_calls=[_tool_call("c2", "create_note", {"title": "Beach", "body": "A beach day."})])),
        _resp(_msg(content="I saved it as a new note.")),
    ]
    result = run_turn(
        client=FakeClient(scripted), session_store=SessionStore(ttl_seconds=60), session_id=None,
        user_message="save a note about this page", model="fake",
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=lambda name, args: {"status": "ok"},
        max_iterations=6, max_history=40,
        system_prompt="sys", action_guard=_guard(),
    )
    assert result.reply == "I saved it as a new note."
    assert [t["name"] for t in result.tool_trace] == ["look_at_page", "create_note"]

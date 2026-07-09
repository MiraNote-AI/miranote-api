from __future__ import annotations

from poc.chatbot import journal, tools


def test_journal_tools_drop_docs_but_keep_text_and_quotes():
    names = {t["function"]["name"] for t in journal.journal_tools(tools.TOOLS)}
    assert names.isdisjoint(journal.DOCS_TOOL_NAMES)
    assert {"polish_text", "expand_text", "find_quote", "create_note"} <= names


def test_create_note_dispatch_is_a_pure_handoff(tmp_path):
    from poc.chatbot.config import ChatbotConfig

    config = ChatbotConfig(
        docs_root=tmp_path,
        model="fake",
        max_tool_iterations=6,
        max_history_messages=40,
        session_ttl_seconds=3600,
    )
    result = tools.dispatch(config, "create_note", {"title": "t", "body": "b"})
    assert result["status"].startswith("draft handed")


def test_render_notes_formats_title_date_body():
    block = journal.render_notes(
        [
            {"title": "Noodle shop by the bridge", "date": "2026-06-30", "body": "warm broth"},
            {"title": "", "body": "", "date": ""},
        ]
    )
    assert '- "Noodle shop by the bridge" (2026-06-30): warm broth' in block
    assert '- "Untitled"' in block
    assert block.startswith("[Pages from the user's own MiraNote library")
    assert block.endswith("[End of pages]")


def test_render_notes_caps_count_and_body():
    notes = [{"title": f"n{i}", "body": "x" * 2000, "date": ""} for i in range(20)]
    block = journal.render_notes(notes)
    assert block.count('- "n') == journal.MAX_NOTES
    assert "x" * (journal.MAX_BODY_CHARS + 1) not in block


def test_compose_user_message_states_empty_match_list():
    composed = journal.compose_user_message("any noodles?", [])
    assert composed.startswith("[No pages in the user's MiraNote library matched")
    assert composed.endswith("any noodles?")


def test_compose_user_message_prepends_block():
    composed = journal.compose_user_message(
        "when was it?", [{"title": "Paris", "date": "2026-05-01", "body": "rain"}]
    )
    assert composed.index("[Pages from") < composed.index("when was it?")

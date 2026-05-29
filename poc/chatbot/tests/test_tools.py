from __future__ import annotations
from pathlib import Path

from poc.chatbot import tools


def test_tools_schema_lists_three_functions():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert sorted(names) == ["list_docs", "read_doc", "search_docs"]
    for t in tools.TOOLS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_dispatch_routes_to_list_docs(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "list_docs", {"subdir": "."})
    assert isinstance(out, list)
    assert any(item["path"] == "alpha.md" for item in out)


def test_dispatch_routes_to_read_doc(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "read_doc", {"path": "alpha.md"})
    assert "First doc body." in out["content"]


def test_dispatch_routes_to_search_docs(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "search_docs", {"query": "apples"})
    assert len(out) >= 1


def test_dispatch_unknown_tool_returns_error(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "nope", {})
    assert "error" in out
    assert "unknown tool" in out["error"].lower()


def test_dispatch_wraps_exceptions(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "read_doc", {"path": "../escape.md"})
    assert "error" in out
    assert "outside" in out["error"].lower()


def test_dispatch_missing_required_arg(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "search_docs", {})
    assert "error" in out

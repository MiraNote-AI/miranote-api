from __future__ import annotations
from pathlib import Path

import pytest

from poc.chatbot import tools
from poc.chatbot.config import ChatbotConfig


def _cfg(docs_root: Path) -> ChatbotConfig:
    return ChatbotConfig(
        docs_root=docs_root,
        model="fake",
        max_tool_iterations=6,
        max_history_messages=40,
        session_ttl_seconds=3600,
    )


def test_tools_schema_lists_four_functions():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert sorted(names) == ["list_docs", "read_doc", "search_docs", "set_docs_root"]
    for t in tools.TOOLS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_dispatch_routes_to_list_docs(tmp_docs: Path):
    out = tools.dispatch(_cfg(tmp_docs), "list_docs", {"subdir": "."})
    assert isinstance(out, list)
    assert any(item["path"] == "alpha.md" for item in out)


def test_dispatch_routes_to_read_doc(tmp_docs: Path):
    out = tools.dispatch(_cfg(tmp_docs), "read_doc", {"path": "alpha.md"})
    assert "First doc body." in out["content"]


def test_dispatch_routes_to_search_docs(tmp_docs: Path):
    out = tools.dispatch(_cfg(tmp_docs), "search_docs", {"query": "apples"})
    assert len(out) >= 1


def test_dispatch_unknown_tool_returns_error(tmp_docs: Path):
    out = tools.dispatch(_cfg(tmp_docs), "nope", {})
    assert "error" in out
    assert "unknown tool" in out["error"].lower()


def test_dispatch_wraps_exceptions(tmp_docs: Path):
    out = tools.dispatch(_cfg(tmp_docs), "read_doc", {"path": "../escape.md"})
    assert "error" in out
    assert "outside" in out["error"].lower()


def test_dispatch_missing_required_arg(tmp_docs: Path):
    out = tools.dispatch(_cfg(tmp_docs), "search_docs", {})
    assert "error" in out


def test_dispatch_set_docs_root_mutates_config(tmp_docs: Path, tmp_path: Path):
    cfg = _cfg(tmp_docs)
    new_dir = tmp_path / "newdocs"
    new_dir.mkdir()
    out = tools.dispatch(cfg, "set_docs_root", {"path": str(new_dir)})
    assert out["docs_root"] == str(new_dir.resolve())
    assert cfg.docs_root == new_dir.resolve()
    # Subsequent fs calls now use the new root.
    (new_dir / "fresh.md").write_text("from the new root", encoding="utf-8")
    listed = tools.dispatch(cfg, "list_docs", {"subdir": "."})
    assert [item["path"] for item in listed] == ["fresh.md"]


def test_dispatch_set_docs_root_rejects_missing(tmp_docs: Path, tmp_path: Path):
    cfg = _cfg(tmp_docs)
    out = tools.dispatch(cfg, "set_docs_root", {"path": str(tmp_path / "nope")})
    assert "error" in out
    assert "does not exist" in out["error"]
    # Config was NOT mutated on failure.
    assert cfg.docs_root == tmp_docs.resolve()


def test_dispatch_set_docs_root_rejects_file(tmp_docs: Path, tmp_path: Path):
    cfg = _cfg(tmp_docs)
    f = tmp_path / "notadir.txt"
    f.write_text("hi")
    out = tools.dispatch(cfg, "set_docs_root", {"path": str(f)})
    assert "error" in out
    assert "not a directory" in out["error"]

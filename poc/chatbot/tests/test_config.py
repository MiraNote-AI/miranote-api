from __future__ import annotations
from pathlib import Path

import pytest

from poc.chatbot.config import ChatbotConfig


def _make(tmp_path: Path) -> ChatbotConfig:
    return ChatbotConfig(
        docs_root=tmp_path,
        model="fake",
        max_tool_iterations=6,
        max_history_messages=40,
        session_ttl_seconds=3600,
    )


def test_initial_docs_root_is_resolved(tmp_path: Path):
    cfg = _make(tmp_path)
    assert cfg.docs_root == tmp_path.resolve()


def test_set_docs_root_accepts_existing_dir(tmp_path: Path):
    cfg = _make(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    out = cfg.set_docs_root(str(other))
    assert cfg.docs_root == other.resolve()
    assert out == {"docs_root": str(other.resolve())}


def test_set_docs_root_rejects_missing(tmp_path: Path):
    cfg = _make(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        cfg.set_docs_root(str(tmp_path / "nope"))
    # Unchanged on failure.
    assert cfg.docs_root == tmp_path.resolve()


def test_set_docs_root_rejects_file(tmp_path: Path):
    cfg = _make(tmp_path)
    f = tmp_path / "x.txt"
    f.write_text("hi")
    with pytest.raises(ValueError, match="not a directory"):
        cfg.set_docs_root(str(f))


def test_set_docs_root_expands_tilde(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = _make(tmp_path)
    cfg.set_docs_root("~")
    assert cfg.docs_root == tmp_path.resolve()

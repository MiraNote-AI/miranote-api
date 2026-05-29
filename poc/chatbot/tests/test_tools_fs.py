from __future__ import annotations
from pathlib import Path
import pytest

from poc.chatbot import tools_fs


def test_resolve_path_accepts_relative_inside_root(tmp_docs: Path):
    p = tools_fs._resolve_path(tmp_docs, "alpha.md")
    assert p == tmp_docs / "alpha.md"


def test_resolve_path_accepts_nested(tmp_docs: Path):
    p = tools_fs._resolve_path(tmp_docs, "nested/gamma.md")
    assert p == tmp_docs / "nested" / "gamma.md"


def test_resolve_path_rejects_parent_escape(tmp_docs: Path):
    with pytest.raises(ValueError, match="outside"):
        tools_fs._resolve_path(tmp_docs, "../escape.md")


def test_resolve_path_rejects_absolute_outside(tmp_docs: Path):
    with pytest.raises(ValueError, match="outside"):
        tools_fs._resolve_path(tmp_docs, "/etc/passwd")

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


def test_list_docs_root_returns_top_level(tmp_docs):
    out = tools_fs.list_docs(tmp_docs, ".")
    paths = sorted(item["path"] for item in out)
    assert paths == ["alpha.md", "beta.md", "nested/gamma.md"]
    for item in out:
        assert isinstance(item["size_bytes"], int) and item["size_bytes"] > 0


def test_list_docs_subdir(tmp_docs):
    out = tools_fs.list_docs(tmp_docs, "nested")
    paths = [item["path"] for item in out]
    assert paths == ["nested/gamma.md"]


def test_list_docs_rejects_escape(tmp_docs):
    with pytest.raises(ValueError, match="outside"):
        tools_fs.list_docs(tmp_docs, "../")


def test_read_doc_returns_content(tmp_docs):
    out = tools_fs.read_doc(tmp_docs, "alpha.md")
    assert out["path"] == "alpha.md"
    assert "First doc body." in out["content"]
    assert out["truncated"] is False


def test_read_doc_truncates_large_file(tmp_docs):
    big = tmp_docs / "big.md"
    big.write_text("x" * (40 * 1024), encoding="utf-8")  # 40 KB
    out = tools_fs.read_doc(tmp_docs, "big.md")
    assert len(out["content"].encode("utf-8")) <= 32 * 1024
    assert out["truncated"] is True


def test_read_doc_rejects_escape(tmp_docs):
    with pytest.raises(ValueError, match="outside"):
        tools_fs.read_doc(tmp_docs, "../escape.md")


def test_read_doc_missing_file(tmp_docs):
    with pytest.raises(FileNotFoundError):
        tools_fs.read_doc(tmp_docs, "nope.md")

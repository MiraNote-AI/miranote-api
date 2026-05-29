"""Read-only file-system tools sandboxed under a docs root.

Every public function takes `docs_root: Path` so the module is pure and easy
to test. The dispatcher in `tools.py` binds the active `DOCS_ROOT` at call
time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

_MAX_FILES = 200
_MAX_BYTES = 32 * 1024


def _resolve_path(docs_root: Path, rel_or_abs: str) -> Path:
    """Resolve a user-supplied path and reject anything outside docs_root.

    Raises ValueError if the resolved path escapes the sandbox.
    """
    root = docs_root.resolve()
    candidate = (root / rel_or_abs).resolve() if not Path(rel_or_abs).is_absolute() else Path(rel_or_abs).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(f"path '{rel_or_abs}' resolves outside DOCS_ROOT") from e
    return candidate


def list_docs(docs_root: Path, subdir: str = ".") -> List[Dict[str, object]]:
    """Recursively list files under docs_root/subdir.

    Returns at most _MAX_FILES entries, each {"path": str (relative to docs_root), "size_bytes": int}.
    Hidden files and directories (leading dot) are skipped.
    """
    root = docs_root.resolve()
    start = _resolve_path(docs_root, subdir)
    if not start.exists() or not start.is_dir():
        raise ValueError(f"subdir '{subdir}' is not a directory under DOCS_ROOT")
    out: List[Dict[str, object]] = []
    for p in sorted(start.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if not p.is_file():
            continue
        out.append({
            "path": p.relative_to(root).as_posix(),
            "size_bytes": p.stat().st_size,
        })
        if len(out) >= _MAX_FILES:
            break
    return out


def read_doc(docs_root: Path, path: str) -> Dict[str, object]:
    """Read a UTF-8 text file under docs_root. Truncates to 32 KB."""
    target = _resolve_path(docs_root, path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"file '{path}' not found under DOCS_ROOT")
    raw = target.read_bytes()
    truncated = len(raw) > _MAX_BYTES
    head = raw[:_MAX_BYTES]
    # Decode safely; replace any partial multibyte at the boundary.
    content = head.decode("utf-8", errors="replace")
    return {
        "path": target.relative_to(docs_root.resolve()).as_posix(),
        "content": content,
        "truncated": truncated,
    }

"""Read-only file-system tools sandboxed under a docs root.

Every public function takes `docs_root: Path` so the module is pure and easy
to test. The dispatcher in `tools.py` binds the active `DOCS_ROOT` at call
time.
"""
from __future__ import annotations

from pathlib import Path


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

from __future__ import annotations
from pathlib import Path
import pytest


@pytest.fixture
def tmp_docs(tmp_path: Path) -> Path:
    """A temporary DOCS_ROOT with three small markdown files."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "alpha.md").write_text("# Alpha\nFirst doc body.\nApples and oranges.\n", encoding="utf-8")
    (root / "beta.md").write_text("# Beta\nSecond doc.\nBananas only.\n", encoding="utf-8")
    sub = root / "nested"
    sub.mkdir()
    (sub / "gamma.md").write_text("# Gamma\nNested doc body.\nApples again.\n", encoding="utf-8")
    return root

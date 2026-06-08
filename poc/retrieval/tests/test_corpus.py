from __future__ import annotations
import json
from pathlib import Path

import pytest


CORPUS_DIR = Path(__file__).parent.parent / "corpus"
THEMES = {
    "love", "loss", "grief", "hope", "despair", "loneliness",
    "friendship", "family", "nostalgia", "time", "nature", "seasons",
    "home", "travel", "work", "ambition", "rest", "joy", "gratitude",
    "regret", "turning-point", "perseverance", "solitude", "change",
    "farewell",
}


def _load(name: str):
    f = CORPUS_DIR / name
    if not f.exists():
        pytest.skip(f"{name} not present (corpus not built yet)")
    return json.loads(f.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,expected_lang", [
    ("quotes_en.json", "en"),
    ("quotes_zh.json", "zh"),
])
def test_corpus_file_well_formed(name, expected_lang):
    data = _load(name)
    assert isinstance(data, list)
    assert len(data) >= 100, f"{name} should have at least 100 entries"
    ids = set()
    for entry in data:
        assert isinstance(entry, dict)
        for key in ("id", "text", "author", "source", "lang", "themes"):
            assert key in entry, f"missing {key} in {entry}"
        assert entry["lang"] == expected_lang
        assert entry["id"] not in ids, f"duplicate id {entry['id']}"
        ids.add(entry["id"])
        assert isinstance(entry["text"], str) and entry["text"].strip()
        assert isinstance(entry["themes"], list)
        for t in entry["themes"]:
            assert t in THEMES, f"unknown theme {t!r} in {entry['id']}"


def test_no_duplicate_text_across_corpus():
    seen = {}
    for name in ("quotes_en.json", "quotes_zh.json"):
        f = CORPUS_DIR / name
        if not f.exists():
            continue
        for entry in json.loads(f.read_text(encoding="utf-8")):
            text = entry["text"]
            if text in seen:
                pytest.fail(f"duplicate text across files: {text[:40]!r} (both {seen[text]} and {entry['id']})")
            seen[text] = entry["id"]

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


def test_zh_author_diversity():
    data = _load("quotes_zh.json")
    counts = {}
    for entry in data:
        counts[entry["author"]] = counts.get(entry["author"], 0) + 1
    top_author, top = max(counts.items(), key=lambda kv: kv[1])
    assert top <= len(data) * 0.10, (
        f"author {top_author!r} has {top}/{len(data)} entries; "
        "no single author may exceed 10 percent of the zh corpus"
    )


def test_zh_both_eras_present():
    data = _load("quotes_zh.json")
    eras = {}
    for entry in data:
        eras[entry.get("era")] = eras.get(entry.get("era"), 0) + 1
    assert eras.get("Tang", 0) >= 50, f"want >=50 Tang entries, got {eras}"
    assert eras.get("Song", 0) >= 50, f"want >=50 Song entries, got {eras}"


def test_zh_empty_theme_cap():
    data = _load("quotes_zh.json")
    empty = sum(1 for entry in data if not entry["themes"])
    assert empty <= len(data) * 0.05, (
        f"{empty}/{len(data)} zh entries have empty themes; "
        "cap is 5 percent -- failed tagging batches must be retried"
    )

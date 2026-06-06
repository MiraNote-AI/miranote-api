"""Offline corpus builder.

Reads source datasets from poc/retrieval/sources/ (gitignored) and emits
the curated corpus JSONs to poc/retrieval/corpus/. The LLM is used ONLY
to assign 1-3 theme tags from a fixed taxonomy; it NEVER generates or
edits the quote text or author -- those are copied verbatim from the
source datasets.

Sources:
  - Chinese: chinese-poetry/chinese-poetry (Tang poems + Song ci),
    discovered via the poet.tang.*.json / ci.song.*.json file-name globs
    so the (CJK) anthology directory names are never hard-coded here.
  - English: sources/en_raw.json, shape [{text, author, source}].

Usage:
    PYTHONPATH=../.. ./poc/retrieval/.venv/bin/python3 scripts/build_corpus.py \\
        --sources poc/retrieval/sources \\
        --out poc/retrieval/corpus \\
        --target-en 500 --target-zh 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


THEMES = [
    "love", "loss", "grief", "hope", "despair", "loneliness",
    "friendship", "family", "nostalgia", "time", "nature", "seasons",
    "home", "travel", "work", "ambition", "rest", "joy", "gratitude",
    "regret", "turning-point", "perseverance", "solitude", "change",
    "farewell",
]


TAG_SYSTEM = (
    "You assign theme tags to short quotes. You will receive a JSON array "
    "of N quote strings. Respond with ONLY a JSON array of N elements, in "
    "the same order; each element is an array of 1-3 tags chosen from this "
    "exact list (lowercase, exact spelling): " + ", ".join(THEMES) + ". "
    "Tag by emotional / semantic theme. No prose, no markdown fences."
)


def _zh_lines(
    sources_dir: Path, file_glob: str, era: str, source_key: str,
    max_files: int = 8,
) -> List[Dict[str, Any]]:
    """Pull 10-30 char lines from chinese-poetry files matching file_glob.

    file_glob is matched one level under the chinese-poetry repo (e.g.
    ``*/poet.tang.*.json``) so we never hard-code the CJK anthology
    directory names in this ASCII source file.
    """
    out: List[Dict[str, Any]] = []
    repo = sources_dir / "chinese-poetry"
    for f in sorted(repo.glob(file_glob))[:max_files]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for poem in data:
            for line in poem.get("paragraphs", []):
                line = line.strip()
                if 10 <= len(line) <= 30:
                    out.append({
                        "text": line,
                        "author": poem.get("author", ""),
                        "source": poem.get(source_key, ""),
                        "lang": "zh",
                        "era": era,
                    })
    return out


def load_zh_from_chinese_poetry(sources_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(_zh_lines(sources_dir, "*/poet.tang.*.json", "Tang", "title"))
    rows.extend(_zh_lines(sources_dir, "*/ci.song.*.json", "Song", "rhythmic"))
    return rows


def load_en_from_sources(sources_dir: Path) -> List[Dict[str, Any]]:
    """Read sources/en_raw.json into the candidate list shape."""
    f = sources_dir / "en_raw.json"
    if not f.exists():
        return []
    raw = json.loads(f.read_text(encoding="utf-8"))
    return [
        {
            "text": r["text"].strip(),
            "author": r.get("author", "anon"),
            "source": r.get("source", ""),
            "lang": "en",
        }
        for r in raw
        if r.get("text", "").strip()
    ]


def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = r["text"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _parse_tag_batch(raw: str, n: int) -> List[List[str]]:
    """Parse the LLM's batched tag response into n tag-lists.

    Falls back to empty tag-lists if the response is malformed or the
    length does not match -- empty themes are schema-valid, so a flaky
    tagging call degrades gracefully instead of failing the build.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        parsed = json.loads(s)
    except Exception:
        return [[] for _ in range(n)]
    if not isinstance(parsed, list) or len(parsed) != n:
        return [[] for _ in range(n)]
    out: List[List[str]] = []
    for item in parsed:
        if isinstance(item, list):
            out.append([t for t in item if isinstance(t, str) and t in THEMES][:3])
        else:
            out.append([])
    return out


def tag_themes_batched(
    client: OpenAI, model: str, rows: List[Dict[str, Any]], batch: int = 25,
) -> None:
    """Assign rows[i]['themes'] in place, batching LLM calls for speed."""
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = [r["text"] for r in chunk]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": TAG_SYSTEM},
                    {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
                ],
            )
            tags = _parse_tag_batch(resp.choices[0].message.content, len(chunk))
        except Exception as e:  # noqa: BLE001
            print(f"  batch at {i} tagging failed ({e}); leaving themes empty")
            tags = [[] for _ in chunk]
        for r, t in zip(chunk, tags):
            r["themes"] = t
        print(f"  tagged {min(i + batch, len(rows))}/{len(rows)}")


def assign_ids(rows: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    return [{**r, "id": f"{lang}_{i:04d}"} for i, r in enumerate(rows, start=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-en", type=int, default=500)
    parser.add_argument("--target-zh", type=int, default=500)
    parser.add_argument("--batch", type=int, default=25)
    args = parser.parse_args()

    load_dotenv(args.out.parent / ".env")
    key = os.getenv("LLM_API_KEY")
    if not key:
        sys.exit("LLM_API_KEY required")
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    print(f"Loading sources from {args.sources}")
    zh = dedupe(load_zh_from_chinese_poetry(args.sources))[: args.target_zh]
    en = dedupe(load_en_from_sources(args.sources))[: args.target_en]
    print(f"  ZH candidates: {len(zh)}, EN candidates: {len(en)}")

    args.out.mkdir(parents=True, exist_ok=True)

    print("Tagging ZH...")
    tag_themes_batched(client, model, zh, args.batch)
    zh = assign_ids(zh, "zh")
    (args.out / "quotes_zh.json").write_text(
        json.dumps(zh, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {len(zh)} entries to quotes_zh.json")

    print("Tagging EN...")
    tag_themes_batched(client, model, en, args.batch)
    en = assign_ids(en, "en")
    (args.out / "quotes_en.json").write_text(
        json.dumps(en, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {len(en)} entries to quotes_en.json")


if __name__ == "__main__":
    main()

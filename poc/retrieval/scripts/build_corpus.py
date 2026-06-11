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
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    max_files: int = 40,
) -> List[Dict[str, Any]]:
    """Pull 10-30 char lines from chinese-poetry files matching file_glob.

    file_glob is matched one level under the chinese-poetry repo (e.g.
    ``*/poet.tang.*.json``) so we never hard-code the CJK anthology
    directory names in this ASCII source file. Lines without an author
    attribution and lines containing the missing-character placeholder
    (U+25A1) are skipped -- both make poor product content.
    """
    out: List[Dict[str, Any]] = []
    repo = sources_dir / "chinese-poetry"
    for f in sorted(repo.glob(file_glob))[:max_files]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for poem in data:
            author = (poem.get("author") or "").strip()
            if not author:
                continue
            for line in poem.get("paragraphs", []):
                line = line.strip()
                # Skip corrupt or editorially annotated lines: U+25A1 is
                # the missing-character placeholder; ASCII brackets carry
                # upstream role markers like "[zhu]" that are not verse.
                if "□" in line or "[" in line or "]" in line:
                    continue
                if 10 <= len(line) <= 30:
                    out.append({
                        "text": line,
                        "author": author,
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


def _norm(text: str) -> str:
    """Lowercase and keep only alphanumerics + CJK -- a normalized key for
    duplicate detection that ignores punctuation / casing / whitespace."""
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)


def fuzzy_dedupe(
    rows: List[Dict[str, Any]], target: int, threshold: float = 0.82,
) -> List[Dict[str, Any]]:
    """Keep up to `target` rows, dropping exact-normalized duplicates and
    near-duplicates -- where the difflib similarity of the normalized text is
    >= threshold against an already-kept row (e.g. "Most people are about as
    happy..." vs "Most folks are as happy..."). Processes in order and stops
    once `target` is reached, so it stays cheap even on a large candidate pool.
    """
    kept: List[Dict[str, Any]] = []
    kept_norms: List[str] = []
    seen = set()
    for r in rows:
        key = _norm(r["text"])
        if not key or key in seen:
            continue
        if any(SequenceMatcher(None, key, k).ratio() >= threshold for k in kept_norms):
            continue
        seen.add(key)
        kept.append(r)
        kept_norms.append(key)
        if len(kept) >= target:
            break
    return kept


def _parse_tag_batch(raw: str, n: int) -> Optional[List[List[str]]]:
    """Parse the LLM's batched tag response into n tag-lists.

    Returns None when the response is malformed or the length does not
    match, so the caller can RETRY the batch instead of silently
    shipping empty themes (the pre-rebuild corpus had 75 empty entries
    from exactly that silent fallback).
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        parsed = json.loads(s)
    except Exception:
        return None
    if not isinstance(parsed, list) or len(parsed) != n:
        return None
    out: List[List[str]] = []
    for item in parsed:
        if isinstance(item, list):
            out.append([t for t in item if isinstance(t, str) and t in THEMES][:3])
        else:
            out.append([])
    # A right-length response where EVERY element filtered to empty is a
    # wrong-shape answer (strings instead of arrays, off-taxonomy tags),
    # not a judgment that nothing fits -- treat as unusable so the
    # caller retries instead of silently shipping an empty batch.
    if out and all(not tags for tags in out):
        return None
    return out


def tag_themes_batched(
    client: OpenAI, model: str, rows: List[Dict[str, Any]], batch: int = 25,
    attempts: int = 3,
) -> None:
    """Assign rows[i]['themes'] in place, batching LLM calls for speed.

    A batch whose call errors or whose response cannot be parsed is
    retried up to `attempts` times before degrading to empty themes.
    """
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = [r["text"] for r in chunk]
        tags: Optional[List[List[str]]] = None
        for attempt in range(1, attempts + 1):
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
                print(f"  batch at {i} attempt {attempt} errored: {e}")
                tags = None
            if tags is not None:
                break
            if attempt < attempts:
                print(f"  batch at {i} attempt {attempt} unusable; retrying")
        if tags is None:
            print(f"  batch at {i} failed {attempts} attempts; leaving themes empty")
            tags = [[] for _ in chunk]
        for r, t in zip(chunk, tags):
            r["themes"] = t
        print(f"  tagged {min(i + batch, len(rows))}/{len(rows)}")


def diverse_select(
    rows: List[Dict[str, Any]],
    target: int,
    seed: int = 42,
    author_cap_frac: float = 0.02,
    threshold: float = 0.82,
) -> List[Dict[str, Any]]:
    """Pick up to `target` rows with author and era diversity.

    The naive head-take filled the whole zh quota from the opening
    volumes of Quan Tang Shi (one author). This instead:
      - shuffles candidates with a FIXED seed (rebuilds are reproducible),
      - caps each author at max(2, target * author_cap_frac) entries,
      - balances eras toward an even split (a row whose era already holds
        its half of the quota is skipped while the other era still has
        candidates),
      - applies the same exact + fuzzy near-duplicate rejection as
        fuzzy_dedupe.
    """
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    author_cap = max(2, int(target * author_cap_frac))
    era_quota = target // 2
    kept: List[Dict[str, Any]] = []
    kept_norms: List[str] = []
    seen = set()
    by_author: Dict[str, int] = {}
    by_era: Dict[str, int] = {}

    def admit(r: Dict[str, Any], era_limit: Optional[int]) -> bool:
        if by_author.get(r["author"], 0) >= author_cap:
            return False
        if era_limit is not None and by_era.get(r.get("era", ""), 0) >= era_limit:
            return False
        key = _norm(r["text"])
        if not key or key in seen:
            return False
        if any(SequenceMatcher(None, key, k).ratio() >= threshold for k in kept_norms):
            return False
        seen.add(key)
        kept.append(r)
        kept_norms.append(key)
        by_author[r["author"]] = by_author.get(r["author"], 0) + 1
        by_era[r.get("era", "")] = by_era.get(r.get("era", ""), 0) + 1
        return True

    # Pass 1: balanced eras. Pass 2: top up from anything left if one
    # era could not fill its half.
    for r in shuffled:
        if len(kept) >= target:
            break
        admit(r, era_quota)
    if len(kept) < target:
        for r in shuffled:
            if len(kept) >= target:
                break
            admit(r, None)
    return kept


def assign_ids(rows: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    return [{**r, "id": f"{lang}_{i:04d}"} for i, r in enumerate(rows, start=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-en", type=int, default=500)
    parser.add_argument("--target-zh", type=int, default=500)
    parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--dedupe-threshold", type=float, default=0.82)
    parser.add_argument(
        "--langs", default="zh,en",
        help="comma list of corpora to rebuild (zh, en); others untouched",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    langs = {p.strip() for p in args.langs.split(",") if p.strip()}

    load_dotenv(args.out.parent / ".env")
    key = os.getenv("LLM_API_KEY")
    if not key:
        sys.exit("LLM_API_KEY required")
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    print(f"Loading sources from {args.sources}")
    args.out.mkdir(parents=True, exist_ok=True)

    if "zh" in langs:
        pool = load_zh_from_chinese_poetry(args.sources)
        print(f"  ZH candidate pool: {len(pool)}")
        zh = diverse_select(
            pool, args.target_zh, seed=args.seed,
            threshold=args.dedupe_threshold,
        )
        authors = len({r["author"] for r in zh})
        eras = {r["era"]: sum(1 for x in zh if x["era"] == r["era"]) for r in zh}
        print(f"  ZH selected: {len(zh)} ({authors} authors, eras {eras})")
        print("Tagging ZH...")
        tag_themes_batched(client, model, zh, args.batch)
        zh = assign_ids(zh, "zh")
        (args.out / "quotes_zh.json").write_text(
            json.dumps(zh, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {len(zh)} entries to quotes_zh.json")

    if "en" in langs:
        en = fuzzy_dedupe(load_en_from_sources(args.sources), args.target_en, args.dedupe_threshold)
        print(f"  EN candidates: {len(en)}")
        print("Tagging EN...")
        tag_themes_batched(client, model, en, args.batch)
        en = assign_ids(en, "en")
        (args.out / "quotes_en.json").write_text(
            json.dumps(en, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  wrote {len(en)} entries to quotes_en.json")


if __name__ == "__main__":
    main()

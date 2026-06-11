# Corpus sources

This directory holds the curated quote / poetry corpus used by the
retrieval POC. The JSON files here are the source of truth -- the
`scripts/build_corpus.py` script regenerates them from the datasets
under `sources/` (gitignored), but the JSON committed here is what
ships and what `build_index.py` embeds.

## Files

- `quotes_en.json` -- English quotes (500)
- `quotes_zh.json` -- Chinese classical poetry lines (500: 250 Tang shi
  + 250 Song ci, roughly 250 distinct authors)

The zh set is selected by `diverse_select` in the build script: a
fixed-seed shuffle over the full candidate pool (260k+ lines), a
per-author cap of 2 percent, an even era split, and the same fuzzy
near-duplicate rejection as the en path. Rebuilds with the same seed
are reproducible. (The first shipped build was a single-author
head-take; fixed by the diversity rebuild.)

## Per-entry schema

```json
{
  "id": "<lang>_<4-digit zero-padded>",
  "text": "the quote / line",
  "author": "attributed author",
  "source": "work / title / rhythmic name (may be empty)",
  "lang": "en | zh",
  "era": "Tang | Song",
  "themes": ["hope", "perseverance"]
}
```

`era` is present on Chinese entries only. `themes` holds 0-3 tags from
the fixed taxonomy. An empty list is schema-valid, but the build script
retries a failed tagging batch up to 3 attempts before degrading to
empty (the corpus test caps empties at 5 percent; the current build has
zero). Themes are metadata only: retrieval ranks by embedding
similarity, not by theme.

## Sources

| Dataset | License | URL | Pulled | Used for |
|---|---|---|---|---|
| chinese-poetry/chinese-poetry | MIT | https://github.com/chinese-poetry/chinese-poetry | 2026-06-05 | quotes_zh.json (Tang shi via `poet.tang.*`, Song ci via `ci.song.*`, 250 each) |
| dwyl/quotes | GPL-2.0 | https://github.com/dwyl/quotes | 2026-06-05 | quotes_en.json (English quotations) |

Note on the English set: the dwyl/quotes *compilation* is GPL-2.0; the
individual quotations are public-domain works of their attributed
(historical) authors. If a more permissive provenance is required, swap
`sources/en_raw.json` for another dataset and re-run `build_corpus.py` --
nothing else changes.

The build script copies `(text, author, source)` verbatim from the
sources; the LLM only assigns theme tags -- it never generates or edits
text or authors.

## How to rebuild

```bash
cd poc/retrieval
# 1. Stage sources (gitignored under sources/)
mkdir -p sources && cd sources
git clone --depth=1 https://github.com/chinese-poetry/chinese-poetry.git
# English: produce sources/en_raw.json as [{"text","author","source"}],
#   e.g. mapped from https://github.com/dwyl/quotes (quotes.json)
cd ..
# 2. Build the corpus JSONs (batched LLM theme-tagging).
#    --langs rebuilds one corpus without churning the other;
#    --seed keeps the zh diverse sampling reproducible (default 42).
PYTHONPATH=../.. .venv/bin/python3 scripts/build_corpus.py \
    --sources sources --out corpus --target-en 500 --target-zh 500 \
    --langs zh,en
```

## Rule 3

CJK is permitted in this directory via the `poc/*/corpus/*.json` and
`poc/*/corpus/*.md` allowlist entries (see `MiraNote-AI/.github`
`checks/no_cjk_or_emoji.py`).

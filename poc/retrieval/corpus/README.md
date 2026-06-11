# Corpus sources

This directory holds the curated quote / poetry corpus used by the
retrieval POC. The JSON files here are the source of truth -- the
`scripts/build_corpus.py` script regenerates them from the datasets
under `sources/` (gitignored), but the JSON committed here is what
ships and what `build_index.py` embeds.

## Files

- `quotes_en.json` -- English quotes (500)
- `quotes_zh.json` -- Chinese classical poetry lines (500, all Tang shi)

Known limitation of the current zh build: candidates are consumed in
file order from the head of Quan Tang Shi, so the 500-entry quota fills
before author variety appears -- the shipped set is single-author
(Emperor Taizong). Retrieval quality is unaffected (ranking is by
embedding similarity), but attribution variety is poor. Backlog:
sample across the whole dataset (and add Song ci) in a rebuild.

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
the fixed taxonomy. An empty list is schema-valid; note that 75 zh
entries currently have empty themes because three 25-entry tagging
batches failed during the build and were absorbed by the graceful
fallback rather than re-run (backlog: re-tag those batches). Themes are
metadata only: retrieval ranks by embedding similarity, not by theme.

## Sources

| Dataset | License | URL | Pulled | Used for |
|---|---|---|---|---|
| chinese-poetry/chinese-poetry | MIT | https://github.com/chinese-poetry/chinese-poetry | 2026-06-05 | quotes_zh.json (Tang shi via `poet.tang.*`; Song ci loading exists in the script but no ci lines made the current 500 cut) |
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
# 2. Build the corpus JSONs (batched LLM theme-tagging)
PYTHONPATH=../.. .venv/bin/python3 scripts/build_corpus.py \
    --sources sources --out corpus --target-en 500 --target-zh 500
```

## Rule 3

CJK is permitted in this directory via the `poc/*/corpus/*.json` and
`poc/*/corpus/*.md` allowlist entries (see `MiraNote-AI/.github`
`checks/no_cjk_or_emoji.py`).

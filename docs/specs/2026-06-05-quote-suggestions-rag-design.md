# Quote suggestions via local RAG -- POC design

- **Date:** 2026-06-05
- **Author:** mengjia (Claude-assisted)
- **Status:** Draft, awaiting implementation plan
- **Scope:** new POC at `miranote-api/poc/retrieval/`, plus integration
  touchpoints in `poc/text-clean-expand/` (UI) and `poc/chatbot/`
  (a new tool).
- **Reference:** the "quote suggestions" line item from meeting §3.2
  (`meeting_3_summary.md`, 2026-05-30) that was cut from the
  2026-06-05 text-features spec
  (`docs/specs/2026-06-05-text-features-and-voice-sentiment-design.md`
  §2 non-goals). This spec brings it back as its own work, deliberately
  scoped larger because we are taking the chance to lay down a reusable
  semantic-retrieval foundation for the project.
- **PRs go up in parallel** with the three open feature PRs (#10,
  #11, #12). Different files, different POCs -- no merge conflicts.
  Rebases handled as main moves.

## 1. Goal

Given a short piece of user text (a journal entry, a half-formed
thought, a feeling), suggest 1-3 famous quotes or pieces of poetry that
genuinely fit -- **never invented, never mis-attributed**. Bilingual
(English + Chinese, including Chinese classical poetry).

A second-order goal: build the retrieval layer **as reusable
infrastructure**, not a one-off, so the chatbot's future semantic
docs search and the web app's eventual note search ride on the same
embeddings + vector store + retriever pipeline.

## 2. Non-goals

- Verifying a quote via an external API at runtime (Wikiquote,
  Brainyquote). The trust model is "if it's in our curated corpus we
  vouch for it"; corpus curation is the verification stage.
- Hosted vector stores (Pinecone, Qdrant cloud). POC is local.
- Re-architecting the chatbot's existing `search_docs` substring tool
  in this PR. It stays as is; future PR can migrate it onto this
  retrieval layer.
- A frontend for managing the corpus. The corpus is a JSON file
  edited by hand or by an offline LLM-assisted script. No UI.
- Personalised recommendations (per-user quote preference, history,
  feedback signals). Future.
- Quote attribution disambiguation (the "Mark Twain probably never
  said it" Wikiquote rabbit hole). We accept attributions as they
  appear in our curated sources; mistakes there are fixed by editing
  the JSON.

## 3. Architecture

```
+-------------------+     +-------------------+
| text-clean-expand |     | chatbot           |
|   Quote button    |     |   find_quote tool |
| (UI Text tab)     |     |                   |
+--------+----------+     +---------+---------+
         |                          |
         |   HTTP (port 8004)       |
         +------------+-------------+
                      v
        +--------------------------------+
        |  poc/retrieval/                |
        |  POST /quotes                  |  business: quote recs
        |  POST /search                  |  generic: top-K hits
        |  GET  /health                  |
        +---------------+----------------+
                        |
        +---------------+----------------+
        |  Reranker (LLM, JSON output)   |  picks 1-3 from top-10
        +---------------+----------------+
                        |
        +---------------+----------------+
        |  Retriever                     |
        |   embedder.encode(query)       |
        |   store.search(vec, k)         |
        +---------------+----------------+
                        |
        +---------------+----------------+
        |  Embedder (BGE-M3, lazy load)  |
        |  Store    (sqlite-vec, file)   |
        +---------------+----------------+
                        |
        +---------------v----------------+
        |  corpus/quotes_en.json (500)   |
        |  corpus/quotes_zh.json (500)   |
        |  data/index.db   (embeddings)  |
        +--------------------------------+
```

Each box has one clear responsibility and is independently testable.
Replacing the embedder, the store, the reranker LLM, or the corpus
should be a one-file change.

## 4. Corpus

**Target size:** ~1,000 entries: ~500 English + ~500 Chinese.

**Source plan:**
- **English** (~500): mix of Wikiquote (CC-BY-SA, attribution required)
  and Project Gutenberg public-domain literature excerpts. Examples of
  appropriate sources: Marcus Aurelius (Meditations), Thoreau (Walden),
  Lao Tzu (Tao Te Ching English translations), Emerson, Whitman,
  Shakespeare. Modern quotes only if confidently verifiable from a
  primary source.
- **Chinese** (~500): Tang/Song poetry pulled from existing public
  datasets (e.g. `chinese-poetry/chinese-poetry` on GitHub, MIT-licensed
  Tang/Song collections). Includes shi, ci, sample selections from
  modern public-domain authors (Lu Xun, Zhu Ziqing, Lin Huiyin where
  copyright permits).

**Curation method -- fully automated:**
- An offline **selection script** ingests the source datasets, dedupes,
  filters by length and basic quality heuristics, and asks the LLM to
  pick the most journal-relevant entries and tag each with themes from
  the fixed taxonomy.
- **The LLM never invents text and never invents authors.** It only
  copies `(text, author, source)` triples straight from the dataset
  rows; the only generative step is the themes tag. The trust anchor
  is therefore the **source dataset**, not the LLM -- pick datasets
  with reliable attribution and the corpus inherits that reliability.
- Output committed as version-controlled JSON; future corpus edits
  go through PR (the script is re-runnable, but the JSON is the
  source of truth once committed).
- The script and its prompt live in `poc/retrieval/scripts/build_corpus.py`
  so the curation process is reproducible.

**Per-entry schema:**
```json
{
  "id": "zh_042",
  "text": "<the quote / line of poetry, as the original source has it>",
  "author": "<attributed author, exactly as primary source spells it>",
  "source": "<work or context, e.g. '游山西村' or 'Meditations, Book V'>",
  "lang": "zh" | "en",
  "era": "<optional, e.g. '宋', '20c'>",
  "themes": ["hope", "perseverance", ...]
}
```

**Theme taxonomy** (~25 fixed tags, English):
`love`, `loss`, `grief`, `hope`, `despair`, `loneliness`, `friendship`,
`family`, `nostalgia`, `time`, `nature`, `seasons`, `home`, `travel`,
`work`, `ambition`, `rest`, `joy`, `gratitude`, `regret`,
`turning-point`, `perseverance`, `solitude`, `change`, `farewell`.

Themes are an English-only controlled vocabulary regardless of the
quote's language; the LLM reranker reasons over them as a coarse
filter only -- the actual matching is done by embeddings.

**Provenance:**
`corpus/README.md` lists every source dataset with its license,
URL, and date of pull. Anything we cannot attribute confidently does
not enter the corpus.

## 5. Embedder and vector store

### 5.1 Embedder: `BAAI/bge-m3`

- 1024-dim multilingual dense embedding model from BAAI.
- Trained on 100+ languages; explicitly competitive on Chinese.
- Loaded via the `sentence-transformers` package:
  `SentenceTransformer("BAAI/bge-m3")`.
- ~1.3 GB, auto-downloads to `~/.cache/huggingface/` on first call.
  Same lazy-load pattern as the voice POC's emotion model.
- Configurable via env var `EMBEDDING_MODEL`; if a different model
  reads better on a target language we can swap with one line.

### 5.2 Store: `sqlite-vec`

- Single-file SQLite database with the `sqlite-vec` extension loaded.
- Schema: one virtual table indexed on the 1024-dim embedding column,
  plus a regular table for the JSON metadata, joined by `id`.
- Index size for 1,000 entries: ~4 MB on disk.
- File is gitignored at `poc/retrieval/data/index.db`; built by the
  `build_index.py` CLI script from the JSON corpus.
- Search latency: well under 10 ms for top-10 over 1,000 entries.
- Future swap target: `pgvector` (production); `chromadb` (if
  sqlite-vec proves wobbly during the POC). The `Store` interface
  is small enough that swapping is a one-file change.

## 6. API surface

### 6.1 `POST /quotes`

**Business endpoint.** Returns ranked quote suggestions with
explanations.

```
Request:
{
  "text": "<user's free text, journal snippet, feeling>",
  "max": <int, 1-5, default 3>,
  "lang": "auto" | "en" | "zh" | "both"   // default "auto"
}

Response (200):
{
  "matches": [
    {
      "text": "<quote>",
      "author": "<author>",
      "source": "<source>",
      "lang": "en" | "zh",
      "score": <float, 0-1, cosine sim>,
      "why": "<one sentence in user's input language>"
    },
    ...
  ]
}
```

`matches` is an empty array if the reranker decides nothing fits well
enough (no hard threshold; the LLM judges). Empty array is a normal
response, not an error -- the UI shows a friendly "nothing felt
right; try a different phrasing".

`lang` field semantics:
- `"auto"` (default): retrieval pulls from both languages; reranker
  picks whatever fits best.
- `"en"` / `"zh"`: restrict candidates to that language only.
- `"both"`: same as `"auto"` but reranker tries to return a mix
  (at most one per language up to `max`).

### 6.2 `POST /search`

**Generic endpoint.** Plain top-K semantic search over the corpus
(or a future named index). No LLM reranking; just raw retrieval.

```
Request:
{
  "query": "<query text>",
  "k": <int, 1-50, default 10>,
  "namespace": "quotes"          // reserved; only "quotes" today
}

Response (200):
{
  "hits": [
    { "id": "...", "text": "...", "score": <float>, "metadata": {...} },
    ...
  ]
}
```

`/quotes` is implemented as `/search` (k=10) followed by reranker.
Exposing `/search` separately lets future consumers (chatbot doc
retrieval, web app note search) plug in without going through quote-
specific logic.

### 6.3 `GET /health`

```
{
  "status": "ok",
  "embedder": "BAAI/bge-m3",
  "embedder_loaded": <bool, true after first request>,
  "store": "sqlite-vec",
  "corpus_size": 1000,
  "namespaces": ["quotes"]
}
```

## 7. Reranker

A single LLM call that takes the user's text plus the top-10 candidates
from the retriever and emits a structured JSON list of picks with
explanations.

**Prompt shape:**
```
SYSTEM: You re-rank candidate quotes for emotional / semantic fit with
        a user's text. Pick at most {max} of the {n} candidates that
        truly fit. For each pick, write a one-sentence "why" in the
        same language as the user's text. If none of the candidates
        feels right, return an empty array. NEVER invent quotes;
        NEVER edit text or author; only choose from the given list.

        Output strictly as JSON: [{"id": <int>, "why": "<sentence>"}].

USER:   Text: {user_text}

        Candidates:
        1. [{lang}] {text} -- {author}, {source}
        2. ...
        10. ...
```

Picks reference candidates by their **list index in the prompt** (1-10).
The server resolves these back to corpus entries to build the response
-- the LLM only sees and selects from candidates we provided. This
makes hallucination structurally impossible: a pick that doesn't map
to one of the 10 IDs is treated as a parse error.

**Error handling** (mirrors the `/keywords` endpoint pattern):
- LLM emits malformed JSON: `502` with the raw output truncated to 200
  chars in `detail` so we can debug.
- LLM picks an out-of-range index: same `502`.
- Reranker returns an empty array: pass through; UI handles.

**Cost / latency:** one ~500-800 token prompt, ~200-token JSON
response. ~400 ms with DeepSeek v4-flash, in line with the other text
endpoints.

## 8. Integration

### 8.1 text-clean-expand UI

The Text tab gets a 7th action in the dropdown: `Quote`. Unlike the
existing 6 actions which call text-clean-expand's own backend, `Quote`
calls **`http://localhost:8004/quotes`** cross-origin (CORS allow-all
on the retrieval server).

UI sub-controls under the action row when `Quote` is selected:
- `Lang:` radio with options `auto` / `en` / `zh` / `both` (default `auto`)
- `Max:` number input (default 3)

Result rendering: a vertical list of cards (one per match), each card
showing:
- The quote text (larger font, italic, with quote marks)
- A muted footer: `-- {author}, {source} | match: {score%}`
- A small "why" row underneath in muted text

If `matches: []`, render a friendly empty-state ("Nothing felt quite
right -- try rephrasing or adding context.").

### 8.2 chatbot tool

New tool `find_quote(text, max?)` added to the chatbot's tool registry.
HTTP-delegates to `http://localhost:8004/quotes` exactly like the
existing six text tools delegate to text-clean-expand via `TextClient`.

- New env var `RETRIEVAL_API_URL` (default `http://localhost:8004`).
- New module `poc/chatbot/retrieval_client.py` -- a tiny httpx wrapper
  mirroring `text_client.py`; or extend `text_client.py` if that feels
  natural. (Implementation plan will pick; either is fine.)
- System prompt update: one sentence telling the agent that when the
  user expresses a feeling or mood, `find_quote` is appropriate.
- Tool description with bilingual trigger phrases ("quote", "诗句",
  "名言", "find me a quote about...").

The chatbot's other tools (fs, text transforms) are unchanged.

## 9. Testing strategy

Following the chatbot/text-clean-expand pattern: pytest with stubbed
externals (no real embeddings, no real LLM, no real disk).

- **`tests/test_embedder.py`**: stub
  `sentence_transformers.SentenceTransformer` to return a fixed vector.
  Assert lazy load (model class only instantiated on first call),
  caching (subsequent calls don't re-instantiate), shape is 1024.
- **`tests/test_store.py`**: in-memory sqlite-vec store. Insert known
  vectors, search with a probe vector, assert order by distance.
- **`tests/test_retriever.py`**: stub embedder + store; verify the
  pipeline composes correctly.
- **`tests/test_reranker.py`**: stub LLM client returning scripted
  JSON. Cover happy path, malformed JSON (502), out-of-range index
  (502), empty pick array (passes through).
- **`tests/test_api.py`**: FastAPI `TestClient` with the whole
  retriever stubbed. Verify /quotes shape, /search shape, lang
  filtering, /health shape.
- **`tests/test_corpus.py`**: validate `quotes_en.json` and
  `quotes_zh.json` against the per-entry schema (all required fields,
  themes from taxonomy, no duplicate IDs). Runs against the real
  corpus, no stubs.

Manual smoke before the PR: real BGE-M3 + real LLM + 20 hand-written
test queries (10 EN, 10 ZH) with expected top-1 hits documented. If
hit rate is below ~80%, escalate (swap embedder or expand corpus).

## 10. PR plan

Two PRs, sequenced:

**PR α: `feat/api-retrieval-poc`** -- everything under `poc/retrieval/`.
Self-contained. The corpus JSONs, embedder, store, retriever, reranker,
API, tests, README. Lands first.

**PR β: `feat/api-quote-integration`** -- the text-clean-expand UI
Quote button + the chatbot `find_quote` tool. Depends on PR α being
merged so the smoke test against a live retrieval server is possible.
Branched off PR α's tip locally; rebased onto main after α merges.

Cut order: none. Both required.

## 11. Conventions to honor

Mirror the patterns already established in this repo:
- Python 3.9 compat: `from __future__ import annotations` +
  `typing.Optional/List/Dict`. No PEP-604 `X | None`.
- Rule 3 (no CJK in source). The corpus JSONs are the bilingual
  surface. The current allowlist (see `dotgithub/checks/no_cjk_or_emoji.py`
  `ALLOWLIST_PATTERNS`) covers `**/prompts/*.txt`, `**/static/*`,
  `**/demo_data/*`, and `poc/*/README.md` -- none of which match
  `poc/retrieval/corpus/*.json`. **Phase 0 of PR alpha: open a
  one-line PR against `MiraNote-AI/.github` adding `poc/*/corpus/*.json`
  to ALLOWLIST_PATTERNS, ship it through review, then proceed.**
  This precedes any corpus work to avoid Rule-3 CI failures.
- Conventional Commits, scope `api`, subject <=72 chars.
- PR titles self-explanatory.
- **No admin-bypass** on any PR merge -- wait for Jason review.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Fully-automated curation could pull low-quality entries into the corpus | Use trusted source datasets (chinese-poetry/chinese-poetry for ZH, Wikiquote/Gutenberg for EN); script enforces de-dupe + length filter; weak entries get fixed by editing the JSON in a follow-up PR rather than blocking PR alpha |
| BGE-M3 1.3 GB download surprises new teammates | Same pattern as voice emotion model; README spells it out |
| sqlite-vec is a young project; could be flaky | API surface is small; chromadb fallback in 5 min if needed |
| Reranker LLM emits non-JSON | 502 with raw output in detail (same as keywords endpoint) |
| Embedding quality on classical Chinese poetry is unproven | 20-query smoke test before merging; swap to `BAAI/bge-large-zh` if needed |
| Cross-origin call from text UI to a new port (8004) needs CORS | retrieval `main.py` adds CORSMiddleware from day one |
| `find_quote` tool name collides with the existing `generate_caption` semantic territory | Tool description explicitly says caption=summary, quote=external source -- agent picks based on user intent |
| Corpus growth past ~10,000 entries breaks the "fit everything in prompt for reranker" assumption | Future: cap candidates at top-K=10 (already does); LLM only sees 10 regardless of corpus size, so corpus scaling is purely a retrieval concern |

## 13. Open follow-ups (post-this-work)

- Migrate chatbot's `search_docs` from substring to this retrieval
  layer (different namespace, same store).
- Add the web app's eventual semantic note search as a third consumer.
- Personalisation: track which quotes the user "saved" and bias
  ranking. Needs user accounts / storage, which we don't have yet.
- Wikiquote runtime verification as a third reranker stage (only if
  we ever loosen the curated-corpus constraint).
- Per-language embedding models if the multilingual one underperforms
  on either language.
- Larger corpus (5,000+) once retrieval scaling matters; pgvector
  migration at the same time.

## 14. Estimated effort

| Work block | Hours |
|---|---|
| POC scaffold (dirs, requirements, .env) | 0.5 |
| Embedder + tests | 1.0 |
| Store + tests | 1.0 |
| Retriever + tests | 1.0 |
| Reranker + tests | 1.0 |
| API + tests | 1.5 |
| Corpus build script + run + commit (~1,000 entries, fully automated) | 2.5 |
| README + smoke + PR α | 1.0 |
| text-clean-expand UI Quote button | 1.0 |
| chatbot find_quote tool + tests | 1.5 |
| README + smoke + PR β | 1.0 |
| **Total** | **~12.5 hours**, realistically 1.5 working days |

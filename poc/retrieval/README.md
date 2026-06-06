# Retrieval POC (RAG)

Local semantic-retrieval service. Two endpoints:

- `POST /quotes` -- pick the best quotes from a curated corpus for a
  given user text. Uses an LLM reranker on top of vector search to
  emit 0-3 picks with one-sentence "why" lines. Zero hallucination:
  the LLM only chooses from candidates we retrieved.
- `POST /search` -- generic top-K semantic search (reserved for
  future doc / note search reuse).

Bilingual (English + Chinese classical poetry).

Spec: `docs/specs/2026-06-05-quote-suggestions-rag-design.md`

## Pieces

- `embedder.py` -- BGE-M3 multilingual embeddings via sentence-transformers
- `store.py` -- sqlite-vec backed vector store (single-file SQLite)
- `retriever.py` -- embedder + store composition with language filter
- `reranker.py` -- LLM call with structured JSON output; picks indexed
- `main.py` -- FastAPI app
- `scripts/build_corpus.py` -- offline corpus builder (LLM tags themes
  only; never generates text or authors)
- `scripts/build_index.py` -- offline index builder (embeds corpus, writes index.db)

## Setup

```bash
cd poc/retrieval
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # fill in LLM_API_KEY
```

The BGE-M3 embedding model is ~1.3 GB and auto-downloads to
`~/.cache/huggingface/` on first call (one-time, several minutes).
Same pattern as the voice POC's emotion model.

## Build the corpus + index (one-time)

The committed `corpus/*.json` are the source of truth; the index `.db`
is gitignored and rebuilt from them by `build_index.py`. Source
datasets + licenses are documented in `corpus/README.md`. To regenerate:

```bash
# 1. Stage source datasets under sources/ (gitignored) -- see corpus/README.md
# 2. Build corpus JSONs (batched LLM theme-tagging, a few minutes)
PYTHONPATH=../.. .venv/bin/python3 scripts/build_corpus.py \
    --sources sources --out corpus --target-en 500 --target-zh 500
# 3. Embed + index (first run also downloads the ~1.3 GB model)
PYTHONPATH=../.. .venv/bin/python3 scripts/build_index.py
```

## Run

```bash
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8004 --reload
```

## Try it (curl)

```bash
# English mood
curl -s -X POST http://localhost:8004/quotes \
  -H 'Content-Type: application/json' \
  -d '{"text":"I feel exhausted but I have to keep going","max":3}' \
  | python3 -m json.tool

# Chinese mood
curl -s -X POST http://localhost:8004/quotes \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天有点丧","max":3,"lang":"zh"}' \
  | python3 -m json.tool

# Generic search
curl -s -X POST http://localhost:8004/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"perseverance","k":5}' \
  | python3 -m json.tool

# Health
curl -s http://localhost:8004/health | python3 -m json.tool
```

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests -v
```

## Configuration

See `.env.example`. The important knobs:

- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` -- reranker provider
- `EMBEDDING_MODEL` -- defaults to `BAAI/bge-m3`
- `CORPUS_DIR` -- defaults to `./corpus`
- `INDEX_DB_PATH` -- defaults to `./data/index.db`
- `MAX_PICKS` -- default 3 (reranker output cap)
- `RETRIEVE_K` -- default 10 (retriever top-K before reranker)

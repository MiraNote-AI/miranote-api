"""Embed the corpus JSONs and write to the sqlite-vec index.

Reads quotes_en.json + quotes_zh.json from the corpus dir, embeds every
entry via the configured embedder, inserts into the store. Runs once;
re-run when the corpus changes (it will replace existing rows by id).

Usage:
    PYTHONPATH=../.. ./poc/retrieval/.venv/bin/python3 scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from poc.retrieval import config, embedder
from poc.retrieval.store import Store


def main():
    corpus_dir = config.CORPUS_DIR
    index_path = config.INDEX_DB_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for f in sorted(corpus_dir.glob("quotes_*.json")):
        loaded = json.loads(f.read_text(encoding="utf-8"))
        print(f"  {f.name}: {len(loaded)} entries")
        entries.extend(loaded)

    if not entries:
        sys.exit(f"No corpus entries found in {corpus_dir}")

    print(f"Total: {len(entries)} entries")
    print(f"Loading embedder ({config.EMBEDDING_MODEL})...")
    print("(first run will download ~1.3 GB to ~/.cache/huggingface/)")

    # Batch embedding for speed.
    texts = [e["text"] for e in entries]
    batch = 32
    print(f"Encoding {len(texts)} texts in batches of {batch}...")
    vectors = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        vecs = embedder.encode(chunk)
        vectors.extend(vecs)
        print(f"  {min(i+batch, len(texts))}/{len(texts)}")

    print(f"Writing to {index_path}...")
    if index_path.exists():
        index_path.unlink()
    store = Store(db_path=index_path, dim=1024)
    for entry, vec in zip(entries, vectors):
        store.insert(entry["id"], entry, vec)
    print(f"Done. {store.count()} entries indexed.")


if __name__ == "__main__":
    main()

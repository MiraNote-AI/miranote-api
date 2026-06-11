"""Compose embedder + store into a search pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from poc.retrieval import embedder
from poc.retrieval.store import Store


class Retriever:
    def __init__(self, store: Store):
        self._store = store

    def search(
        self,
        query: str,
        k: int = 10,
        lang: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to k hits ordered by descending similarity.

        Each hit: {id, text, author?, source?, lang?, themes?, score}.
        score in [0, 1] (1 = identical vector).
        If lang is given ('en' or 'zh'), filter to that language.
        Filtering is done post-hoc on a slightly oversized k to compensate.
        """
        # Oversample so post-filter still has k results in mixed corpora.
        retrieval_k = k * 4 if lang else k
        vec = embedder.encode_one(query)
        raw = self._store.search(vec, k=retrieval_k)

        hits: List[Dict[str, Any]] = []
        for r in raw:
            payload = r["payload"]
            if lang and payload.get("lang") != lang:
                continue
            # The vecs table uses distance_metric=cosine, so
            # distance = 1 - cosine_sim and this mapping recovers the
            # similarity exactly (clamped at 0 for opposing vectors).
            score = max(0.0, 1.0 - r["distance"])
            hits.append({
                "id": r["id"],
                "score": round(score, 4),
                **payload,
            })
            if len(hits) >= k:
                break
        return hits

from __future__ import annotations
import numpy as np


def _unit(seed: int, dim: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def test_retriever_returns_hits_with_score_and_payload(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert(
        "zh_1",
        {"text": "zh text one", "author": "author1", "lang": "zh"},
        _unit(1),
    )
    in_memory_store.insert(
        "en_1",
        {"text": "Rest when weary", "author": "anon", "lang": "en"},
        _unit(2),
    )
    r = Retriever(in_memory_store)
    hits = r.search("anything", k=2)
    assert len(hits) == 2
    assert all("score" in h for h in hits)
    assert all("text" in h and "author" in h and "lang" in h for h in hits)


def test_retriever_filters_by_lang(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert("zh_1", {"text": "z1", "lang": "zh"}, _unit(1))
    in_memory_store.insert("zh_2", {"text": "z2", "lang": "zh"}, _unit(2))
    in_memory_store.insert("en_1", {"text": "e1", "lang": "en"}, _unit(3))
    r = Retriever(in_memory_store)
    hits = r.search("query", k=10, lang="zh")
    assert {h["id"] for h in hits} == {"zh_1", "zh_2"}


def test_retriever_lang_none_returns_all(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert("zh_1", {"text": "z", "lang": "zh"}, _unit(1))
    in_memory_store.insert("en_1", {"text": "e", "lang": "en"}, _unit(2))
    r = Retriever(in_memory_store)
    hits = r.search("query", k=10, lang=None)
    assert {h["id"] for h in hits} == {"zh_1", "en_1"}


def test_retriever_score_in_0_to_1_range(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert("a", {"text": "a"}, _unit(1))
    r = Retriever(in_memory_store)
    hits = r.search("any", k=1)
    assert 0.0 <= hits[0]["score"] <= 1.0

from __future__ import annotations
import numpy as np


def _unit(seed: int, dim: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def test_store_starts_empty(in_memory_store):
    assert in_memory_store.count() == 0


def test_insert_and_count(in_memory_store):
    in_memory_store.insert("a", {"text": "alpha"}, _unit(1))
    in_memory_store.insert("b", {"text": "beta"}, _unit(2))
    assert in_memory_store.count() == 2


def test_search_returns_nearest_first(in_memory_store):
    """Inserting a vector and then querying with the exact same vector
    must return that item as the top hit."""
    target = _unit(42)
    other = _unit(99)
    in_memory_store.insert("target", {"text": "the one"}, target)
    in_memory_store.insert("other", {"text": "not it"}, other)
    hits = in_memory_store.search(target, k=2)
    assert hits[0]["id"] == "target"
    assert hits[0]["payload"]["text"] == "the one"
    assert hits[0]["distance"] < hits[1]["distance"]


def test_search_k_limits_results(in_memory_store):
    for i in range(5):
        in_memory_store.insert(f"item_{i}", {"text": f"#{i}"}, _unit(i))
    hits = in_memory_store.search(_unit(0), k=3)
    assert len(hits) == 3


def test_insert_replaces_on_same_id(in_memory_store):
    in_memory_store.insert("dup", {"text": "first"}, _unit(1))
    in_memory_store.insert("dup", {"text": "second"}, _unit(2))
    assert in_memory_store.count() == 1
    hits = in_memory_store.search(_unit(2), k=1)
    assert hits[0]["payload"]["text"] == "second"

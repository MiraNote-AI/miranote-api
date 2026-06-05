from __future__ import annotations

import numpy as np


def test_health_returns_status_and_config(api_client):
    test_client, _, _ = api_client
    r = test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedder"] == "BAAI/bge-m3"
    assert body["store"] == "sqlite-vec"
    assert "corpus_size" in body
    assert body["namespaces"] == ["quotes"]


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _seed_store(test_client_app, entries):
    """Helper: insert rows directly into the live store the API uses."""
    import retrieval_main_for_tests as main
    for id_, payload in entries:
        main.store.insert(id_, payload, _unit(abs(hash(id_)) % 1000))


def test_search_returns_top_k_with_score(api_client):
    test_client, _, _ = api_client
    _seed_store(test_client, [
        ("zh_1", {"text": "z1", "lang": "zh"}),
        ("en_1", {"text": "e1", "lang": "en"}),
    ])
    r = test_client.post("/search", json={"query": "anything", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert "hits" in body
    assert len(body["hits"]) == 2
    assert all("score" in h for h in body["hits"])
    assert all("metadata" in h for h in body["hits"])


def test_search_rejects_unknown_namespace(api_client):
    test_client, _, _ = api_client
    r = test_client.post("/search", json={"query": "x", "namespace": "bogus"})
    assert r.status_code == 422


def test_search_default_k_is_10(api_client):
    test_client, _, _ = api_client
    for i in range(15):
        _seed_store(test_client, [(f"x_{i}", {"text": f"#{i}"})])
    r = test_client.post("/search", json={"query": "x"})
    assert r.status_code == 200
    assert len(r.json()["hits"]) == 10

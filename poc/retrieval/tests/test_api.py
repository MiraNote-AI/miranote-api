from __future__ import annotations


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

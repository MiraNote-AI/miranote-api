from __future__ import annotations
import numpy as np


def test_encode_one_returns_1024d_vector(stub_embedder_model):
    from poc.retrieval import embedder
    v = embedder.encode_one("hello")
    assert v.shape == (1024,)
    assert v.dtype == np.float32


def test_encode_batch_returns_n_by_dim(stub_embedder_model):
    from poc.retrieval import embedder
    v = embedder.encode(["a", "b", "c"])
    assert v.shape == (3, 1024)


def test_embedder_lazy_loads_once(stub_embedder_model):
    """Second call must not re-instantiate the SentenceTransformer."""
    from poc.retrieval import embedder
    embedder.encode_one("a")
    embedder.encode_one("b")
    embedder.encode_one("c")
    # _MODEL is set after first call and not replaced
    assert embedder._MODEL is stub_embedder_model
    # Stub recorded all three encode invocations
    assert len(stub_embedder_model.encode_calls) == 3


def test_encoded_vectors_are_normalised(stub_embedder_model):
    from poc.retrieval import embedder
    v = embedder.encode_one("anything")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5

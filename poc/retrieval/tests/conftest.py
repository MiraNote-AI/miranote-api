from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest


# --- Embedder stub ---------------------------------------------------

class _StubModel:
    """Drop-in for SentenceTransformer that returns deterministic vectors."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.encode_calls: List[List[str]] = []

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.encode_calls.append(list(texts))
        # Deterministic: hash-based, normalised
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            out[i] = v
        return out


@pytest.fixture
def stub_embedder_model(monkeypatch):
    """Replace SentenceTransformer with a stub that returns deterministic vectors."""
    stub = _StubModel()
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *args, **kwargs: stub,
    )
    # Also reset embedder._MODEL so a fresh load picks up the stub.
    from poc.retrieval import embedder
    monkeypatch.setattr(embedder, "_MODEL", None)
    return stub


# --- Store fixture ---------------------------------------------------

@pytest.fixture
def in_memory_store():
    """sqlite-vec store backed by :memory: -- isolated per test."""
    from poc.retrieval.store import Store
    return Store(db_path=":memory:", dim=1024)


# --- LLM stub --------------------------------------------------------

class _FakeChatCompletions:
    def __init__(self):
        self.scripted: List[str] = []
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("FakeChatCompletions: no more scripted responses")
        content = self.scripted.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeLLM:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())

    def reply_with(self, *responses: str):
        self.chat.completions.scripted.extend(responses)


@pytest.fixture
def fake_llm():
    return _FakeLLM()


# --- TestClient ------------------------------------------------------

@pytest.fixture
def api_client(stub_embedder_model, fake_llm, monkeypatch):
    """FastAPI TestClient with all externals stubbed."""
    os.environ.setdefault("LLM_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake_llm)

    main_path = Path(__file__).parent.parent / "main.py"
    if "retrieval_main_for_tests" in sys.modules:
        del sys.modules["retrieval_main_for_tests"]
    spec = importlib.util.spec_from_file_location(
        "retrieval_main_for_tests", main_path
    )
    main = importlib.util.module_from_spec(spec)
    sys.modules["retrieval_main_for_tests"] = main
    spec.loader.exec_module(main)

    from fastapi.testclient import TestClient
    return TestClient(main.app), fake_llm, stub_embedder_model

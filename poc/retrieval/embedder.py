"""BGE-M3 multilingual embedder.

Wraps sentence-transformers in a lazy-loading module so importing the
package is cheap. First call to encode() downloads the model (~1.3 GB
to ~/.cache/huggingface/) and pins it in memory for subsequent calls.

Embeddings are L2-normalised so cosine similarity == dot product, and
ranking against a vector store can use either metric interchangeably.
"""
from __future__ import annotations

from threading import Lock
from typing import List

import numpy as np

from poc.retrieval import config

_MODEL = None
_LOCK = Lock()


def _get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer
                _MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
    return _MODEL


def encode(texts: List[str]) -> np.ndarray:
    """Embed N texts to a (N, dim) float32 numpy array, L2-normalised."""
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def encode_one(text: str) -> np.ndarray:
    """Embed a single text to a (dim,) float32 vector."""
    return encode([text])[0]

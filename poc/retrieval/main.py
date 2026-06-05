"""MiraNote POC -- semantic retrieval over a local quote corpus.

Backend-only. Two endpoints:
- POST /search  -- generic top-K semantic search over a corpus
- POST /quotes  -- business endpoint: /search + LLM reranker + "why" lines

Uses BGE-M3 (multilingual) embeddings via sentence-transformers and a
sqlite-vec store. The reranker runs through any OpenAI-compatible LLM.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

from poc.retrieval import config, reranker
from poc.retrieval.retriever import Retriever
from poc.retrieval.store import Store


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)
    namespace: str = Field("quotes")


class SearchHit(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    hits: List[SearchHit]


if not config.LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY is required. Set it in .env")

client_kwargs: Dict[str, Any] = {"api_key": config.LLM_API_KEY}
if config.LLM_BASE_URL:
    client_kwargs["base_url"] = config.LLM_BASE_URL
llm = OpenAI(**client_kwargs)


def _open_store() -> Store:
    """Open the index DB if it exists; create empty one otherwise."""
    index_path = config.INDEX_DB_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return Store(db_path=index_path, dim=1024)


store = _open_store()
retriever = Retriever(store)


app = FastAPI(title="MiraNote Retrieval", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embedder": config.EMBEDDING_MODEL,
        "store": "sqlite-vec",
        "corpus_size": store.count(),
        "namespaces": ["quotes"],
    }


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Generic top-K semantic search over the corpus."""
    if req.namespace != "quotes":
        raise HTTPException(status_code=422, detail=f"unknown namespace: {req.namespace}")
    hits = retriever.search(req.query, k=req.k)
    return SearchResponse(
        hits=[
            SearchHit(
                id=h["id"],
                text=h["text"],
                score=h["score"],
                metadata={k: v for k, v in h.items() if k not in ("id", "text", "score")},
            )
            for h in hits
        ]
    )

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
from typing import Any, Dict, List, Literal, Optional

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


class QuotesRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max: int = Field(3, ge=1, le=5)
    lang: Literal["auto", "en", "zh", "both"] = Field("auto")


class QuoteMatch(BaseModel):
    text: str
    author: Optional[str] = None
    source: Optional[str] = None
    lang: Optional[str] = None
    score: float
    why: str


class QuotesResponse(BaseModel):
    matches: List[QuoteMatch]


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


@app.post("/quotes", response_model=QuotesResponse)
async def quotes(req: QuotesRequest):
    """Pick the best quotes from the corpus for the user's text."""
    lang_filter = None if req.lang in ("auto", "both") else req.lang
    candidates = retriever.search(req.text, k=config.RETRIEVE_K, lang=lang_filter)
    if not candidates:
        return QuotesResponse(matches=[])

    # Sort candidates by ID for deterministic ordering in reranker prompt
    candidates = sorted(candidates, key=lambda c: c["id"])

    try:
        picks = reranker.rerank(
            llm, config.LLM_MODEL, req.text, candidates, max_picks=req.max,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    matches: List[QuoteMatch] = []
    for p in picks:
        c = candidates[p["index"] - 1]  # 1-indexed from reranker
        matches.append(QuoteMatch(
            text=c["text"],
            author=c.get("author"),
            source=c.get("source"),
            lang=c.get("lang"),
            score=c["score"],
            why=p["why"],
        ))
    return QuotesResponse(matches=matches)

"""
MiraNote POC -- Photo Collections API (port 8006)

Takes up to 10 small photos of one life fragment, runs two model calls
(A: look, B: write) with program checks between them, and returns the
finished push / story / ending content as an async job:

    POST /collections/compose      -> 202 {"job_id"}
    GET  /collections/jobs/{id}    -> running | ok | rejected | error
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import beta_auth
import config
import llm
import texts

_llm_status = llm.LLM_UNCONFIGURED


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Probe the model before serving, so /health tells the truth."""
    global _llm_status
    _llm_status = await asyncio.to_thread(llm.probe)
    yield


app = FastAPI(title="MiraNote Photo Collections", version="0.1.0", lifespan=lifespan)

# Reachable from the public internet through the Cloudflare tunnel, so every
# request needs a beta token. Installed before CORSMiddleware on purpose: the
# most recently added middleware is the outermost, and CORS must stay outside
# the gate to answer a browser preflight, which carries no Authorization
# header.
beta_auth.install(app)

_CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGIN", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bad_request() -> JSONResponse:
    return JSONResponse(
        {"status": "error", "reason": "bad_request", "message": texts.message("bad_request")},
        status_code=400,
    )


@app.post("/collections/compose")
async def compose_route():
    return _bad_request()


@app.get("/collections/jobs/{job_id}")
async def job_route(job_id: str):
    return JSONResponse({"status": "expired"}, status_code=404)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm": _llm_status,
        "model": config.LLM_MODEL,
        "call_a_mode": config.CALL_A_MODE,
        "call_b_mode": config.CALL_B_MODE,
    }

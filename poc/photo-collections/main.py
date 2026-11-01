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
import json
import os
from contextlib import asynccontextmanager
from pathlib import PurePath
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import beta_auth
import compose
import config
import jobs
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


# Jobs live in this process only; replaced in tests to inject a clock.
STORE = jobs.JobStore()

REQUIRED_META = (
    "group_key", "date", "weekday", "start_time", "end_time", "raw_count",
    "covers_all", "recent_question_ids", "photos",
)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_meta(raw) -> Optional[dict]:
    """The parsed meta when it matches handoff 5.1, else None."""
    if not isinstance(raw, str):
        return None
    try:
        meta = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(meta, dict) or any(k not in meta for k in REQUIRED_META):
        return None
    if not isinstance(meta["group_key"], str) or not meta["group_key"].strip():
        return None
    if not isinstance(meta["date"], str) or not _is_int(meta["raw_count"]):
        return None
    if not isinstance(meta["covers_all"], bool) or not isinstance(meta["recent_question_ids"], list):
        return None
    if meta.get("place") is not None and not isinstance(meta["place"], str):
        return None
    photos = meta["photos"]
    if not isinstance(photos, list) or not 1 <= len(photos) <= config.MAX_PHOTOS:
        return None
    ids = []
    for p in photos:
        if not isinstance(p, dict) or not isinstance(p.get("id"), str) or not p["id"]:
            return None
        if compose.parse_time(p.get("taken_at")) is None:
            return None
        if p.get("labels") is not None and not isinstance(p["labels"], list):
            return None
        if p.get("represents") is not None and not _is_int(p["represents"]):
            return None
        ids.append(p["id"])
    if len(set(ids)) != len(ids):
        return None
    return meta


async def _read_photos(form, ids) -> Optional[Dict[str, bytes]]:
    """Uploaded photos by id, in memory only, when they match meta's ids."""
    files = form.getlist("photos")
    if not 1 <= len(files) <= config.MAX_PHOTOS:
        return None
    images: Dict[str, bytes] = {}
    for f in files:
        filename = getattr(f, "filename", None)
        if not filename or not hasattr(f, "read"):
            return None
        stem = PurePath(filename).stem
        if stem in images:
            return None
        data = await f.read(config.MAX_PHOTO_BYTES + 1)
        await f.close()
        if not data or len(data) > config.MAX_PHOTO_BYTES:
            return None
        images[stem] = data
    if set(images) != set(ids):
        return None
    return images


@app.post("/collections/compose")
async def compose_route(request: Request):
    """Validate and queue one group; 202 {"job_id"} at once."""
    try:
        form = await request.form(max_files=config.MAX_PHOTOS + 1)
    except Exception:  # noqa: BLE001 -- not multipart, or malformed
        return _bad_request()
    try:
        meta = _check_meta(form.get("meta"))
        if meta is None:
            return _bad_request()
        images = await _read_photos(form, [p["id"] for p in meta["photos"]])
        if images is None:
            return _bad_request()
    finally:
        await form.close()
    job_id = STORE.submit(lambda deadline: compose.run(meta, images, deadline))
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/collections/jobs/{job_id}")
async def job_route(job_id: str):
    result = STORE.get(job_id)
    if result is None:
        return JSONResponse({"status": "expired"}, status_code=404)
    return JSONResponse(result)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm": _llm_status,
        "model": config.LLM_MODEL,
        "call_a_mode": config.CALL_A_MODE,
        "call_b_mode": config.CALL_B_MODE,
    }

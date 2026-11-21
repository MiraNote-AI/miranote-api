"""
MiraNote POC -- Photo Collections API (port 8006)

Takes up to 10 small photos of one life fragment, runs two model calls
(A: look, B: write) with program checks between them, and returns the
finished push / story / ending content as an async job:

    POST /collections/compose      -> 202 {"job_id"}
    GET  /collections/jobs/{id}    -> running | ok | rejected | error
    GET  /collections/jobs?ids=a,b -> {"jobs": {id: that body | expired}}
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import ai_auth
import compose
import config
import jobs
import llm
import logclean
import rounds
import texts

_llm_status = llm.LLM_UNCONFIGURED


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Probe the model before serving, so /health tells the truth."""
    global _llm_status
    if config.LLM_WORKERS < 2 * config.WORKERS:
        print("[collections] warning: PHOTO_LLM_WORKERS={} is below 2 x PHOTO_WORKERS={}; "
              "jobs may queue behind each other's model calls".format(
                  config.LLM_WORKERS, config.WORKERS))
    threading.Thread(
        target=logclean.run_daily,
        args=(config.LOG_PATH, lambda: datetime.now(timezone.utc), time.sleep),
        name="logclean",
        daemon=True,
    ).start()
    _llm_status = await asyncio.to_thread(llm.probe)
    yield


app = FastAPI(title="MiraNote Photo Collections", version="0.1.0", lifespan=lifespan)

# Reachable from the public internet through the Cloudflare tunnel, so every
# request needs our access token or (while ACCEPT_BETA_TOKENS is on) the beta
# token. Installed before CORSMiddleware on purpose: the most recently added
# middleware is the outermost, and CORS must stay outside the gate to answer a
# browser preflight, which carries no Authorization header.
# Compose answers 202 and calls the model in a job, which settles its own
# ticket (compose_route).
ai_auth.install(app, metered={"/collections/compose": ai_auth.Metered("photo_push", model=config.LLM_MODEL)})

_CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOW_ORIGIN", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _busy() -> JSONResponse:
    # The phone waits 10 s on a 429 and submits again (ComposeSession.submit).
    return JSONResponse({"detail": "busy"}, status_code=429, headers={"Retry-After": "10"})


def _bad_request() -> JSONResponse:
    return JSONResponse(
        {"status": "error", "reason": "bad_request", "message": texts.message("bad_request")},
        status_code=400,
    )


# Jobs live in this process only; replaced in tests to inject a clock.
STORE = jobs.JobStore()

REQUIRED_META = (
    "group_key", "date", "weekday", "start_time", "end_time", "raw_count",
    "covers_all", "recent_question_types", "photos", "bucket",
)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _check_photo(p) -> bool:
    """One meta.photos entry: types as handoff 5.1 sends them."""
    if not isinstance(p, dict) or not isinstance(p.get("id"), str) or not p["id"]:
        return False
    if compose.parse_time(p.get("taken_at")) is None:
        return False
    if p.get("labels") is not None and not _is_str_list(p["labels"]):
        return False
    if p.get("faces") is not None and not _is_int(p["faces"]):
        return False
    for flag in ("favorite", "edited"):
        if p.get(flag) is not None and not isinstance(p[flag], bool):
            return False
    if p.get("represents") is not None and not (_is_int(p["represents"]) and p["represents"] >= 1):
        return False
    return True


def _check_repeat_meta(meta: dict) -> Optional[dict]:
    """A repeat-theme meta: a theme, 5..9 photos (days are checked by call A)."""
    if not isinstance(meta.get("group_key"), str) or not meta["group_key"].strip():
        return None
    if not _is_str_list(meta.get("recent_question_types")):
        return None
    tz = meta.get("tz")
    if tz is not None and not (isinstance(tz, str) and len(tz) <= config.MAX_TZ_CHARS):
        return None
    theme = meta.get("theme")
    if not isinstance(theme, dict):
        return None
    for key in ("id", "name", "noun", "same_part"):
        value = theme.get(key)
        if not isinstance(value, str) or len(value) > rounds.MAX_TEXT_CHARS:
            return None
        # Round 1 may give no noun or same part; spec 6.9 has a no-noun CTA.
        if key in ("id", "name") and not value.strip():
            return None
    if theme.get("where") is not None and not rounds.check_where(theme["where"]):
        return None
    photos = meta.get("photos")
    if not isinstance(photos, list) or not config.REPEAT_MIN_PHOTOS <= len(photos) <= config.REPEAT_MAX_PHOTOS:
        return None
    if not all(_check_photo(p) for p in photos):
        return None
    ids = [p["id"] for p in photos]
    if len(set(ids)) != len(ids):
        return None
    return meta


def _check_meta(raw) -> Optional[dict]:
    """The parsed meta when it matches handoff 5.1, else None."""
    if not isinstance(raw, str):
        return None
    try:
        meta = json.loads(raw)
    except ValueError:
        return None
    if isinstance(meta, dict) and "mode" in meta:  # null is not a mode
        if meta["mode"] == "repeat":
            return _check_repeat_meta(meta)
        if meta["mode"] != "life_fragment":
            return None
    if not isinstance(meta, dict) or any(k not in meta for k in REQUIRED_META):
        return None
    if not isinstance(meta["group_key"], str) or not meta["group_key"].strip():
        return None
    if not isinstance(meta["date"], str):
        return None
    if not (_is_int(meta["weekday"]) and 1 <= meta["weekday"] <= 7):
        return None
    if not isinstance(meta["start_time"], str) or not isinstance(meta["end_time"], str):
        return None
    if not (_is_int(meta["raw_count"]) and meta["raw_count"] >= 1):
        return None
    if not isinstance(meta["covers_all"], bool) or not _is_str_list(meta["recent_question_types"]):
        return None
    if not isinstance(meta["bucket"], str) or meta["bucket"] not in config.BUCKET_IDS:
        return None
    tz = meta.get("tz")
    if tz is not None and not (isinstance(tz, str) and len(tz) <= config.MAX_TZ_CHARS):
        return None
    if meta.get("place") is not None and not isinstance(meta["place"], str):
        return None
    photos = meta["photos"]
    if not isinstance(photos, list) or not 1 <= len(photos) <= config.MAX_PHOTOS:
        return None
    if not all(_check_photo(p) for p in photos):
        return None
    ids = [p["id"] for p in photos]
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
        # JPEG only, sized from its header: a small file can still decode to
        # a huge image (review of api#97). The phone sends JPEG.
        size = compose.jpeg_size(data)
        if size is None or max(size) > config.MAX_PHOTO_PX:
            return None
        images[stem] = data
    if set(images) != set(ids):
        return None
    return images


@app.post("/collections/compose")
async def compose_route(request: Request):
    """Validate and queue one group; 202 {"job_id"} at once."""
    if STORE.full():  # before reading the photos; submit checks again
        return _busy()
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
    # The model runs after this answer, so the job settles the reserved unit:
    # charged only when it produced a push. Any answer but 202 is settled
    # failed by the gate.
    ticket = ai_auth.take_ticket(request)

    def run(deadline: float) -> dict:
        ok = False
        try:
            result = compose.run(meta, images, deadline)
            ok = result.get("status") == "ok"
            return result
        finally:
            if ticket is not None:
                ticket.settle(ok)

    try:
        job_id = STORE.submit(run)
    except jobs.Busy:
        return _busy()
    return JSONResponse({"job_id": job_id}, status_code=202)



async def _read_round_photos(form, numbers, limit: int) -> Optional[Dict[int, bytes]]:
    """Round photos by number ("3.jpg" is photo 3), in memory only, when they
    are exactly the numbers meta lists. Small by contract: 336 px, 200 KB."""
    files = form.getlist("photos")
    if not 1 <= len(files) <= limit:
        return None
    images: Dict[int, bytes] = {}
    for f in files:
        filename = getattr(f, "filename", None)
        if not filename or not hasattr(f, "read"):
            return None
        stem = PurePath(filename).stem
        if not stem.isdigit() or int(stem) in images:
            return None
        data = await f.read(config.ROUND_MAX_BYTES + 1)
        await f.close()
        if not data or len(data) > config.ROUND_MAX_BYTES:
            return None
        size = compose.jpeg_size(data)
        if size is None or max(size) > config.ROUND_MAX_PX:
            return None
        images[int(stem)] = data
    if set(images) != set(numbers):
        return None
    return images


async def _round_route(request: Request, kind: str, check, limit: int, build, log_extra):
    """Validate a multipart round request, call the model off the event loop."""
    try:
        form = await request.form(max_files=limit + 1)
    except Exception:  # noqa: BLE001 -- not multipart, or malformed
        return _bad_request()
    try:
        meta = check(form.get("meta"))
        if meta is None:
            return _bad_request()
        images = await _read_round_photos(form, rounds.photo_numbers(meta), limit)
        if images is None:
            return _bad_request()
    finally:
        await form.close()
    messages = build(meta, images)
    status, body = await run_in_threadpool(rounds.call, kind, messages, log_extra(meta))
    if kind == "round1" and status == 200:
        body["where"] = rounds.group_where(meta, texts.json("names"))
    return JSONResponse(body, status_code=status)


@app.post("/collections/rounds/1")
async def round1_route(request: Request):
    """Name a pile and pick a split; the model's reply goes back verbatim."""
    return await _round_route(
        request, "round1", rounds.check_round1, config.ROUND1_MAX_PHOTOS,
        rounds.messages_round1,
        lambda m: {"photos": len(m["photos"]), "pile": {k: m["pile"][k] for k in ("photo_count", "day_count")}, "reask": m.get("old") is not None})


@app.post("/collections/rounds/2")
async def round2_route(request: Request):
    """Check push candidates against a theme and the compare lists."""
    return await _round_route(
        request, "round2", rounds.check_round2, config.ROUND2_MAX_PHOTOS,
        rounds.messages_round2, lambda m: {"photos": len(m["photos"])})


@app.post("/collections/rounds/shorten")
async def shorten_route(request: Request):
    """Ask for shorter names; text only."""
    try:
        body = rounds.check_shorten(await request.json())
    except Exception:  # noqa: BLE001 -- not JSON
        body = None
    if body is None:
        return _bad_request()
    status, out = await run_in_threadpool(
        rounds.call, "shorten", rounds.messages_shorten(body["items"]), {"items": len(body["items"])})
    return JSONResponse(out, status_code=status)


MAX_BATCH_IDS = 20  # the phone sends at most 7; the bound stops abuse of one request


@app.get("/collections/jobs")
async def batch_route(ids: Optional[str] = None):
    """Poll many jobs at once: {"jobs": {id: body}}, an unknown id reads expired."""
    wanted = list(dict.fromkeys(i.strip() for i in (ids or "").split(",") if i.strip()))
    if not wanted or len(wanted) > MAX_BATCH_IDS:
        return _bad_request()
    return JSONResponse({"jobs": {i: STORE.get(i) or {"status": "expired"} for i in wanted}})


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
        "call_b_model": config.CALL_B_MODEL,
        "call_a_mode": config.CALL_A_MODE,
        "call_b_mode": config.CALL_B_MODE,
    }

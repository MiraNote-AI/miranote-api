"""The compose pipeline (AI doc 0, copy refine handoff 2):

    call A -> check A -> templates -> call B (sees the photos) -> program
      checks -> call C (checks against the photos) -> pick detail sentence,
      up to 3 questions -> memory (handoff 5.2, copy refine handoff 5.7)

run() returns exactly the job result body. It never raises for a model or
check failure; every outcome is one of ok / rejected / error. One JSON line
per job goes to config.LOG_PATH (AI doc 6) -- text only, no image bytes.
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

import body
import check_a
import config
import llm
import questions
import texts
import verify

ATTEMPTS = 2  # each call is tried once more after a failure
# Kept back from every call so the program can still put the answer
# together before the job cap (full check 2026-11-05: a late cover
# rewrite used to turn a ready answer into a timeout).
CALL_MARGIN_SECONDS = 2.0

LOG_LOCK = threading.Lock()  # shared with logclean's purge


# ---------- small helpers ----------

def parse_time(value) -> Optional[datetime]:
    """An ISO-8601 taken_at as the phone sent it (local wall time + offset)."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    if v.endswith("Z") or v.endswith("z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def _sort_key(dt: Optional[datetime]) -> float:
    if dt is None:
        return float("inf")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_HEIC_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1")


def image_mime(data: bytes) -> Optional[str]:
    """The media type of an upload by its magic bytes; None if not an image
    we accept (JPEG, PNG, HEIC)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[4:8] == b"ftyp" and data[8:12] in _HEIC_BRANDS:
        return "image/heic"
    return None


def daypart_of(hour: int) -> str:
    """5-11 morning, 11-14 noon, 14-17 afternoon, 17-19 evening, else night."""
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "noon"
    if 14 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 19:
        return "evening"
    return "night"


def date_label(date: str) -> str:
    """'2026-10-18' -> '10.18', '2026-09-05' -> '9.05'."""
    parts = str(date).split("-")
    if len(parts) >= 3:
        try:
            return "{}.{:02d}".format(int(parts[1]), int(parts[2][:2]))
        except ValueError:
            pass
    return str(date)


def bucket_name(meta: dict) -> str:
    """The Chinese push bucket name for the id the phone sent in meta.bucket."""
    return texts.json("names")["buckets"][meta["bucket"]]


def _place(meta: dict) -> Optional[str]:
    place = meta.get("place")
    if isinstance(place, str) and place.strip():
        return place.strip()
    return None


def _result(status: str, reason: str) -> dict:
    return {"status": status, "reason": reason, "message": texts.message(reason)}


# ---------- prompts ----------

def _image_part(data: bytes) -> dict:
    return {"type": "image_url", "image_url": {
        "url": "data:{};base64,{}".format(
            image_mime(data) or "image/jpeg", base64.b64encode(data).decode("ascii"))}}


def shrink(data: bytes, px: int) -> bytes:
    """`data` as a JPEG at most `px` on the long side; unchanged when it is
    already that small or cannot be decoded here (calls B and C then see
    the photo as sent)."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        if max(img.size) <= px:
            return data
        img = img.convert("RGB")
        img.thumbnail((px, px))
        out = io.BytesIO()
        img.save(out, "JPEG", quality=80)
        return out.getvalue()
    except Exception:  # noqa: BLE001 -- any decode failure keeps the original
        return data


def _messages_shorten(cands: List[body.Candidate], scene: str) -> List[dict]:
    """The shorten round: text only, the sentences cut for length."""
    names = texts.json("names")
    lines = "\n".join(names["shorten_line"].format(
        i=i + 1, text=body.clean(c.text), n=body.char_len(body.clean(c.text)),
        style=c.style or names["none"],
        ids=names["list_sep"].join(c.photo_ids) or names["none"]) for i, c in enumerate(cands))
    return [
        {"role": "system", "content": texts.raw("collect_b_system").strip()},
        {"role": "user", "content": texts.render("shorten_b_user", lines=lines,
                                                 scene=scene or names["none"])},
    ]


def _messages_shorten_q(qs: List[questions.Question], scene: str) -> List[dict]:
    """The question shorten round (decision 2026-11-05 evening, question
    item 5): text only, the questions cut for length."""
    names = texts.json("names")
    lines = "\n".join(names["shorten_q_line"].format(
        i=i + 1, text=q.text, n=body.char_len(q.text), type=q.type)
        for i, q in enumerate(qs))
    return [
        {"role": "system", "content": texts.raw("collect_b_system").strip()},
        {"role": "user", "content": texts.render("shorten_q_user", lines=lines,
                                                 scene=scene or names["none"])},
    ]


def _split(name: str) -> Tuple[str, str, str]:
    """A prompt with photo lines: (header, per-photo template, task)."""
    header, _, rest = texts.raw(name).partition("=====PHOTO=====\n")
    photo_tpl, _, task = rest.partition("=====TASK=====\n")
    return header, photo_tpl.strip("\n"), task


def _hhmm(photo: dict) -> str:
    dt = parse_time(photo.get("taken_at"))
    return dt.strftime("%H:%M") if dt else ""


def _count_text(counts: Dict[str, int], names: dict) -> str:
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return names["sep"].join(names["b_count"].format(name=k, n=v) for k, v in ordered)


def _messages_a(meta: dict, photos: List[dict], images: Dict[str, bytes]) -> List[dict]:
    names = texts.json("names")
    header, photo_tpl, task = _split("collect_a_user")
    weekday = names["weekdays"].get(str(meta.get("weekday")), "")
    content: List[dict] = [{"type": "text", "text": header.format_map({
        "date": meta.get("date"),
        "weekday": weekday,
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
        "place": _place(meta) or names["none"],
        "raw_count": meta.get("raw_count"),
        "sent_count": len(photos),
    }).strip("\n")}]
    for p in photos:
        represents = int(p.get("represents") or 1)
        content.append({"type": "text", "text": photo_tpl.format_map({
            "id": p["id"],
            "time": _hhmm(p),
            "labels": names["list_sep"].join(p.get("labels") or []) or names["none"],
            "faces": int(p.get("faces") or 0),
            "favorite": names["a_favorite"] if p.get("favorite") is True else "",
            "represents": represents,
            "merged": names["a_merged"] if represents > 1 else "",
        })})
        content.append(_image_part(images[p["id"]]))
    content.append({"type": "text", "text": task.format_map(
        {"schema_a": texts.raw("schema_a_example").strip()})})
    return [
        {"role": "system", "content": texts.raw("collect_a_system").strip()},
        {"role": "user", "content": content},
    ]


def _header_values(meta: dict, daypart: str, counts: Dict[str, int], names: dict) -> dict:
    return {
        "date": meta.get("date"),
        "weekday": names["weekdays"].get(str(meta.get("weekday")), ""),
        "daypart": names["dayparts"][daypart],
        "place": _place(meta) or names["none"],
        "counts": _count_text(counts, names) or names["none"],
    }


def _messages_b(meta: dict, a: check_a.AResult, daypart: str, counts: Dict[str, int],
                by_id: Dict[str, dict], images: Dict[str, bytes]) -> List[dict]:
    """Call B sees the selected photos, in time order, with call A's
    descriptions; the cover is marked (handoff 2.2)."""
    names = texts.json("names")
    header, photo_tpl, task = _split("collect_b_user")
    values = _header_values(meta, daypart, counts, names)
    values["k"] = len(a.selected)
    content: List[dict] = [{"type": "text", "text": header.format_map(values).strip("\n")}]
    for pid in a.selected:
        p = a.photos[pid]
        content.append({"type": "text", "text": photo_tpl.format_map({
            "id": pid, "time": _hhmm(by_id[pid]),
            "cover": names["b_cover"] if pid == a.peak else "",
            "description": p.description or names["none"], "people": p.people_count,
        })})
        content.append(_image_part(images[pid]))
    content.append({"type": "text", "text": task.format_map({
        "type_table": questions.prompt_types(),
        "recent_types": questions.prompt_recent(list(meta.get("recent_question_types") or [])),
    })})
    return [
        {"role": "system", "content": texts.raw("collect_b_system").strip()},
        {"role": "user", "content": content},
    ]


def _messages_b_peak(meta: dict, a: check_a.AResult, daypart: str, counts: Dict[str, int],
                     by_id: Dict[str, dict], images: Dict[str, bytes],
                     too_long: List[str], scene: str) -> List[dict]:
    """The extra call B: detail sentences about the cover photo only. The
    first round's cover sentences cut only for length are shown back to be
    shortened (the model overshoots 22 characters; first real runs)."""
    names = texts.json("names")
    long_lines = "\n".join(names["too_long_line"].format(text=t, n=body.char_len(t))
                           for t in too_long) or names["none"]
    header, photo_tpl, task = _split("collect_b_peak_user")
    p = a.photos[a.peak]
    values = _header_values(meta, daypart, counts, names)
    values["scene"] = scene or names["none"]
    content: List[dict] = [
        {"type": "text", "text": header.format_map(values).strip("\n")},
        {"type": "text", "text": photo_tpl.format_map({
            "id": a.peak, "time": _hhmm(by_id[a.peak]),
            "description": p.description or names["none"], "people": p.people_count})},
        _image_part(images[a.peak]),
        {"type": "text", "text": task.format_map({"peak_id": a.peak, "too_long": long_lines})},
    ]
    return [
        {"role": "system", "content": texts.raw("collect_b_system").strip()},
        {"role": "user", "content": content},
    ]


def _messages_c(ids: List[str], cover: str, images: Dict[str, bytes],
                sentences: List[str], qs: List[str]) -> List[dict]:
    names = texts.json("names")
    photo_tpl, task = verify.user_text(sentences, qs)
    content: List[dict] = []
    for pid in ids:
        content.append({"type": "text", "text": photo_tpl.format_map({
            "id": pid, "cover": names["b_cover"] if pid == cover else ""})})
        content.append(_image_part(images[pid]))
    content.append({"type": "text", "text": task})
    return [
        {"role": "system", "content": texts.raw("verify_c_system").strip()},
        {"role": "user", "content": content},
    ]


def _check_rewrite(text: str) -> dict:
    b = check_a.parse_json(text)
    if not isinstance(b.get("candidates", []), list):
        raise check_a.CheckFailed("candidates is not a list")
    return b


def _check_b(text: str) -> dict:
    b = check_a.parse_json(text)
    if not isinstance(b.get("candidates", []), list):
        raise check_a.CheckFailed("candidates is not a list")
    q = b.get("question")
    if not isinstance(q, dict) or not isinstance(q.get("ranked"), list):
        raise check_a.CheckFailed("question.ranked missing or not a list")
    return b


# ---------- the pipeline ----------

class _Job:
    def __init__(self, meta: dict, deadline: float):
        self.meta = meta
        self.deadline = deadline
        self.started = time.monotonic()
        self.calls: List[dict] = []
        self.log: Dict[str, Any] = {"group_key": meta.get("group_key"),
                                    "bucket": meta.get("bucket"), "tz": meta.get("tz")}

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def cost(self) -> float:
        return sum(c["cost_cny"] for c in self.calls)

    def call(self, which: str, messages: List[dict], per_call: float,
             check, attempts: int = ATTEMPTS) -> Tuple[Optional[Any], str]:
        """Try a call up to `attempts` times. Returns (checked value, "") or
        (None, last failure: timeout | model_error | check_failed)."""
        last = "model_error"
        for attempt in range(1, attempts + 1):
            left = self.remaining() - CALL_MARGIN_SECONDS
            if left <= 0:
                return None, "deadline"
            entry = {"call": which, "attempt": attempt, "model": llm.model_for(which),
                     "latency_ms": 0, "prompt_tokens": 0,
                     "completion_tokens": 0, "cost_cny": 0.0, "raw": "", "error": ""}
            self.calls.append(entry)
            t0 = time.monotonic()
            try:
                reply = llm.complete(messages, call=which, timeout=min(per_call, left))
            except llm.ModelTimeout as error:
                entry.update(latency_ms=int((time.monotonic() - t0) * 1000),
                             error="timeout: " + str(error)[:200])
                last = "timeout"
                continue
            except llm.ModelError as error:
                entry.update(latency_ms=int((time.monotonic() - t0) * 1000),
                             error="model_error: " + str(error)[:200])
                last = "model_error"
                continue
            entry.update(latency_ms=reply.latency_ms, prompt_tokens=reply.prompt_tokens,
                         completion_tokens=reply.completion_tokens,
                         cost_cny=round(reply.cost_cny, 6), raw=reply.text)
            try:
                return check(reply.text), ""
            except check_a.CheckFailed as error:
                entry["error"] = "check_failed: " + error.reason[:200]
                last = "check_failed"
        if self.remaining() <= CALL_MARGIN_SECONDS:
            return None, "deadline"
        return None, last

    def finish(self, result: dict) -> dict:
        # A result completed after the job deadline is not delivered: the
        # phone has been told (or will be told) "timeout" by then.
        if time.monotonic() > self.deadline and not (
                result.get("status") == "error" and result.get("reason") == "timeout"):
            self.log["lateResult"] = True
            self.log["lateStatus"] = result.get("status")
            result = _result("error", "timeout")
        self.log.update(
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            status=result["status"],
            reason=result.get("reason", ""),
            calls=self.calls,
            costCny=round(self.cost(), 6),
            latencyMs=int((time.monotonic() - self.started) * 1000),
        )
        _write_log(self.log)
        return result


def _write_log(entry: dict) -> None:
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        if config.LLM_API_KEY:
            line = line.replace(config.LLM_API_KEY, "<LLM_API_KEY>")
        path = config.LOG_PATH
        with LOG_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as error:  # noqa: BLE001 -- a log failure never fails a job
        print("[collections] could not write the compose log: {}".format(error))


# Full-width comma, full stop, semicolon: where a description is cut into clauses.
_CLAUSE_END = re.compile("([\uff0c\u3002\uff1b])")


def _description_for_facts(description: str, a: check_a.AResult) -> str:
    """Call A's description without the clauses that name other photos
    ("same framing as P1"): facts become the photo's summary in the saved
    page (full check 2026-11-05)."""
    parts = _CLAUSE_END.split(description)
    kept = []
    for i in range(0, len(parts), 2):
        clause, end = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
        if clause.strip() and body.marker_leak(clause, a) is None:
            kept.append(clause + end)
    return "".join(kept)


def _facts_line(p: check_a.PhotoFacts, a: check_a.AResult, names: dict) -> str:
    none = names["none"]
    return names["facts_line"].format(
        subjects=names["sep"].join(s.name for s in p.subjects) or none,
        description=_description_for_facts(p.description, a) or none,
        people=p.people_count,
    )


def _when(meta: dict, by_id: Dict[str, dict], peak: str, names: dict) -> Tuple[str, body.When]:
    """The daypart id by the cover's time, and the weekday of the group's
    day (meta.weekday): a cover taken after midnight still belongs to the
    night before, like the date in the title (full check 2026-11-05)."""
    peak_time = parse_time(by_id[peak].get("taken_at"))
    daypart = daypart_of(peak_time.hour if peak_time else 12)
    weekday = names["weekdays"].get(str(meta.get("weekday")), "")
    return daypart, body.When(weekday=weekday, daypart=names["dayparts"][daypart])


def _hard_bodies(ai: List[body.Candidate], a: check_a.AResult):
    """Model sentences the program passes, and the rejected ones."""
    passed, rejected = [], []
    for c in ai:
        reason = body.reject_reason(c, a)
        if reason:
            rejected.append((c.text, reason))
        else:
            passed.append(c)
    return passed, rejected


def _verify(job: "_Job", ids: List[str], cover: str, images: Dict[str, bytes],
            ai: List[body.Candidate], qs: List[questions.Question]):
    """Call C on what the program passed. Returns (status, sentences, their
    rejections, questions, their rejections, log). status: skipped (nothing
    to check) | ok | failed (two failures: nothing model-written survives)
    | deadline. A sentence cites the cover exactly when C says it mentions
    it (handoff 2.3 item 3)."""
    log: Dict[str, Any] = {}
    if not ai and not qs:
        return "skipped", ai, [], qs, [], log
    texts_s = [body.clean(c.text) for c in ai]
    texts_q = [q.text for q in qs]
    result, fail = job.call(
        "c", _messages_c(ids, cover, images, texts_s, texts_q), config.CALL_C_TIMEOUT,
        lambda text: verify.parse(text, len(ai), len(qs)))
    if result is None:
        status = "deadline" if fail == "deadline" else "failed"
        log["failure"] = fail
        return status, [], [], [], [], log
    log["sentences"] = {}
    log["questions"] = {}
    kept_s, rej_s = [], []
    for i, c in enumerate(ai):
        v = result.sentences[verify.sentence_id(i)]
        log["sentences"][verify.sentence_id(i)] = {
            "text": texts_s[i], "grounded": v.grounded, "mentions_cover": v.mentions_cover,
            "fluent": v.fluent, "reason": v.reason}
        if not v.grounded:
            rej_s.append((c.text, "c_ungrounded:" + v.reason[:80]))
            continue
        if not v.fluent:
            rej_s.append((c.text, "c_not_fluent:" + v.reason[:80]))
            continue
        ids_ = [pid for pid in c.photo_ids if pid != cover]
        if v.mentions_cover:
            ids_ = [cover] + ids_
        kept_s.append(body.Candidate(text=c.text, source=c.source, photo_ids=ids_, rank=c.rank,
                                     style=c.style))
    kept_q, rej_q = [], []
    for i, q in enumerate(qs):
        v = result.questions[verify.question_id(i)]
        log["questions"][verify.question_id(i)] = {
            "text": q.text, "grounded": v.grounded, "safe": v.safe, "reason": v.reason}
        if not v.grounded:
            rej_q.append((q.text, "c_ungrounded:" + v.reason[:80]))
        elif not v.safe:
            rej_q.append((q.text, "c_unsafe:" + v.reason[:80]))
        else:
            kept_q.append(q)
    return "ok", kept_s, rej_s, kept_q, rej_q, log


def run(meta: dict, images: Dict[str, bytes], deadline: float) -> dict:
    """Compose one memory. meta and images are already validated. Never
    raises: an unexpected error still writes the log line (raw output and
    cost of the calls made so far) and answers error / model_error."""
    job = _Job(meta, deadline)
    try:
        return _compose(job, meta, images)
    except Exception as error:  # noqa: BLE001 -- a bug must not lose the log
        traceback.print_exc()
        job.log["exception"] = "{}: {}".format(type(error).__name__, str(error)[:300])
        return job.finish(_result("error", "model_error"))


def _compose(job: "_Job", meta: dict, images: Dict[str, bytes]) -> dict:
    names = texts.json("names")

    photos = sorted(meta["photos"], key=lambda p: _sort_key(parse_time(p.get("taken_at"))))
    input_ids = [p["id"] for p in photos]
    by_id = {p["id"]: p for p in photos}

    # ---- call A ----
    a, fail = job.call(
        "a", _messages_a(meta, photos, images), config.CALL_A_TIMEOUT,
        lambda text: check_a.validate(check_a.parse_json(text), input_ids),
    )
    if a is None:
        reason = {"deadline": "timeout", "timeout": "timeout", "model_error": "model_error",
                  "check_failed": "check_failed"}[fail]
        status = "rejected" if reason == "check_failed" else "error"
        return job.finish(_result(status, reason))
    job.log["a"] = {"verdict": a.verdict, "selected": a.selected, "peak": a.peak,
                    "peak_replaced_from": a.peak_replaced_from, "skip_reason": a.skip_reason}
    if a.verdict != "ok":
        return job.finish(_result("rejected", a.verdict))

    # ---- program prep ----
    daypart, when = _when(meta, by_id, a.peak, names)
    counts = body.program_counts(a)
    tpls = body.templates(a, meta, when)
    raw_count = int(meta.get("raw_count") or 0)

    # Calls B and C see smaller photos (user decision 2026-11-05).
    images = dict(images)
    for pid in a.selected:
        images[pid] = shrink(images[pid], config.SMALL_IMAGE_PX)

    # ---- call B (failing twice, or running into the job cap, is not a
    # rejection: templates + fallback questions) ----
    b, fail = job.call("b", _messages_b(meta, a, daypart, counts, by_id, images),
                       config.CALL_B_TIMEOUT, _check_b)
    if b is None:
        job.log["b_failure"] = fail
    # The scene line is the model's own note (decision 2026-11-05 evening,
    # item 4): logged and passed to the later rounds, never sent to the phone.
    scene = b.get("scene") if isinstance(b, dict) and isinstance(b.get("scene"), str) else ""
    job.log["scene"] = scene
    first_ai = body.from_b(b)
    job.log["b_candidates"] = [{"text": c.text, "style": c.style, "photo_ids": c.photo_ids}
                               for c in first_ai]
    ai, body_rejected = _hard_bodies(first_ai, a)
    too_long = [body.clean(c.text) for c in first_ai
                if a.peak in c.photo_ids and body.reject_reason(c, a) == "too_long"]
    model_qs = questions.from_b(b)

    # Question shorten round (decision 2026-11-05 evening, question item 5):
    # questions cut only for length -> one text-only call (no retry); the
    # shortened ones take their place and go through the program checks and
    # call C with the rest. Still too long -> dropped as before. It runs
    # before call C, so it only spends time C would not need (review
    # 2026-11-05: a slow reply here used to leave C no time at all).
    long_qs = questions.too_long(model_qs, a)
    spare = job.remaining() - config.CALL_C_TIMEOUT - CALL_MARGIN_SECONDS
    if long_qs and spare > CALL_MARGIN_SECONDS:
        sq, _ = job.call("b", _messages_shorten_q(long_qs, scene),
                         min(config.CALL_B_TIMEOUT, spare), _check_b, attempts=1)
        replies = questions.from_b(sq)
        job.log["question_shorten"] = {"asked": [q.text for q in long_qs],
                                       "replies": [q.text for q in replies]}
        model_qs = questions.with_shortened(model_qs, long_qs, replies)
    qs, q_rejected = questions.hard_check(model_qs, a)

    # ---- call C: only what it passes is used; two failures -> templates,
    # F0 and fallback questions only (handoff 2.4) ----
    c_status, ai, c_rej_s, qs, c_rej_q, c_log = _verify(job, a.selected, a.peak, images, ai, qs)
    if c_status == "deadline":
        # Out of time before anything was checked: same as call C failing
        # twice (templates, F0 and fallback questions only).
        c_status = "failed"
    job.log["c"] = dict(c_log, status=c_status)
    body_rejected += c_rej_s
    q_rejected += c_rej_q

    # ---- detail sentence ----
    detail = body.pick(ai, tpls, a, raw_count=raw_count, when=when)
    first_rejected = body_rejected + list(detail.rejected)
    detail.rejected = list(first_rejected)

    # Shorten round (user decision 2026-11-05): no checked model sentence
    # reaches the cover and some were cut only for length -> one text-only
    # call B (no retry) shortens them, nothing added; the results keep their
    # sentence's rank and photos and go through the program checks and call
    # C like the first round. Skipped when call C failed.
    long_ones = [c for c in first_ai if body.reject_reason(c, a) == "too_long"]
    shortened = False
    if (long_ones and not body.cites_peak(detail, a) and c_status != "failed"
            and job.remaining() > CALL_MARGIN_SECONDS):
        shortened = True
        sb, _ = job.call("b", _messages_shorten(long_ones, scene), config.CALL_B_TIMEOUT,
                         _check_rewrite, attempts=1)
        s_new = body.from_b(sb)[:len(long_ones)]
        # A reply keeps its own photos when they are selected ones (it may
        # drop a sentence, full check 2026-11-05). The long sentences were
        # asked in rank order, so a reply missing some keeps its own order.
        same_count = len(s_new) == len(long_ones)
        for i, c in enumerate(s_new):
            own = [pid for pid in c.photo_ids if pid in a.selected]
            c.photo_ids = own or (list(long_ones[i].photo_ids) if same_count else [])
            if same_count:
                c.rank = long_ones[i].rank
                c.style = c.style or long_ones[i].style
        s_new = [c for c in s_new if c.photo_ids]
        s_ai, s_rejected = _hard_bodies(s_new, a)
        sc_status, s_ai, sc_rej, _, _, sc_log = _verify(job, a.selected, a.peak, images, s_ai, [])
        if sc_status == "deadline":
            s_ai = []  # optional round out of time: keep the answer we have
        job.log["shorten"] = {"asked": [body.clean(c.text) for c in long_ones],
                              "rejected": s_rejected + sc_rej, "c": dict(sc_log, status=sc_status)}
        ai = ai + s_ai
        first_rejected = first_rejected + s_rejected + sc_rej
        detail = body.pick(ai, tpls, a, raw_count=raw_count, when=when)
        detail.rejected = list(first_rejected)
    job.log["shortened"] = shortened

    # The body must describe the cover (user decision 2026-10-31): no model
    # sentence about the peak survived -> one extra call B for the peak only
    # (no retry, inside the job deadline), checked by the program and call C
    # on the cover alone; still none -> a model sentence about another photo
    # makes that photo the cover; a template or F0 keeps the cover. Skipped
    # when call C failed: nothing unchecked may be shown.
    peak_rewrite = False
    peak_switched = False
    old_peak = a.peak
    if (not body.cites_peak(detail, a) and b is not None and c_status != "failed"
            and job.remaining() > CALL_MARGIN_SECONDS):
        peak_rewrite = True
        rb, _ = job.call("b", _messages_b_peak(meta, a, daypart, counts, by_id, images, too_long,
                                               scene),
                         config.CALL_B_TIMEOUT, _check_rewrite, attempts=1)
        r_ai, r_rejected = _hard_bodies(body.from_b(rb), a)
        rc_status, r_ai, rc_rej, _, _, rc_log = _verify(job, [a.peak], a.peak, images, r_ai, [])
        if rc_status == "deadline":
            r_ai = []  # optional round out of time: keep the answer we have
        second = body.pick(r_ai, [], a, raw_count=raw_count, when=when)
        r_all = r_rejected + rc_rej + second.rejected
        job.log["peak_rewrite"] = {"rejected": r_all, "chosen": second.text,
                                   "source": second.source,
                                   "c": dict(rc_log, status=rc_status)}
        if body.cites_peak(second, a):
            detail = body.BodyPick(text=second.text, source="llm",
                                   rejected=first_rejected + r_all,
                                   photo_ids=second.photo_ids, score=second.score,
                                   style=second.style)
    if not body.cites_peak(detail, a) and detail.source == "llm":
        # Only a model sentence citing a usable photo (selected, unflagged)
        # can take the cover; the rest give way to the next survivor, and a
        # template winner keeps the cover. F0 only when nothing is left.
        def usable(c) -> Optional[str]:
            return next((pid for pid in c.photo_ids
                         if pid in a.selected and not a.photos[pid].flags), None)

        dropped = [(c.text, "no_cover_photo") for c in ai if usable(c) is None]
        detail = body.pick([c for c in ai if usable(c) is not None], tpls, a,
                           raw_count=raw_count, when=when)
        detail.rejected = first_rejected + [d for d in dropped if d not in first_rejected]
        new_peak = None
        if detail.source == "llm":
            new_peak = next((pid for pid in detail.photo_ids
                             if pid in a.selected and not a.photos[pid].flags), None)
        if new_peak is not None:
            a.peak = new_peak
            peak_switched = True
            daypart, when = _when(meta, by_id, a.peak, names)
    daypart_name = when.daypart
    job.log["peakRewrite"] = peak_rewrite
    job.log["peakSwitched"] = peak_switched
    if peak_switched:
        job.log["peak_switch"] = {"from": old_peak, "to": a.peak, "daypart": daypart}

    # ---- questions: the survivors in the model's order, or B1-B3 ----
    final_qs = questions.final(qs, daypart_name)
    job.log["body"] = {"rejected": detail.rejected, "chosen": detail.text,
                       "style": detail.style, "source": detail.source,
                       "templates": [t.text for t in tpls]}
    job.log["question"] = {"ranked": [{"text": q.text, "type": q.type, "why": q.why}
                                      for q in model_qs],
                           "rejected": q_rejected,
                           "final": [q.payload() for q in final_qs]}
    job.log["bodySource"] = detail.source
    job.log["questionSource"] = final_qs[0].source

    # ---- memory ----
    md = date_label(meta.get("date"))
    place = _place(meta)
    fact_line = names["fact_line"].format(md=md, place=place) if place else md
    others = [pid for pid in input_ids if pid in a.selected and pid != a.peak]
    story = [{"id": a.peak, "takenAt": by_id[a.peak].get("taken_at"), "isPeak": True}]
    story += [{"id": pid, "takenAt": by_id[pid].get("taken_at"), "isPeak": False} for pid in others]
    is_f0 = detail.source == "fallback_F0"
    memory = {
        "id": meta.get("group_key"),
        "mode": "life_fragment",
        "date": meta.get("date"),
        "daypart": daypart,
        "bucket": bucket_name(meta),
        "push": {
            "title": names["push_title"].format(fact=fact_line, daypart=daypart_name),
            "body": detail.text,
            "coverPhotoId": a.peak,
        },
        "story": {"firstFrameLabel": fact_line, "photos": story},
        "ending": {
            "factLine": fact_line,
            "detail": "" if is_f0 else detail.text,
            "questions": [q.payload() for q in final_qs],
            "ctaLabel": names["cta"].format(daypart=daypart_name),
        },
        "facts": {pid: _facts_line(a.photos[pid], a, names) for pid in a.selected},
        "generation": {
            "bodySource": detail.source,
            "questionSource": final_qs[0].source,
            "copyVersion": config.COPY_VERSION,
            "model": config.LLM_MODEL,
            "costCny": round(job.cost(), 4),
            "latencyMs": int((time.monotonic() - job.started) * 1000),
            "peakSwitched": peak_switched,
            "peakRewrite": peak_rewrite,
        },
    }
    return job.finish({"status": "ok", "memory": memory})

"""The compose pipeline (AI doc 0):

    call A -> check A -> templates + question menus -> call B -> check B
      -> pick detail sentence + question -> memory (handoff 5.2)

run() returns exactly the job result body. It never raises for a model or
check failure; every outcome is one of ok / rejected / error. One JSON line
per job goes to config.LOG_PATH (AI doc 6) -- text only, no image bytes.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import body
import check_a
import config
import llm
import questions
import texts

ATTEMPTS = 2  # each call is tried once more after a failure

# Phone label (15 broad categories) -> push bucket key, handoff 7.3.
_LABEL_BUCKET = {
    "food_drink": "food",
    "sky": "sky",
    "plant": "nature",
    "water": "nature",
    "animal": "animal",
    "people": "people",
    "built": "city",
    "street": "city",
    "vehicle": "city",
    "interior": "home",
    "tableware": "home",
    "event": "event",
    "recreation": "event",
}
_BUCKET_ORDER = ["food", "sky", "nature", "animal", "people", "city", "home", "event", "other"]

_log_lock = threading.Lock()


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


def bucket(photos: List[dict]) -> str:
    """Push bucket by majority of the sent photos' labels (each bucket counted
    once per photo); ties go to the earlier bucket in handoff 7.3's order."""
    counts: Dict[str, int] = {}
    for p in photos:
        seen = set()
        for label in p.get("labels") or []:
            seen.add(_LABEL_BUCKET.get(label, "other"))
        for b in seen:
            counts[b] = counts.get(b, 0) + 1
    key = "other"
    if counts:
        key = max(_BUCKET_ORDER, key=lambda b: (counts.get(b, 0), -_BUCKET_ORDER.index(b)))
    return texts.json("names")["buckets"][key]


def _place(meta: dict) -> Optional[str]:
    place = meta.get("place")
    if isinstance(place, str) and place.strip():
        return place.strip()
    return None


def _result(status: str, reason: str) -> dict:
    return {"status": status, "reason": reason, "message": texts.message(reason)}


# ---------- prompts ----------

def _messages_a(meta: dict, photos: List[dict], images: Dict[str, bytes]) -> List[dict]:
    names = texts.json("names")
    raw = texts.raw("collect_a_user")
    header, _, rest = raw.partition("=====PHOTO=====\n")
    photo_tpl, _, task = rest.partition("=====TASK=====\n")
    photo_tpl = photo_tpl.strip("\n")
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
        dt = parse_time(p.get("taken_at"))
        represents = int(p.get("represents") or 1)
        content.append({"type": "text", "text": photo_tpl.format_map({
            "id": p["id"],
            "time": dt.strftime("%H:%M") if dt else "",
            "labels": names["list_sep"].join(p.get("labels") or []) or names["none"],
            "faces": int(p.get("faces") or 0),
            "favorite": names["a_favorite"] if p.get("favorite") is True else "",
            "represents": represents,
            "merged": names["a_merged"] if represents > 1 else "",
        })})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(images[p["id"]]).decode("ascii")}})
    content.append({"type": "text", "text": task.format_map(
        {"schema_a": texts.raw("schema_a_example").strip()})})
    return [
        {"role": "system", "content": texts.raw("collect_a_system").strip()},
        {"role": "user", "content": content},
    ]


def _messages_b(meta: dict, a: check_a.AResult, daypart: str, counts: Dict[str, int],
                cands: List[questions.QCandidate]) -> List[dict]:
    names = texts.json("names")
    none = names["none"]
    type_names = names["subject_types"]
    lines = []
    for pid in a.selected:
        p = a.photos[pid]
        subjects = names["list_sep"].join(
            names["b_subject"].format(name=s.name, type=type_names.get(s.type, s.type))
            for s in p.subjects)
        lines.append(names["b_photo_line"].format(
            id=pid,
            subjects=subjects or none,
            details=names["list_sep"].join(p.details) or none,
            text=names["list_sep"].join(p.text) or none,
            people=p.people_count,
        ))
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    count_text = names["sep"].join(names["b_count"].format(name=k, n=v) for k, v in ordered)
    user = texts.render(
        "collect_b_user",
        date=meta.get("date"),
        daypart=names["dayparts"][daypart],
        place=_place(meta) or none,
        k=len(a.selected),
        peak_id=a.peak,
        counts=count_text or none,
        photo_lines="\n".join(lines),
        question_block=questions.prompt_block(cands),
    )
    return [
        {"role": "system", "content": texts.raw("collect_b_system").strip()},
        {"role": "user", "content": user},
    ]


def _messages_b_peak(meta: dict, a: check_a.AResult, daypart: str,
                     counts: Dict[str, int]) -> List[dict]:
    """The extra call B: detail sentences about the cover photo only."""
    names = texts.json("names")
    none = names["none"]
    type_names = names["subject_types"]
    p = a.photos[a.peak]
    subjects = names["list_sep"].join(
        names["b_subject"].format(name=s.name, type=type_names.get(s.type, s.type))
        for s in p.subjects)
    peak_line = names["b_photo_line"].format(
        id=a.peak, subjects=subjects or none,
        details=names["list_sep"].join(p.details) or none,
        text=names["list_sep"].join(p.text) or none, people=p.people_count)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    count_text = names["sep"].join(names["b_count"].format(name=k, n=v) for k, v in ordered)
    user = texts.render(
        "collect_b_peak_user",
        date=meta.get("date"),
        daypart=names["dayparts"][daypart],
        place=_place(meta) or none,
        peak_id=a.peak,
        counts=count_text or none,
        peak_line=peak_line,
    )
    return [
        {"role": "system", "content": texts.raw("collect_b_system").strip()},
        {"role": "user", "content": user},
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
        self.log: Dict[str, Any] = {"group_key": meta.get("group_key")}

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
            left = self.remaining()
            if left <= 0:
                return None, "deadline"
            entry = {"call": which, "attempt": attempt, "latency_ms": 0, "prompt_tokens": 0,
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
        if self.remaining() <= 0:
            return None, "deadline"
        return None, last

    def finish(self, result: dict) -> dict:
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
        with _log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as error:  # noqa: BLE001 -- a log failure never fails a job
        print("[collections] could not write the compose log: {}".format(error))


def _facts_line(p: check_a.PhotoFacts, names: dict) -> str:
    none = names["none"]
    return names["facts_line"].format(
        subjects=names["sep"].join(s.name for s in p.subjects) or none,
        details=names["sep"].join(p.details) or none,
        people=p.people_count,
    )


def run(meta: dict, images: Dict[str, bytes], deadline: float) -> dict:
    """Compose one memory. meta and images are already validated."""
    job = _Job(meta, deadline)
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
    peak_time = parse_time(by_id[a.peak].get("taken_at"))
    daypart = daypart_of(peak_time.hour if peak_time else 12)
    daypart_name = names["dayparts"][daypart]
    counts = body.program_counts(a)
    tpls = body.templates(a, meta)
    cands = questions.candidates(a, meta, daypart)

    # ---- call B (failing twice is not a rejection: templates + fallback) ----
    b, fail = job.call("b", _messages_b(meta, a, daypart, counts, cands),
                       config.CALL_B_TIMEOUT, _check_b)
    if b is None and fail == "deadline":
        return job.finish(_result("error", "timeout"))
    ai = body.from_b(b)
    ranked = (b or {}).get("question", {}).get("ranked") or []

    # ---- detail sentence ----
    raw_count = int(meta.get("raw_count") or 0)
    detail = body.pick(ai, tpls, a, counts, raw_count=raw_count)
    first_rejected = list(detail.rejected)

    # The body must describe the cover (user decision 2026-10-31): no model
    # sentence about the peak survived -> one extra call B for the peak only
    # (no retry, inside the job deadline); still none -> a model sentence
    # about another photo makes that photo the cover; a template or F0
    # keeps the cover.
    peak_rewrite = False
    peak_switched = False
    old_peak = a.peak
    if not body.cites_peak(detail, a) and job.remaining() > 0:
        peak_rewrite = True
        rb, _ = job.call("b", _messages_b_peak(meta, a, daypart, counts),
                         config.CALL_B_TIMEOUT, _check_rewrite, attempts=1)
        second = body.pick(body.from_b(rb), [], a, counts, raw_count=raw_count,
                           scope=[a.peak])
        job.log["peak_rewrite"] = {"rejected": second.rejected, "chosen": second.text,
                                   "source": second.source}
        if body.cites_peak(second, a):
            detail = body.BodyPick(text=second.text, source="llm",
                                   rejected=first_rejected + second.rejected,
                                   photo_ids=second.photo_ids, score=second.score)
    if not body.cites_peak(detail, a) and detail.source == "llm":
        # Only a model sentence citing a usable photo (selected, unflagged)
        # can take the cover; the rest give way to the next survivor, and a
        # template winner keeps the cover. F0 only when nothing is left.
        def usable(c) -> Optional[str]:
            return next((pid for pid in c.photo_ids
                         if pid in a.selected and not a.photos[pid].flags), None)

        dropped = [(c.text, "no_cover_photo") for c in ai if usable(c) is None]
        detail = body.pick([c for c in ai if usable(c) is not None], tpls, a, counts,
                           raw_count=raw_count)
        detail.rejected = first_rejected + [d for d in dropped if d not in first_rejected]
        new_peak = None
        if detail.source == "llm":
            new_peak = next((pid for pid in detail.photo_ids
                             if pid in a.selected and not a.photos[pid].flags), None)
        if new_peak is not None:
            a.peak = new_peak
            peak_switched = True
            peak_time = parse_time(by_id[a.peak].get("taken_at"))
            daypart = daypart_of(peak_time.hour if peak_time else 12)
            daypart_name = names["dayparts"][daypart]
    job.log["peakRewrite"] = peak_rewrite
    job.log["peakSwitched"] = peak_switched
    if peak_switched:
        job.log["peak_switch"] = {"from": old_peak, "to": a.peak, "daypart": daypart}

    # ---- question (offered questions rebuilt for a switched cover: menus,
    # conditions and the daypart all follow the new peak; a ranked id that
    # is no longer offered falls through to B1-B3) ----
    final_cands = questions.candidates(a, meta, daypart) if peak_switched else cands
    q = questions.pick(ranked, final_cands, list(meta.get("recent_question_ids") or []),
                       daypart_name)
    job.log["body"] = {"rejected": detail.rejected, "chosen": detail.text,
                       "source": detail.source, "templates": [t.text for t in tpls]}
    job.log["question"] = {"offered": [c.id for c in cands],
                           "offered_after_switch": [c.id for c in final_cands],
                           "ranked": ranked,
                           "rejected": q.rejected, "chosen_id": q.id, "chosen": q.text,
                           "fill": q.fill, "source": q.source}
    job.log["bodySource"] = detail.source
    job.log["questionSource"] = q.source

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
        "bucket": bucket(photos),
        "push": {
            "title": names["push_title"].format(fact=fact_line, daypart=daypart_name),
            "body": detail.text,
            "coverPhotoId": a.peak,
        },
        "story": {"firstFrameLabel": fact_line, "photos": story},
        "ending": {
            "factLine": fact_line,
            "detail": "" if is_f0 else detail.text,
            "question": q.text,
            "questionId": q.id,
            "ctaLabel": names["cta"].format(daypart=daypart_name),
        },
        "facts": {pid: _facts_line(a.photos[pid], names) for pid in a.selected},
        "generation": {
            "bodySource": detail.source,
            "questionSource": q.source,
            "model": config.LLM_MODEL,
            "costCny": round(job.cost(), 4),
            "latencyMs": int((time.monotonic() - job.started) * 1000),
            "peakSwitched": peak_switched,
            "peakRewrite": peak_rewrite,
        },
    }
    return job.finish({"status": "ok", "memory": memory})

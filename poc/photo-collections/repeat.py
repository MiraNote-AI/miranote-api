"""Repeat themes: compose a push for a theme the phone found across days.

Call A describes every photo (the life-fragment call A without the
selection step); the days rule then needs at least REPEAT_MIN_DAYS distinct
local days among the photos it keeps. Calls B and C follow in a later task.
"""

from __future__ import annotations

import traceback
from typing import Dict, List

import check_a
import compose
import config
import texts


def local_days(photos: List[dict]) -> Dict[str, str]:
    """Photo id -> local yyyy-MM-dd, read from taken_at as the phone sent it
    (the string carries the phone's offset; it is not converted to UTC)."""
    return {p["id"]: str(p["taken_at"]).strip()[:10] for p in photos}


def _messages_a(meta: dict, photos: List[dict], images: Dict[str, bytes]) -> List[dict]:
    names = texts.json("names")
    header, photo_tpl, task = compose._split("repeat_a_user")
    theme = meta["theme"]
    content: List[dict] = [{"type": "text", "text": header.format_map({
        "theme_name": theme["name"],
        "same_part": theme["same_part"],
        "sent_count": len(photos),
    }).strip("\n")}]
    for p in photos:
        content.append({"type": "text", "text": photo_tpl.format_map({
            "id": p["id"],
            "date": str(p["taken_at"]).strip()[:10],
            "time": compose._hhmm(p),
            "labels": names["list_sep"].join(p.get("labels") or []) or names["none"],
            "faces": int(p.get("faces") or 0),
            "favorite": names["a_favorite"] if p.get("favorite") is True else "",
        })})
        content.append(compose._image_part(images[p["id"]]))
    content.append({"type": "text", "text": task.format_map(
        {"schema_a": texts.raw("repeat_schema_a_example").strip()})})
    return [
        {"role": "system", "content": texts.raw("collect_a_system").strip()},
        {"role": "user", "content": content},
    ]


def run(meta: dict, images: Dict[str, bytes], deadline: float) -> dict:
    """Compose one repeat-theme push. Never raises (as compose.run)."""
    job = compose._Job(meta, deadline)
    job.log["mode"] = "repeat"
    job.log["theme"] = meta["theme"].get("id")
    try:
        return _run(job, meta, images)
    except Exception as error:  # noqa: BLE001 -- a bug must not lose the log
        traceback.print_exc()
        job.log["exception"] = "{}: {}".format(type(error).__name__, str(error)[:300])
        return job.finish(compose._result("error", "model_error"))


def _run(job: "compose._Job", meta: dict, images: Dict[str, bytes]) -> dict:
    photos = sorted(meta["photos"], key=lambda p: compose._sort_key(compose.parse_time(p.get("taken_at"))))
    input_ids = [p["id"] for p in photos]
    days = local_days(photos)

    a, fail = job.call(
        "a", _messages_a(meta, photos, images), config.CALL_A_TIMEOUT,
        lambda text: check_a.validate_repeat(check_a.parse_json(text), input_ids, days),
    )
    if a is None:
        reason = {"deadline": "timeout", "timeout": "timeout", "model_error": "model_error",
                  "check_failed": "check_failed"}[fail]
        status = "rejected" if reason == "check_failed" else "error"
        return job.finish(compose._result(status, reason))
    job.log["a"] = {"verdict": a.verdict, "selected": a.selected, "peak": a.peak,
                    "peak_replaced_from": a.peak_replaced_from, "skip_reason": a.skip_reason,
                    "photos": {pid: check_a.log_facts(f) for pid, f in a.photos.items()}}
    if a.verdict != "ok":
        return job.finish(compose._result("rejected", a.verdict))

    # Calls B and C arrive with the next task; until then a good call A ends here.
    job.log["stage"] = "a_only"
    return job.finish(compose._result("error", "model_error"))

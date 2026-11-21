"""Repeat themes: compose a push for a theme the phone found across days
(handoff 6.9, decisions 2026-11-20 H1-H4).

Call A describes every photo (the life-fragment call A without the
selection step); the days rule then needs at least REPEAT_MIN_DAYS distinct
local days among the photos it keeps. Calls B and C and the shorten rounds
run as for a life fragment (compose.draft) with the repeat prompts, word
list, question types and templates T1 -> T2 -> F0. There is no cover
rewrite and no cover switch: a sentence about the theme fits the cover.
"""

from __future__ import annotations

import time
import traceback
from typing import Dict, List

import body
import check_a
import compose
import config
import questions
import rounds
import texts

MODE = "repeat"


def local_days(photos: List[dict]) -> Dict[str, str]:
    """Photo id -> local yyyy-MM-dd, read from taken_at as the phone sent it
    (the string carries the phone's offset; it is not converted to UTC)."""
    return {p["id"]: str(p["taken_at"]).strip()[:10] for p in photos}


def month_range(taken_at: List[str]) -> str:
    """The local months of these ISO times as the title shows them
    (names.txt month_one, month_span, or month_years when they cross a year,
    "2025.12-2026.2"). Each month is read with the string's own offset, not
    in UTC; "" when none parses."""
    names = texts.json("names")
    months = sorted({(dt.year, dt.month) for dt in map(compose.parse_time, taken_at) if dt})
    if not months:
        return ""
    (y1, m1), (y2, m2) = months[0], months[-1]
    if (y1, m1) == (y2, m2):
        return names["month_one"].format(m=m1)
    if y1 == y2:
        return names["month_span"].format(m1=m1, m2=m2)
    return names["month_years"].format(y1=y1, m1=m1, y2=y2, m2=m2)


def _messages_a(meta: dict, photos: List[dict], images: Dict[str, bytes]) -> List[dict]:
    names = texts.json("names")
    header, photo_tpl, task = compose.split_prompt("repeat_a_user")
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
            "time": compose.hhmm(p),
            "labels": names["list_sep"].join(p.get("labels") or []) or names["none"],
            "faces": int(p.get("faces") or 0),
            "favorite": names["a_favorite"] if p.get("favorite") is True else "",
        })})
        content.append(compose.image_part(images[p["id"]]))
    content.append({"type": "text", "text": task.format_map(
        {"schema_a": texts.raw("repeat_schema_a_example").strip()})})
    return [
        {"role": "system", "content": texts.raw("repeat_a_system").strip()},
        {"role": "user", "content": content},
    ]


def _messages_b(meta: dict, a: check_a.AResult, by_id: Dict[str, dict],
                images: Dict[str, bytes], months: str) -> List[dict]:
    """Call B sees the kept photos in time order with call A's descriptions,
    the theme, its months and where it is concentrated (H1)."""
    names = texts.json("names")
    theme = meta["theme"]
    header, photo_tpl, task = compose.split_prompt("repeat_b_user")
    where = rounds.where_line(theme["where"], names) if theme.get("where") else ""
    content: List[dict] = [{"type": "text", "text": header.format_map({
        "theme_name": theme["name"],
        "noun": (theme.get("noun") or "").strip() or names["none"],
        "same_part": theme["same_part"],
        "months": months,
        "where": where or names["none"],
        "k": len(a.selected),
    }).strip("\n")}]
    for pid in a.selected:
        p = a.photos[pid]
        content.append({"type": "text", "text": photo_tpl.format_map({
            "id": pid, "date": str(by_id[pid]["taken_at"]).strip()[:10],
            "time": compose.hhmm(by_id[pid]),
            "cover": names["b_cover"] if pid == a.peak else "",
            "description": p.description or names["none"], "people": p.people_count,
        })})
        content.append(compose.image_part(images[pid]))
    content.append({"type": "text", "text": task.format_map({
        "type_table": questions.prompt_types(MODE),
        "recent_types": questions.prompt_recent(list(meta.get("recent_question_types") or []), MODE),
    })})
    return [
        {"role": "system", "content": texts.raw(compose.B_SYSTEM[MODE]).strip()},
        {"role": "user", "content": content},
    ]


def run(meta: dict, images: Dict[str, bytes], deadline: float) -> dict:
    """Compose one repeat-theme push. Never raises (as compose.run)."""
    job = compose.Job(meta, deadline)
    job.log["mode"] = MODE
    job.log["theme"] = meta["theme"].get("id")
    try:
        return _run(job, meta, images)
    except Exception as error:  # noqa: BLE001 -- a bug must not lose the log
        traceback.print_exc()
        job.log["exception"] = "{}: {}".format(type(error).__name__, str(error)[:300])
        return job.finish(compose.result_body("error", "model_error"))


def _run(job: "compose.Job", meta: dict, images: Dict[str, bytes]) -> dict:
    names = texts.json("names")
    theme = meta["theme"]
    photos = sorted(meta["photos"], key=lambda p: compose.sort_key(compose.parse_time(p.get("taken_at"))))
    input_ids = [p["id"] for p in photos]
    by_id = {p["id"]: p for p in photos}
    days = local_days(photos)

    # ---- call A ----
    a, fail = job.call(
        "a", _messages_a(meta, photos, images), config.CALL_A_TIMEOUT,
        lambda text: check_a.validate_repeat(check_a.parse_json(text), input_ids, days),
    )
    if a is None:
        return job.finish(compose.a_failure(fail))
    job.log_a(a)
    if a.verdict != "ok":
        return job.finish(compose.result_body("rejected", a.verdict))

    # ---- program prep: the page is the kept photos ----
    kept = a.selected
    months = month_range([by_id[pid]["taken_at"] for pid in kept])
    title = names["repeat_title"].format(name=theme["name"], months=months)
    noun = (theme.get("noun") or "").strip()
    peak_time = compose.parse_time(by_id[a.peak].get("taken_at"))
    daypart = compose.daypart_of(peak_time.hour if peak_time else 12)
    when = body.When(weekday="", daypart=names["dayparts"][daypart])
    tpls = body.templates_repeat(theme, kept)
    f0 = tpls.pop().text  # F0 is always last
    job.log["title"] = title

    images = dict(images)
    for pid in kept:
        images[pid] = compose.shrink(images[pid], config.SMALL_IMAGE_PX)

    # ---- calls B and C, the shorten rounds (no cover rewrite, H2) ----
    d = compose.draft(job, a, images, _messages_b(meta, a, by_id, images, months), tpls,
                      raw_count=len(kept), when=when, mode=MODE, f0=f0,
                      c_values={"theme_name": theme["name"], "same_part": theme["same_part"]})
    detail = d.detail
    final_qs = questions.final(d.qs, noun, MODE)
    job.log["peakRewrite"] = False
    job.log["peakSwitched"] = False
    job.log["body"] = {"rejected": detail.rejected, "chosen": detail.text,
                       "style": detail.style, "source": detail.source,
                       "templates": [t.text for t in tpls] + [f0]}
    job.log["question"] = {"ranked": [{"text": q.text, "type": q.type, "why": q.why}
                                      for q in d.model_qs],
                           "rejected": d.q_rejected,
                           "final": [q.payload() for q in final_qs]}
    job.log["bodySource"] = detail.source
    job.log["questionSource"] = final_qs[0].source

    # ---- memory: peak first, then the rest by time ----
    story = [{"id": a.peak, "takenAt": by_id[a.peak].get("taken_at"), "isPeak": True}]
    story += [{"id": pid, "takenAt": by_id[pid].get("taken_at"), "isPeak": False}
              for pid in kept if pid != a.peak]
    memory = {
        "id": meta.get("group_key"),
        "mode": MODE,
        "date": days[kept[-1]],
        "daypart": daypart,
        "bucket": "",
        "themeId": theme["id"],
        "push": {"title": title, "body": detail.text, "coverPhotoId": a.peak},
        "story": {"firstFrameLabel": title, "photos": story},
        "ending": {
            "factLine": title,
            "detail": "" if detail.source == "fallback_F0" else detail.text,
            "questions": [q.payload() for q in final_qs],
            "ctaLabel": names["repeat_cta"].format(noun=noun or names["repeat_noun_fallback"]),
        },
        "facts": {pid: compose.facts_line(a.photos[pid], a, names) for pid in kept},
        "generation": {
            "bodySource": detail.source,
            "questionSource": final_qs[0].source,
            "copyVersion": config.REPEAT_COPY_VERSION,
            "model": config.LLM_MODEL,
            "costCny": round(job.cost(), 4),
            "latencyMs": int((time.monotonic() - job.started) * 1000),
            "peakSwitched": False,
            "peakRewrite": False,
        },
    }
    return job.finish({"status": "ok", "memory": memory})

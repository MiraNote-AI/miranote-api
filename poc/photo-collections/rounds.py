"""Repeat themes: the judge rounds the phone asks Qwen through this service.

Round 1 names a pile of photos from many days and picks how to split it;
round 2 checks push candidates before a push; shorten asks for shorter
names. The reply goes back to the phone verbatim -- the phone reads and
checks it. Nothing here is metered, and no image bytes reach the log.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import compose
import config
import llm
import texts

DAYPARTS = ("morning", "noon", "afternoon", "evening", "night")
WEEKPARTS = ("weekday", "weekend", "mixed")
MAX_GROUPS = 6
MAX_LABELS = 8
MAX_NAME_CHARS = 60  # an old or pushed name; the phone sends at most 10-character names
MAX_TEXT_CHARS = 200  # shorten input is long on purpose
MAX_OLD_GROUPS = 12
MAX_PUSHED = 20
MAX_BLOCKED = 50
MAX_SHORTEN = 7

_GROUP_A = re.compile(r"^A[0-6]$")
_GROUP_B = re.compile(r"^B[0-6]$")

# round kind -> model call kind
CALLS = {"round1": "r1", "round2": "r2", "shorten": "rs"}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_text(value, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _load(raw) -> Optional[dict]:
    if not isinstance(raw, str):
        return None
    try:
        meta = json.loads(raw)
    except ValueError:
        return None
    return meta if isinstance(meta, dict) else None


def check_where(where) -> bool:
    if not isinstance(where, dict):
        return False
    if where.get("daypart") is not None and where["daypart"] not in DAYPARTS:
        return False
    share = where.get("daypart_share")
    if share is not None and not (_is_num(share) and 0 <= share <= 1):
        return False
    for key in ("hour_from", "hour_to"):
        hour = where.get(key)
        if hour is not None and not (_is_int(hour) and 0 <= hour <= 23):
            return False
    if where.get("weekpart") not in WEEKPARTS:
        return False
    places = where.get("places")
    return places is None or (_is_int(places) and places >= 0)


MAX_TAGS = 8


def _check_tags(tags) -> bool:
    if not isinstance(tags, list) or len(tags) > MAX_TAGS:
        return False
    for t in tags:
        if not isinstance(t, dict) or not isinstance(t.get("label"), str) or not 1 <= len(t["label"]) <= 60:
            return False
        if not all(_is_num(t.get(k)) and 0 <= t[k] <= 1 for k in ("in", "out")):
            return False
    return True


def _check_split(split, pattern) -> bool:
    if not isinstance(split, list) or len(split) > MAX_GROUPS:
        return False
    ids = []
    for g in split:
        if not isinstance(g, dict) or not isinstance(g.get("id"), str) or not pattern.match(g["id"]):
            return False
        if g["id"].endswith("0") or not (_is_int(g.get("days")) and g["days"] >= 0):
            return False
        if not check_where(g.get("where")):
            return False
        if g.get("tags") is not None and not _check_tags(g["tags"]):
            return False
        ids.append(g["id"])
    return len(set(ids)) == len(ids)


def check_round1(raw: str) -> Optional[dict]:
    """The parsed round-1 meta when it matches the contract, else None."""
    meta = _load(raw)
    if meta is None:
        return None
    pile = meta.get("pile")
    if not isinstance(pile, dict) or not all(_is_int(pile.get(k)) and pile[k] >= 1
                                              for k in ("photo_count", "day_count")):
        return None
    if not _check_split(meta.get("split_a"), _GROUP_A) or not _check_split(meta.get("split_b"), _GROUP_B):
        return None
    photos = meta.get("photos")
    if not isinstance(photos, list) or not 1 <= len(photos) <= config.ROUND1_MAX_PHOTOS:
        return None
    for p in photos:
        if not isinstance(p, dict) or not _is_int(p.get("n")):
            return None
        if not isinstance(p.get("a"), str) or not _GROUP_A.match(p["a"]):
            return None
        if not isinstance(p.get("b"), str) or not _GROUP_B.match(p["b"]):
            return None
        labels = p.get("labels")
        if labels is not None and not (
                isinstance(labels, list) and len(labels) <= MAX_LABELS
                and all(isinstance(v, str) and len(v) <= MAX_NAME_CHARS for v in labels)):
            return None
    if sorted(p["n"] for p in photos) != list(range(1, len(photos) + 1)):
        return None
    old = meta.get("old")
    if old is not None:
        if not isinstance(old, dict) or not _is_text(old.get("pile_name"), MAX_NAME_CHARS):
            return None
        groups = old.get("groups")
        if not (isinstance(groups, list) and len(groups) <= MAX_OLD_GROUPS
                and all(_is_text(g, MAX_NAME_CHARS) for g in groups)):
            return None
    return meta


def check_round2(raw: str) -> Optional[dict]:
    """The parsed round-2 meta when it matches the contract, else None."""
    meta = _load(raw)
    if meta is None:
        return None
    theme = meta.get("theme")
    if not isinstance(theme, dict) or not _is_text(theme.get("name"), MAX_TEXT_CHARS):
        return None
    for k in ("noun", "same_part"):  # a pile has neither; empty is fine
        v = theme.get(k, "")
        if not (isinstance(v, str) and len(v) <= MAX_TEXT_CHARS):
            return None
    photos = meta.get("photos")
    if not isinstance(photos, list) or not 1 <= len(photos) <= config.ROUND2_MAX_PHOTOS:
        return None
    ns = []
    for p in photos:
        if not isinstance(p, dict) or not (_is_int(p.get("n")) and p["n"] >= 1):
            return None
        ns.append(p["n"])
    if len(set(ns)) != len(ns):
        return None
    for key, limit in (("pushed", MAX_PUSHED), ("blocked", MAX_BLOCKED)):
        names = meta.get(key)
        if not (isinstance(names, list) and len(names) <= limit
                and all(_is_text(v, MAX_NAME_CHARS) for v in names)):
            return None
    return meta


def check_shorten(body) -> Optional[dict]:
    """The body when it is 1-7 items of key, name, same_part; else None."""
    if not isinstance(body, dict):
        return None
    items = body.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_SHORTEN:
        return None
    keys = []
    for item in items:
        if not isinstance(item, dict) or not all(
                _is_text(item.get(k), MAX_TEXT_CHARS) for k in ("key", "name")):
            return None
        same = item.get("same_part")  # a pile has none: "" is fine
        if not (isinstance(same, str) and len(same) <= MAX_TEXT_CHARS):
            return None
        keys.append(item["key"])
    return body if len(set(keys)) == len(keys) else None


def photo_numbers(meta: dict) -> List[int]:
    return [p["n"] for p in meta["photos"]]


# ---------- prompts ----------

def where_line(where: dict, names: dict) -> str:
    """One group's "where it is concentrated" line, parts left out when unknown."""
    parts = []
    if where.get("daypart"):
        parts.append(names["where_daypart"].format(daypart=names["dayparts"][where["daypart"]]))
    if where.get("hour_from") is not None and where.get("hour_to") is not None:
        parts.append(names["where_hours"].format(**{"from": where["hour_from"], "to": where["hour_to"]}))
    parts.append(names["where_weekpart"][where["weekpart"]])
    places = where.get("places")
    table = names["where_places"]
    if places is None:
        parts.append(table["null"])
    elif places >= 2:
        parts.append(table["many"].format(n=places))
    else:
        parts.append(table[str(places)])
    return names["where_sep"].join(p for p in parts if p)


def group_where(meta: dict, names: dict) -> Dict[str, str]:
    return {g["id"]: where_line(g["where"], names) for g in meta["split_a"] + meta["split_b"]}


def _split_lines(head_key: str, split: List[dict], names: dict, wheres: Dict[str, str],
                 with_tags: bool = False) -> List[str]:
    if not split:
        return []
    ids = names["sep"].join(g["id"] for g in split)
    days = names["sep"].join(str(g["days"]) for g in split)
    out = [names[head_key].format(ids=ids, days=days)]
    for g in split:
        out.append(names["round1_group_line"].format(id=g["id"], where=wheres[g["id"]]))
        if with_tags:
            out += [names["round1_tag_line"].format(label=t["label"], inside=round(t["in"] * 100),
                                                    outside=round(t["out"] * 100))
                    for t in g.get("tags") or []]
    return out


def _tagged(numbers: List[int], images: Dict[int, bytes], names: dict) -> List[dict]:
    parts: List[dict] = []
    for n in numbers:
        parts.append({"type": "text", "text": names["photo_tag"].format(n=n)})
        parts.append(compose.image_part(images[n]))
    return parts


def _messages(text: str, tail: List[dict]) -> List[dict]:
    return [
        {"role": "system", "content": texts.raw("repeat_round_system").strip()},
        {"role": "user", "content": [{"type": "text", "text": text}] + tail},
    ]


def messages_round1(meta: dict, images: Dict[int, bytes]) -> List[dict]:
    names = texts.json("names")
    wheres = group_where(meta, names)
    photos = sorted(meta["photos"], key=lambda p: p["n"])
    lines = [names["round1_pile"].format(
        count=meta["pile"]["photo_count"], days=meta["pile"]["day_count"], sent=len(photos))]
    lines += _split_lines("round1_split_a", meta["split_a"], names, wheres, with_tags=True)
    lines += _split_lines("round1_split_b", meta["split_b"], names, wheres)
    if not meta["split_a"] and not meta["split_b"]:
        lines.append(names["round1_no_groups"])
    lines.append(names["round1_photos_head"].format(items=names["sep"].join(
        names["round1_photo_line"].format(n=p["n"], a=p["a"], b=p["b"]) for p in photos)))
    labelled = [p for p in photos if p.get("labels")]
    if labelled:
        lines.append(names["round1_labels_head"])
        lines += [names["round1_label_line"].format(n=p["n"], labels=names["list_sep"].join(p["labels"]))
                  for p in labelled]
    text = texts.render("repeat_round1_user", input="\n".join(lines),
                        example=texts.raw("repeat_round1_example").strip())
    old = meta.get("old")
    if old:
        text = texts.render("repeat_round1_reask", pile_name=old["pile_name"],
                            groups=names["reask_join"].join(old["groups"])) + "\n" + text
    return _messages(text, _tagged([p["n"] for p in photos], images, names))


def messages_round2(meta: dict, images: Dict[int, bytes]) -> List[dict]:
    names = texts.json("names")
    theme = meta["theme"]
    numbers = sorted(photo_numbers(meta))
    lines = [names["round2_theme"].format(name=theme["name"], noun=theme.get("noun", "").strip() or names["none"],
                                         same_part=theme.get("same_part", "").strip() or names["none"]),
             names["round2_candidates"].format(count=len(numbers))]
    if meta["pushed"]:
        lines.append(names["round2_pushed"].format(names=names["sep"].join(meta["pushed"])))
    if meta["blocked"]:
        lines.append(names["round2_blocked"].format(names=names["sep"].join(meta["blocked"])))
    text = texts.render("repeat_round2_user", input="\n".join(lines),
                        example=texts.raw("repeat_round2_example").strip())
    return _messages(text, _tagged(numbers, images, names))


def messages_shorten(items: List[dict]) -> List[dict]:
    body = json.dumps({"items": [{"key": i["key"], "name": i["name"], "same_part": i["same_part"]}
                                 for i in items]}, ensure_ascii=False)
    return _messages(texts.render("repeat_shorten_user", items=body), [])


# ---------- the call ----------

def _usage(model: str, prompt: int = 0, completion: int = 0, cost: float = 0.0) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "cost_cny": round(cost, 6), "model": model}


def call(kind: str, messages: List[dict], log: dict) -> Tuple[int, dict]:
    """One model call for `kind` (round1, round2 or shorten): (status, body).
    Writes one log line: the raw reply and the numbers, never an image."""
    model_call = CALLS[kind]
    model = llm.model_for(model_call)
    entry = {"kind": kind, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    entry.update(log)
    started = time.monotonic()
    try:
        reply = llm.complete(messages, call=model_call, timeout=config.ROUND_TIMEOUT)
    except (llm.ModelTimeout, llm.ModelError) as error:
        reason = "timeout" if isinstance(error, llm.ModelTimeout) else "model_error"
        entry.update(status="error", reason=reason, error=str(error)[:300], promptTokens=0,
                     completionTokens=0, costCny=0.0, latencyMs=int((time.monotonic() - started) * 1000))
        compose._write_log(entry)
        return 502, {"status": "error", "reason": reason, "usage": _usage(model)}
    entry.update(status="ok", reason="", promptTokens=reply.prompt_tokens,
                 completionTokens=reply.completion_tokens, costCny=round(reply.cost_cny, 6),
                 latencyMs=reply.latency_ms, reply=reply.text)
    compose._write_log(entry)
    body = {"status": "ok", "reply": reply.text,
            "usage": _usage(model, reply.prompt_tokens, reply.completion_tokens, reply.cost_cny)}
    return 200, body

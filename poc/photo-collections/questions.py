"""The ending question: bank, condition filter, menus for the A skeletons,
the block shown to call B, and the program pick with fallbacks (question bank
doc section 3, AI doc 2.2 and 5). The bank lives in prompts/question_bank.txt.

Condition encoding (the plan's reading of the bank's condition column):
  scenes           >= half of the selected photos have one of these scenes
  people           some selected photo has people_count != "0"
  group_face       some selected photo has "2-4" or "5+" people and a
                   prominent face
  not_night        the daypart is not night
  min_selected     at least this many selected photos
  min_span_minutes end_time - start_time of the group, in minutes
  min_food_photos  at least this many selected photos with a food subject
  menu_types       subject types that may fill {thing}
  drop_common      {thing} menu drops prompts/common_objects.txt
  moment_words     word list for the {moment} menu (default action_verbs);
                   A4 uses sight_verbs -- a program-side narrowing added
                   after H1 round 4, not a bank rule: "who noticed it first"
                   needs a sight (light, a thing), not what people are doing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import texts
from body import char_len
from check_a import AResult

MAX_CHARS = 22
TOP_N = 3


@dataclass
class QCandidate:
    id: str
    kind: str  # A | B
    text: str  # daypart filled in; the A slot token kept
    menu: List[str] = field(default_factory=list)
    slot: Optional[str] = None  # thing | moment | None
    raw: str = ""  # bank text, daypart token kept


@dataclass
class QPick:
    text: str
    id: str
    source: str  # llm | fallback_B1 | fallback_B2 | fallback_B3
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    fill: str = ""


def _bank_doc() -> dict:
    return texts.json("question_bank")


def bank() -> List[dict]:
    return list(_bank_doc()["questions"])


def _tokens() -> dict:
    return texts.json("names")["slot_tokens"]


def _minutes(hhmm) -> Optional[int]:
    try:
        h, m = str(hhmm).split(":")[:2]
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _span_minutes(meta: dict) -> Optional[int]:
    start, end = _minutes(meta.get("start_time")), _minutes(meta.get("end_time"))
    if start is None or end is None:
        return None
    if end < start:  # crossed midnight
        end += 24 * 60
    return end - start


def _peak_first(a: AResult) -> List[str]:
    return [a.peak] + [pid for pid in a.selected if pid != a.peak]


def _thing_menu(a: AResult, types: List[str], drop_common: bool) -> List[str]:
    commons = set(texts.words("common_objects")) if drop_common else set()
    out: List[str] = []
    for pid in _peak_first(a):
        for s in a.photos[pid].subjects:
            if s.type in types and s.name not in commons and s.name not in out:
                out.append(s.name)
    return out


def _moment_menu(a: AResult, words: str = "action_verbs") -> List[str]:
    """Action phrases of the selected photos, the peak's first. Empty (so
    the question is not offered) when the peak has none: a program-side
    narrowing added after H1 round 4 -- without it the only fills were other
    photos' everyday scenes while the peak's own thing was left out."""
    verbs = texts.words(words)
    out: List[str] = []
    for pid in _peak_first(a):
        for d in a.photos[pid].details:
            if any(v in d for v in verbs) and d not in out:
                out.append(d)
        if pid == a.peak and not out:
            return []
    return out


def _cond_met(cond: dict, a: AResult, meta: dict, daypart: str) -> bool:
    sel = [a.photos[pid] for pid in a.selected]
    if "scenes" in cond:
        hits = sum(1 for p in sel if p.scene in cond["scenes"])
        if not sel or hits * 2 < len(sel):
            return False
    if cond.get("people") and not any(p.people_count != "0" for p in sel):
        return False
    if cond.get("group_face") and not any(
        p.people_count in ("2-4", "5+") and p.face_prominent for p in sel
    ):
        return False
    if cond.get("not_night") and daypart == "night":
        return False
    if "min_selected" in cond and len(sel) < cond["min_selected"]:
        return False
    if "min_span_minutes" in cond:
        span = _span_minutes(meta)
        if span is None or span < cond["min_span_minutes"]:
            return False
    if "min_food_photos" in cond:
        food = sum(1 for p in sel if any(s.type == "food" for s in p.subjects))
        if food < cond["min_food_photos"]:
            return False
    return True


def _fill_daypart(text: str, daypart_name: str) -> str:
    return text.replace(_tokens()["daypart"], daypart_name)


def candidates(a: AResult, meta: dict, daypart: str) -> List[QCandidate]:
    """The questions offered to call B for this group: conditions met,
    non-empty menu for A skeletons, recent ids removed."""
    names = texts.json("names")
    daypart_name = names["dayparts"].get(daypart, daypart)
    recent = set(meta.get("recent_question_ids") or [])
    out: List[QCandidate] = []
    for q in bank():
        if q["id"] in recent:
            continue
        cond = q.get("cond") or {}
        if not _cond_met(cond, a, meta, daypart):
            continue
        menu: List[str] = []
        if q["slot"] == "thing":
            menu = _thing_menu(a, cond.get("menu_types") or [], bool(cond.get("drop_common")))
        elif q["slot"] == "moment":
            menu = _moment_menu(a, cond.get("moment_words") or "action_verbs")
        if q["kind"] == "A" and not menu:
            continue
        out.append(QCandidate(
            id=q["id"], kind=q["kind"], text=_fill_daypart(q["text"], daypart_name),
            menu=menu, slot=q["slot"], raw=q["text"],
        ))
    return out


def with_daypart(cands: List[QCandidate], daypart: str) -> List[QCandidate]:
    """The same offered questions with the daypart re-rendered (the cover,
    and so the daypart, can change after call B)."""
    name = texts.json("names")["dayparts"].get(daypart, daypart)
    return [QCandidate(id=c.id, kind=c.kind, text=_fill_daypart(c.raw or c.text, name),
                       menu=list(c.menu), slot=c.slot, raw=c.raw) for c in cands]


def prompt_block(cands: List[QCandidate]) -> str:
    """One line per offered question, A skeletons first, as in AI doc 3.2."""
    names = texts.json("names")
    lines = []
    for c in sorted(cands, key=lambda c: 0 if c.kind == "A" else 1):
        if c.kind == "A":
            lines.append(names["b_question_a"].format(
                id=c.id, text=c.text, menu=names["list_sep"].join(c.menu)))
        else:
            lines.append(names["b_question_b"].format(id=c.id, text=c.text))
    return "\n".join(lines)


def _forbidden(text: str) -> Optional[str]:
    for w in texts.json("names")["forbidden_question_words"]:
        if w in text:
            return w
    return None


def pick(ranked: List[dict], cands: List[QCandidate], recent: List[str],
         daypart_name: str) -> QPick:
    """Check the model's top 3 in order; the first that passes wins.
    None passes -> B1 -> B2 -> B3, skipping recently used ids."""
    tokens = _tokens()
    offered = {c.id: c for c in cands}
    recent_set = set(recent or [])
    rejected: List[Tuple[str, str]] = []
    for r in (ranked or [])[:TOP_N]:
        if not isinstance(r, dict):
            rejected.append((str(r), "not_an_object"))
            continue
        qid = str(r.get("id") or "").strip()
        fill = r.get("fill") if isinstance(r.get("fill"), str) else ""
        fill = fill.strip()
        c = offered.get(qid)
        if c is None:
            rejected.append((qid, "not_offered"))
            continue
        if c.kind == "A":
            if fill not in c.menu:
                rejected.append((qid, "fill_not_in_menu"))
                continue
            text = c.text.replace(tokens[c.slot], fill)
        else:
            if fill:
                rejected.append((qid, "b_with_fill"))
                continue
            text = c.text
        text = _fill_daypart(text, daypart_name)
        if char_len(text) > MAX_CHARS:
            rejected.append((qid, "too_long"))
            continue
        word = _forbidden(text)
        if word:
            rejected.append((qid, "forbidden:" + word))
            continue
        if qid in recent_set:
            rejected.append((qid, "recent"))
            continue
        return QPick(text=text, id=qid, source="llm", rejected=rejected, fill=fill)

    by_id = {q["id"]: q for q in bank()}
    order = _bank_doc()["fallback_order"]
    # Every fallback recently used (only possible with a short history of
    # fallbacks): B1 anyway -- a question is required.
    chosen = next((qid for qid in order if qid not in recent_set), order[0])
    return QPick(
        text=_fill_daypart(by_id[chosen]["text"], daypart_name),
        id=chosen,
        source="fallback_" + chosen,
        rejected=rejected,
    )

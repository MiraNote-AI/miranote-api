"""The ending questions (copy refine handoff 4): call B writes three free
questions, each with a type from prompts/question_types.txt; the program
checks them (length, forbidden words, pronouns, type, people), call C checks
grounding and safety, and up to three survivors go to the phone in the
model's order. The phone picks one that does not repeat a neighbouring
night's type. None survives -> B1, B2, B3 from prompts/question_bank.txt.

A repeat theme (mode "repeat") has its own type table (V1's types and five
more) and fallbacks, and also rejects its word list and counts (decision
2026-11-20 H3).

The old bank menus (A skeletons, {thing} / {moment} fills, action and sight
verb lists) are no longer used; the bank file keeps only the fallbacks'
role, and its other questions stay there as reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import texts
from body import char_len, count_hit, marker_leak, repeat_words, strip_echo
from check_a import AResult

MAX_CHARS = 22
TOP_N = 3
TYPE_FILES = {"life_fragment": "question_types", "repeat": "repeat_question_types"}
BANK_FILES = {"life_fragment": "question_bank", "repeat": "repeat_question_bank"}


@dataclass
class Question:
    text: str
    type: str
    source: str  # llm | fallback_B1 | fallback_B2 | fallback_B3
    why: str = ""
    rank: Optional[int] = None

    def payload(self) -> dict:
        return {"text": self.text, "type": self.type, "source": self.source}


def type_table(mode: str = "life_fragment") -> List[dict]:
    return list(texts.json(TYPE_FILES[mode])["types"])


def type_ids(mode: str = "life_fragment") -> List[str]:
    return [t["id"] for t in type_table(mode)]


def prompt_types(mode: str = "life_fragment") -> str:
    """The type table as lines for call B's prompt."""
    names = texts.json("names")
    return "\n".join(names["type_line"].format(**dict({"note": ""}, **t))
                     for t in type_table(mode))


def prompt_recent(recent: List[str], mode: str = "life_fragment") -> str:
    """The neighbouring nights' types by their Chinese names, or "none"."""
    names = texts.json("names")
    by_id = {t["id"]: t["name"] for t in type_table(mode)}
    shown = []
    for r in recent or []:
        if r in by_id and by_id[r] not in shown:
            shown.append(by_id[r])
    return names["sep"].join(shown) or names["none"]


def from_b(b: Optional[dict]) -> List[Question]:
    """Call B's ranked questions, in its order."""
    ranked = ((b or {}).get("question") or {}).get("ranked") or []
    out: List[Question] = []
    for i, r in enumerate(ranked):
        if not isinstance(r, dict) or not isinstance(r.get("text"), str):
            continue
        out.append(Question(
            text=strip_echo(r["text"]),
            type=str(r.get("type") or "").strip(),
            source="llm",
            why=r.get("why") if isinstance(r.get("why"), str) else "",
            rank=i,
        ))
    return out


def _has_people(a: AResult) -> bool:
    return any(a.photos[pid].people_count != "0" for pid in a.selected)


def reject_reason(q: Question, a: AResult, mode: str = "life_fragment") -> Optional[str]:
    """Why the program rejects a model question, or None (handoff 2.3)."""
    names = texts.json("names")
    if not q.text:
        return "empty"
    if char_len(q.text) > MAX_CHARS:
        return "too_long"
    for w in names["forbidden_question_words"]:
        if w in q.text:
            return "forbidden:" + w
    if mode == "repeat":
        hit = next((w for w in repeat_words() if w in q.text), None) or count_hit(q.text)
        if hit:
            return "forbidden:" + hit
    rest = q.text
    for ok in names["question_pronoun_allowed"]:
        rest = rest.replace(ok, "")
    for w in names["question_pronouns"]:
        if w in rest:
            return "pronoun:" + w
    leak = marker_leak(q.text, a)
    if leak:
        return leak
    if q.type not in type_ids(mode):
        return "unknown_type:" + q.type
    if q.type == "people" and not _has_people(a):
        return "people_without_people"
    return None


def hard_check(qs: List[Question], a: AResult,
               mode: str = "life_fragment") -> Tuple[List[Question], List[Tuple[str, str]]]:
    """The model's first TOP_N questions that pass the program, and the
    rejected ones with reasons. A repeated text is dropped."""
    passed: List[Question] = []
    rejected: List[Tuple[str, str]] = []
    for q in qs[:TOP_N]:
        reason = reject_reason(q, a, mode)
        if reason is None and any(p.text == q.text for p in passed):
            reason = "duplicate"
        if reason:
            rejected.append((q.text, reason))
        else:
            passed.append(q)
    return passed, rejected


def too_long(qs: List[Question], a: AResult, mode: str = "life_fragment") -> List[Question]:
    """The model's first TOP_N questions the program cuts only for length:
    the ones the shorten round is asked about (decision 2026-11-05 evening,
    question item 5)."""
    return [q for q in qs[:TOP_N] if reject_reason(q, a, mode) == "too_long"]


def with_shortened(qs: List[Question], long_qs: List[Question],
                   replies: List[Question]) -> List[Question]:
    """The model's first TOP_N questions in its order, each too-long one
    replaced by its shortened version (same rank and type). A reply with a
    different count cannot be matched up and changes nothing."""
    if len(replies) != len(long_qs):
        return list(qs[:TOP_N])
    swap = {id(old): Question(text=new.text, type=old.type, source="llm",
                              why=old.why, rank=old.rank)
            for old, new in zip(long_qs, replies)}
    return [swap.get(id(q), q) for q in qs[:TOP_N]]


def fallbacks(fill: str, mode: str = "life_fragment") -> List[Question]:
    """B1, B2, B3, each with its type. `fill` is the daypart name, or for a
    repeat theme its noun (names.txt repeat_noun_fallback when it has none)."""
    names = texts.json("names")
    doc = texts.json(BANK_FILES[mode])
    if mode == "repeat":
        token, fill = names["slot_tokens"]["noun"], fill or names["repeat_noun_fallback"]
    else:
        token = names["slot_tokens"]["daypart"]
    by_id = {q["id"]: q for q in doc["questions"]}
    return [Question(text=by_id[qid]["text"].replace(token, fill),
                     type=doc["fallback_types"][qid], source="fallback_" + qid)
            for qid in doc["fallback_order"]]


def final(survivors: List[Question], fill: str, mode: str = "life_fragment") -> List[Question]:
    """What the phone gets: the survivors in the model's order, or the three
    fallbacks when none survived (`fill` as in fallbacks)."""
    return list(survivors[:TOP_N]) or fallbacks(fill, mode)

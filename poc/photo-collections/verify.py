"""Call C (copy refine handoff 2.4): the model looks at the selected photos
and only answers, per detail sentence, "is everything it mentions in the
photos", "does it fit the cover" and "does it read" (only plainly broken
sentences fail, decision 2026-11-05 evening item 7), and per question, "is
what it mentions in the photos" and "is it safe". It does not judge style.

Sentences are S1.., questions Q1.., in the order given. An item the reply
leaves out is dropped; an unsure (non-boolean) answer passes "grounded" and
"fluent" and fails "safe" (decision 2026-11-04 item 3). A reply that answers nothing
fails the check (the pipeline retries once; two failures mean templates
and fallback questions only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import check_a
import texts


@dataclass
class Verdict:
    grounded: bool
    reason: str = ""
    mentions_cover: bool = True  # sentences only
    fluent: bool = True  # sentences only
    safe: bool = True  # questions only


@dataclass
class CResult:
    sentences: Dict[str, Verdict] = field(default_factory=dict)
    questions: Dict[str, Verdict] = field(default_factory=dict)


def sentence_id(i: int) -> str:
    return "S{}".format(i + 1)


def question_id(i: int) -> str:
    return "Q{}".format(i + 1)


def user_text(sentences: List[str], questions: List[str]) -> Tuple[str, str]:
    """(the per-photo line template, the task after the photos)."""
    names = texts.json("names")
    raw = texts.raw("verify_c_user")
    _, _, rest = raw.partition("=====PHOTO=====\n")
    photo_tpl, _, task = rest.partition("=====TASK=====\n")
    none = names["none"]
    s_lines = "\n".join(names["c_sentence"].format(id=sentence_id(i), text=t)
                        for i, t in enumerate(sentences)) or none
    q_lines = "\n".join(names["c_question"].format(id=question_id(i), text=t)
                        for i, t in enumerate(questions)) or none
    return photo_tpl.strip("\n"), task.format_map({"sentences": s_lines, "questions": q_lines})


def _bool(entry: dict, key: str, unsure: bool) -> bool:
    """A verdict as a boolean; "true" / "false" strings are read as such and
    anything else (null, "unsure", missing) is the model being unsure, which
    gives `unsure` (decision 2026-11-04 item 3)."""
    value = entry.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return unsure


def _entries(raw: dict, key: str, prefix: str) -> Dict[str, dict]:
    """The items under `key` by id. Ids are read loosely: "S1", "s1" and 1
    all mean S1."""
    items = raw.get(key)
    if items is None:
        items = []
    if not isinstance(items, list):
        raise check_a.CheckFailed(key + " is not a list")
    out: Dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        if isinstance(raw_id, bool):
            continue
        if isinstance(raw_id, int):
            out[prefix + str(raw_id)] = item
        elif isinstance(raw_id, str) and raw_id.strip():
            rid = raw_id.strip().upper()
            out[rid if rid.startswith(prefix) else prefix + rid] = item
    return out


def _reason(e: dict) -> str:
    return e.get("reason") if isinstance(e.get("reason"), str) else ""


def parse(text: str, n_sentences: int, n_questions: int) -> CResult:
    """Every asked sentence and question gets a verdict. An item the reply
    leaves out is unchecked, so dropped; a reply that answers none of the
    asked items fails the check (the pipeline retries once)."""
    raw = check_a.parse_json(text)
    sents, ques = _entries(raw, "sentences", "S"), _entries(raw, "questions", "Q")
    out = CResult()
    answered = 0
    for i in range(n_sentences):
        sid = sentence_id(i)
        e = sents.get(sid)
        if e is None:
            out.sentences[sid] = Verdict(grounded=False, reason="no verdict", mentions_cover=False)
            continue
        answered += 1
        out.sentences[sid] = Verdict(
            grounded=_bool(e, "grounded", unsure=True),
            mentions_cover=_bool(e, "mentions_cover", unsure=False),
            fluent=_bool(e, "fluent", unsure=True),
            reason=_reason(e),
        )
    for i in range(n_questions):
        qid = question_id(i)
        e = ques.get(qid)
        if e is None:
            out.questions[qid] = Verdict(grounded=False, reason="no verdict", safe=False)
            continue
        answered += 1
        out.questions[qid] = Verdict(
            grounded=_bool(e, "grounded", unsure=True),
            safe=_bool(e, "safe", unsure=False),
            reason=_reason(e),
        )
    if n_sentences + n_questions and not answered:
        raise check_a.CheckFailed("no verdict for any item")
    return out

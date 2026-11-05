"""Call C (copy refine handoff 2.4): the model looks at the selected photos
and only answers, per detail sentence, "is everything it mentions in the
photos" and "does it mention the cover", and per question, "is what it
mentions in the photos" and "is it safe". It does not judge style.

Sentences are S1.., questions Q1.., in the order given. A reply that leaves
one out, or answers with a non-boolean, fails the check (the pipeline
retries once; two failures mean templates and fallback questions only).
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


def _bool(entry: dict, key: str, where: str) -> bool:
    value = entry.get(key)
    if not isinstance(value, bool):
        raise check_a.CheckFailed("{} {} is not a boolean".format(where, key))
    return value


def _entries(raw: dict, key: str) -> Dict[str, dict]:
    items = raw.get(key)
    if items is None:
        items = []
    if not isinstance(items, list):
        raise check_a.CheckFailed(key + " is not a list")
    out: Dict[str, dict] = {}
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            out[item["id"].strip()] = item
    return out


def parse(text: str, n_sentences: int, n_questions: int) -> CResult:
    raw = check_a.parse_json(text)
    sents, ques = _entries(raw, "sentences"), _entries(raw, "questions")
    out = CResult()
    for i in range(n_sentences):
        sid = sentence_id(i)
        e = sents.get(sid)
        if e is None:
            raise check_a.CheckFailed("no verdict for " + sid)
        out.sentences[sid] = Verdict(
            grounded=_bool(e, "grounded", sid),
            mentions_cover=_bool(e, "mentions_cover", sid),
            reason=e.get("reason") if isinstance(e.get("reason"), str) else "",
        )
    for i in range(n_questions):
        qid = question_id(i)
        e = ques.get(qid)
        if e is None:
            raise check_a.CheckFailed("no verdict for " + qid)
        out.questions[qid] = Verdict(
            grounded=_bool(e, "grounded", qid),
            safe=_bool(e, "safe", qid),
            reason=e.get("reason") if isinstance(e.get("reason"), str) else "",
        )
    return out

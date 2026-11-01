"""Parse and check call A's output (AI doc 1.3 / 1.4).

validate() either raises CheckFailed (a structural failure: the pipeline
retries call A once, then rejects with check_failed) or returns an AResult
whose verdict says whether the group can be pushed:

    ok                 push it
    sensitive          a selected photo carries a sensitive flag
    skip_screenshots   the model said skip, or every selected photo is flagged
    too_few_selected   fewer than 3 selected after dedupe
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

SUBJECT_TYPES = ("food", "drink", "person", "animal", "object", "scenery")
FLAGS = (
    "screenshot", "document", "unrecognizable",
    "hospital", "funeral", "accident", "injury", "nudity",
)
SENSITIVE_FLAGS = frozenset({"hospital", "funeral", "accident", "injury", "nudity"})
SCENES = (
    "home_indoor", "restaurant_cafe", "shop_mall", "street_city", "park_nature",
    "waterside", "mountain", "transit", "venue_event", "work_study", "other",
)
PEOPLE_COUNTS = ("0", "1", "2-4", "5+")
MIN_SELECTED = 3
MAX_SELECTED = 8

# Chinese curly quotes, written as escapes so this file stays ASCII.
_LQ = "\u201c"
_RQ = "\u201d"


class CheckFailed(Exception):
    """Call A's output is unusable as structured data."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Subject:
    name: str
    type: str


@dataclass
class PhotoFacts:
    id: str
    subjects: List[Subject] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    people_count: str = "0"
    face_prominent: bool = False
    scene: str = "other"
    flags: List[str] = field(default_factory=list)
    keep: bool = True
    drop_reason: str = ""


@dataclass
class AResult:
    photos: Dict[str, PhotoFacts]
    selected: List[str]
    peak: str
    verdict: str
    order: List[str] = field(default_factory=list)  # input ids, time order
    peak_replaced_from: Optional[str] = None
    skip_reason: str = ""


def parse_json(text: Optional[str]) -> dict:
    """The JSON object in a model reply: bare, fenced in ```json, or wrapped
    in prose. Repairs the Chinese-quote slip ported from the prototype's
    photothemes/vlm.py parse_json."""
    if not text or not text.strip():
        raise CheckFailed("empty reply")
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    else:
        s, e = t.find("{"), t.rfind("}")
        if s < 0 or e < 0 or e < s:
            raise CheckFailed("no JSON object found")
        t = t[s:e + 1]
    try:
        value = json.loads(t)
    except json.JSONDecodeError as ex:
        first = "JSON decode error: {}".format(ex)
        # The model sometimes opens/closes a string value with Chinese quotes.
        fixed = re.sub(r"(:\s*)" + _LQ, r'\1"', t)
        fixed = re.sub(_RQ + r'(\s*[,}\]]\s*(?:\n|$|"|\}))', r'"\1', fixed)
        try:
            value = json.loads(fixed)
        except json.JSONDecodeError:
            raise CheckFailed(first) from None
    if not isinstance(value, dict):
        raise CheckFailed("top level is not an object")
    return value


def _strings(value) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _clean_subjects(value) -> List[Subject]:
    """Rule 5: name 2-4 chars after removing whitespace, type in the enum.
    A bad entry is dropped, never truncated."""
    out: List[Subject] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        kind = item.get("type")
        if not isinstance(name, str) or not isinstance(kind, str):
            continue
        name = re.sub(r"\s+", "", name)
        if not 2 <= len(name) <= 4 or kind not in SUBJECT_TYPES:
            continue
        out.append(Subject(name=name, type=kind))
    return out


def _people(value) -> tuple:
    count, face = "0", False
    if isinstance(value, dict):
        raw = value.get("count")
        face = value.get("face_prominent") is True
    else:
        raw = value
    if isinstance(raw, str) and raw.strip() in PEOPLE_COUNTS:
        count = raw.strip()
    else:
        number = None
        if isinstance(raw, bool):
            number = None
        elif isinstance(raw, (int, float)):
            number = int(raw)
        elif isinstance(raw, str) and raw.strip().isdigit():
            number = int(raw.strip())
        if number is not None:
            if number <= 0:
                count = "0"
            elif number == 1:
                count = "1"
            elif number <= 4:
                count = "2-4"
            else:
                count = "5+"
    return count, face


def _photo(item) -> PhotoFacts:
    if not isinstance(item, dict) or not isinstance(item.get("id"), str):
        raise CheckFailed("a photos entry is not an object with an id")
    count, face = _people(item.get("people"))
    scene = item.get("scene")
    return PhotoFacts(
        id=item["id"].strip(),
        subjects=_clean_subjects(item.get("subjects")),
        details=_strings(item.get("details")),
        text=_strings(item.get("text")),
        people_count=count,
        face_prominent=face,
        scene=scene if scene in SCENES else "other",
        flags=[f for f in _strings(item.get("flags")) if f in FLAGS],
        keep=item.get("keep") is not False,
        drop_reason=item.get("drop_reason") if isinstance(item.get("drop_reason"), str) else "",
    )


def validate(raw, input_ids: List[str]) -> AResult:
    """Check call A's parsed output against the ids that were sent."""
    if not isinstance(raw, dict):
        raise CheckFailed("output is not an object")

    # Rule 1: photos covers every input id, nothing extra, nothing twice.
    items = raw.get("photos")
    if not isinstance(items, list):
        raise CheckFailed("photos missing or not a list")
    photos: Dict[str, PhotoFacts] = {}
    for item in items:
        p = _photo(item)
        if p.id in photos:
            raise CheckFailed("photo {} described twice".format(p.id))
        photos[p.id] = p
    missing = [pid for pid in input_ids if pid not in photos]
    extra = [pid for pid in photos if pid not in input_ids]
    if missing:
        raise CheckFailed("photos missing ids {}".format(missing))
    if extra:
        raise CheckFailed("photos has unknown ids {}".format(extra))

    # Rule 2: selected ids are input ids, 3-8 after dedupe, each keep=true.
    sel_raw = raw.get("selected_ids")
    if not isinstance(sel_raw, list) or not all(isinstance(s, str) for s in sel_raw):
        raise CheckFailed("selected_ids missing or not a list of strings")
    wanted = set(s.strip() for s in sel_raw)
    unknown = sorted(wanted - set(input_ids))
    if unknown:
        raise CheckFailed("selected_ids has unknown ids {}".format(unknown))
    selected = [pid for pid in input_ids if pid in wanted]  # dedupe + time order
    if len(selected) > MAX_SELECTED:
        raise CheckFailed("selected {} photos, more than {}".format(len(selected), MAX_SELECTED))
    dropped = [pid for pid in selected if not photos[pid].keep]
    if dropped:
        raise CheckFailed("selected photos marked keep=false: {}".format(dropped))

    # Rule 3: peak must be selected.
    peak = raw.get("peak_id")
    peak = peak.strip() if isinstance(peak, str) else ""
    if peak not in selected:
        raise CheckFailed("peak_id {!r} is not in selected_ids".format(peak))

    skip_reason = raw.get("skip_reason") if isinstance(raw.get("skip_reason"), str) else ""
    result = AResult(
        photos=photos, selected=selected, peak=peak, verdict="ok",
        order=list(input_ids), skip_reason=skip_reason,
    )

    # Rule 4: any sensitive flag on a selected photo, or skip=true -> not pushed.
    if any(SENSITIVE_FLAGS.intersection(photos[pid].flags) for pid in selected):
        result.verdict = "sensitive"
        return result
    if raw.get("skip") is True:
        result.verdict = "skip_screenshots"
        return result

    # Rule 3 continued: a flagged peak is replaced by the unflagged selected
    # photo with the most subjects (tie -> the latest); none left -> not pushed.
    if photos[peak].flags:
        clean = [pid for pid in selected if not photos[pid].flags]
        if not clean:
            result.verdict = "skip_screenshots"
            return result
        best = max(
            clean,
            key=lambda pid: (len(photos[pid].subjects), input_ids.index(pid)),
        )
        result.peak_replaced_from = peak
        result.peak = best

    # Rule 6: fewer than 3 selected -> not pushed.
    if len(selected) < MIN_SELECTED:
        result.verdict = "too_few_selected"
    return result

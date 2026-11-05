"""The detail sentence: templates T1 / T2 / F0, the program's hard checks and
the pick (copy refine handoff 2.3, 2.5, 2.6). Every Chinese word list and
template comes from prompts/.

The program checks only what it can check without the photos: length, the
lexicon, photo ids and "lianpai" in the text. Whether a model sentence is
true to the photos is call C's job (verify.py); the pipeline drops what C
rejects before pick() sees it.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import jieba
import jieba.posseg as pseg

import texts
from check_a import AResult

jieba.setLogLevel(logging.WARNING)
_jieba_lock = threading.Lock()

MAX_CHARS = 22
NOUN_FLAGS = frozenset({"n", "nr", "ns", "nz"})
SOURCES = ("llm", "template_T1", "template_T2", "fallback_F0")


@dataclass
class Candidate:
    text: str
    source: str  # llm | template_T1 | template_T2
    photo_ids: List[str] = field(default_factory=list)
    rank: Optional[int] = None  # the model's order, 0-based; None for templates


@dataclass
class BodyPick:
    text: str
    source: str
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    photo_ids: List[str] = field(default_factory=list)
    score: Optional[float] = None


@dataclass
class When:
    """The cover's weekday and daypart names, for T1 and F0."""
    weekday: str
    daypart: str


@dataclass
class _Lexicon:
    category: str
    words: List[str]
    allowed: List[str]
    skip_t2: bool
    subjects_shield: bool


def char_len(s: str) -> int:
    """Characters after removing whitespace; every other character counts 1."""
    return len(re.sub(r"\s+", "", s))


def _names() -> dict:
    return texts.json("names")


def _lexicon() -> List[_Lexicon]:
    out = []
    for line in texts.lines("lexicon_push_body"):
        parts = line.split("|")
        if len(parts) < 2:
            continue
        exceptions = parts[2].split() if len(parts) > 2 else []
        out.append(_Lexicon(
            category=parts[0].strip(),
            words=parts[1].split(),
            allowed=[e for e in exceptions if not e.startswith("@")],
            skip_t2="@T2" in exceptions,
            subjects_shield="@nosubjects" not in exceptions,
        ))
    return out


def clean(text: str) -> str:
    """The text as checked and shown: trimmed, trailing full stops removed."""
    t = text.strip()
    end = _names()["body_strip_end"]
    while t.endswith(end):
        t = t[: -len(end)].rstrip()
    return t


# ---------- facts of the selected photos ----------

def _subject_names(a: AResult) -> List[str]:
    seen: List[str] = []
    for pid in a.selected:
        for s in a.photos[pid].subjects:
            if s.name not in seen:
                seen.append(s.name)
    return seen


def _texts(a: AResult) -> List[str]:
    return [t for pid in a.selected for t in a.photos[pid].text]


def program_counts(a: AResult) -> Dict[str, int]:
    """Subject name -> number of selected photos it appears in."""
    counts: Dict[str, int] = {}
    for pid in a.selected:
        for name in {s.name for s in a.photos[pid].subjects}:
            counts[name] = counts.get(name, 0) + 1
    return counts


# ---------- templates ----------

def templates(a: AResult, meta: dict, when: When) -> List[Candidate]:
    """T1 (only when the phone covered the whole group) and T2."""
    names = _names()
    out: List[Candidate] = []
    represents = {}
    for p in meta.get("photos") or []:
        if isinstance(p, dict) and "id" in p:
            try:
                represents[p["id"]] = max(1, int(p.get("represents") or 1))
            except (TypeError, ValueError):
                represents[p["id"]] = 1
    if meta.get("covers_all") is True:
        counts = program_counts(a)
        for name in _subject_names(a):
            if counts.get(name, 0) < 2:
                continue
            # n counts every sent photo showing X (each with its merged
            # duplicates), not only the selected ones: "n photos of X" must
            # be true of the whole group.
            with_x = [pid for pid in a.order
                      if any(s.name == name for s in a.photos[pid].subjects)]
            n = sum(represents.get(pid, 1) for pid in with_x)
            sel_with_x = [pid for pid in a.selected if pid in with_x]
            out.append(Candidate(
                text=names["t1"].format(x=name, n=n, weekday=when.weekday,
                                        daypart=when.daypart),
                source="template_T1",
                photo_ids=sel_with_x,
            ))
    if a.selected:
        first, last = a.selected[0], a.selected[-1]
        fs, ls = a.photos[first].subjects, a.photos[last].subjects
        if fs and ls and fs[0].name != ls[0].name:
            out.append(Candidate(
                text=names["t2"].format(first=fs[0].name, last=ls[0].name),
                source="template_T2",
                photo_ids=[first, last],
            ))
    return out


def from_b(b: Optional[dict]) -> List[Candidate]:
    """Model candidates from call B's parsed output (empty if cannot_write)."""
    if not isinstance(b, dict) or b.get("cannot_write") is True:
        return []
    out: List[Candidate] = []
    for i, c in enumerate(b.get("candidates") or []):
        if not isinstance(c, dict) or not isinstance(c.get("text"), str):
            continue
        out.append(Candidate(
            text=c["text"],
            source="llm",
            photo_ids=[x for x in c.get("photo_ids") or [] if isinstance(x, str)],
            rank=i,
        ))
    return out


# ---------- hard checks ----------

def _spans(text: str, needle: str) -> Iterable[Tuple[int, int]]:
    if not needle:
        return
    start = text.find(needle)
    while start >= 0:
        yield start, start + len(needle)
        start = text.find(needle, start + 1)


def _lexicon_hit(t: str, c: Candidate, a: AResult) -> Optional[str]:
    """The first lexicon word in t that is not inside a photo's text, a
    subject name (unless its category says @nosubjects), or its category's
    allowed words."""
    for cat in _lexicon():
        if cat.skip_t2 and c.source == "template_T2":
            continue
        shields = (_subject_names(a) if cat.subjects_shield else []) + _texts(a)
        guards = [g for g in shields + cat.allowed if g]
        for word in cat.words:
            for s, e in _spans(t, word):
                covered = any(
                    gs <= s and e <= ge
                    for g in guards if word in g
                    for gs, ge in _spans(t, g)
                )
                if not covered:
                    return word
    return None


def reject_reason(c: Candidate, a: AResult) -> Optional[str]:
    """Why the program rejects this candidate, or None. Order: empty ->
    length -> lexicon -> (model-written only) photo id -> lianpai."""
    t = clean(c.text)
    if not t:
        return "empty"
    if char_len(t) > MAX_CHARS:
        return "too_long"
    hit = _lexicon_hit(t, c, a)
    if hit:
        return "lexicon:" + hit
    if c.source == "llm":
        leak = marker_leak(t, a)
        if leak:
            return leak
        if _names()["lianpai"] in t:
            return "lianpai"
    return None


def marker_leak(t: str, a: AResult) -> Optional[str]:
    """A sanity check on broken output, for sentences and questions alike:
    the photo ids and the word for "cover" are markers in our prompt, never
    copy a person should see (a question naming P1 and P2 was seen on real
    photos, 2026-11-05)."""
    for pid in a.order:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(pid) + r"(?![A-Za-z0-9])", t):
            return "photo_id:" + pid
    for word in _names()["marker_words"]:
        if word in t:
            return "marker:" + word
    return None


# ---------- scoring ----------

def _nouns(text: str) -> List[str]:
    with _jieba_lock:
        return [w.word for w in pseg.cut(text) if w.flag in NOUN_FLAGS]


def _score(c: Candidate, t: str, a: AResult) -> float:
    """Handoff 2.5: the model's own order first, +1 for tying several
    photos together, -2 for only an everyday thing without light or color."""
    score = 0.0
    if c.source == "llm" and c.rank is not None:
        score += {0: 2.0, 1: 1.0}.get(c.rank, 0.0)
    if len(set(c.photo_ids)) >= 2:
        score += 1
    mentioned = [n for n in _subject_names(a) if n in t] or _nouns(t)
    commons = set(texts.words("common_objects"))
    lit = any(w in t for w in texts.words("feature_words"))
    if mentioned and all(m in commons for m in mentioned) and not lit:
        score -= 2
    return score


def cites_peak(p: "BodyPick", a: AResult) -> bool:
    """The picked sentence is a model sentence about the cover photo."""
    return p.source == "llm" and a.peak in p.photo_ids


def f0_text(raw_count: int, when: When) -> str:
    return _names()["f0"].format(n=raw_count, weekday=when.weekday, daypart=when.daypart)


def pick(ai: List[Candidate], templates: List[Candidate], a: AResult, *,
         raw_count: int, when: When) -> BodyPick:
    """Check every candidate, score the survivors, return the best one; F0
    when none survives. Model candidates are expected to have passed call C
    already; the hard checks run again here for templates and as a guard."""
    rejected: List[Tuple[str, str]] = []
    alive: List[Tuple[float, int, int, int, Candidate, str]] = []
    for order, c in enumerate(list(ai) + list(templates)):
        reason = reject_reason(c, a)
        if reason:
            rejected.append((c.text, reason))
            continue
        t = clean(c.text)
        s = _score(c, t, a)
        # max by: score, AI before template, shorter, earlier
        alive.append((s, 1 if c.source == "llm" else 0, -char_len(t), -order, c, t))
    if not alive:
        return BodyPick(text=f0_text(raw_count, when), source="fallback_F0", rejected=rejected)
    # Peak first (user decision 2026-10-31): when a surviving model sentence
    # cites the cover photo, the pick is made among those only; otherwise
    # among every survivor (model sentences and templates).
    on_peak = [x for x in alive if x[4].source == "llm" and a.peak in x[4].photo_ids]
    best = max(on_peak or alive, key=lambda x: x[:4])
    return BodyPick(
        text=best[5], source=best[4].source, rejected=rejected,
        photo_ids=list(best[4].photo_ids), score=best[0],
    )

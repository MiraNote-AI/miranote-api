"""The detail sentence: templates T1 / T2 / F0, the program's hard checks and
the pick (copy refine handoff 2.3, 2.5, 2.6). Every Chinese word list and
template comes from prompts/.

The program checks only what it can check without the photos: length, the
lexicon, photo ids and "lianpai" in the text. A repeat theme (mode "repeat")
adds its own word list and count patterns on top, and has its own
templates (decision 2026-11-20 H2, H4). Whether a model sentence is
true to the photos is call C's job (verify.py); the pipeline drops what C
rejects before pick() sees it.
"""

from __future__ import annotations

import difflib
import functools
import logging
import re
import threading
import unicodedata
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
# The word lists each mode checks; a repeat theme checks V1's list and its own.
LEXICONS = {"life_fragment": ("lexicon_push_body",),
            "repeat": ("lexicon_push_body", "repeat_lexicon")}


@dataclass
class Candidate:
    text: str
    source: str  # llm | template_T1 | template_T2
    photo_ids: List[str] = field(default_factory=list)
    rank: Optional[int] = None  # the model's order, 0-based; None for templates
    style: str = ""  # the model's tone label (decision 2026-11-05 evening item 3)


@dataclass
class BodyPick:
    text: str
    source: str
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    photo_ids: List[str] = field(default_factory=list)
    score: Optional[float] = None
    style: str = ""


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


def _lexicon(mode: str = "life_fragment") -> List[_Lexicon]:
    out = []
    for name in LEXICONS[mode]:
        for line in texts.lines(name):
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


def repeat_words() -> List[str]:
    """Every word of the repeat-only list (prompts/repeat_lexicon.txt)."""
    return [w for line in texts.lines("repeat_lexicon") if "|" in line
            for w in line.split("|")[1].split()]


@functools.lru_cache(maxsize=4)
def _compiled(patterns: Tuple[str, ...]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def count_hit(text: str) -> Optional[str]:
    """Repeat themes: the first count or frequency in text ("taken 3 times",
    "every day": names.txt repeat_patterns), or None. Each pattern set is
    compiled once."""
    folded = unicodedata.normalize("NFKC", text)
    for rx in _compiled(tuple(_names()["repeat_patterns"])):
        m = rx.search(folded)
        if m:
            return m.group(0)
    return None


def strip_echo(text: str) -> str:
    """The text without prompt annotations the model copied back: a list
    marker in front, "(18 chars)" / "| photo_ids: P1" / "(zoom)" at the end
    (real shorten replies, full check 2026-11-05)."""
    names = _names()
    t = re.sub(names["echo_head"], "", text.strip())
    changed = True
    while changed:
        changed = False
        for pattern in names["echo_tail"]:
            cut = re.sub(pattern, "", t)
            if cut != t:
                t, changed = cut, True
    return t.strip()


def clean(text: str) -> str:
    """The text as checked and shown: trimmed, prompt annotations and
    trailing full stops removed."""
    t = strip_echo(text)
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


def templates_repeat(theme: dict, photo_ids: List[str]) -> List[Candidate]:
    """A repeat theme's templates in fallback order (decision 2026-11-20 H4):
    T1 when the theme has a daypart and a noun, T2 when it has a noun, F0
    always and last. n is the number of photos on the page."""
    names = _names()
    noun = (theme.get("noun") or "").strip()
    daypart = (theme.get("where") or {}).get("daypart")
    # "n photos, n different days" holds: the phone sends one photo per day.
    n = len(photo_ids)
    out: List[Candidate] = []
    if noun and daypart:
        out.append(Candidate(text=names["repeat_t1"].format(daypart=names["dayparts"][daypart],
                                                            noun=noun),
                             source="template_T1", photo_ids=list(photo_ids)))
    if noun:
        out.append(Candidate(text=names["repeat_t2"].format(n=n, noun=noun),
                             source="template_T2", photo_ids=list(photo_ids)))
    out.append(Candidate(text=names["repeat_f0"].format(n=n), source="fallback_F0",
                         photo_ids=list(photo_ids)))
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
            style=c.get("style") if isinstance(c.get("style"), str) else "",
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


def _lexicon_hit(t: str, c: Candidate, a: AResult, mode: str = "life_fragment") -> Optional[str]:
    """The first lexicon word in t that is not inside a photo's text, a
    subject name (unless its category says @nosubjects), or its category's
    allowed words."""
    for cat in _lexicon(mode):
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


def _plain(text: str) -> str:
    """NFKC, then no whitespace and no punctuation (Unicode category P*)."""
    folded = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in folded
                   if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def example_copy(text: str) -> Optional[str]:
    """The prompt example (prompts/repeat_examples.txt) this text copies:
    equal, or at least 0.8 similar, after _plain. None when it is not a copy.
    Repeat themes only; the model copied examples verbatim on a device run
    (2026-11-25)."""
    a = _plain(text)
    if not a:
        return None
    for example in texts.json("repeat_examples"):
        b = _plain(example)
        if a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8:
            return example
    return None


def reject_reason(c: Candidate, a: AResult, mode: str = "life_fragment") -> Optional[str]:
    """Why the program rejects this candidate, or None. Order: empty ->
    length -> lexicon (a repeat theme: then its counts) -> (model-written
    only) photo id -> lianpai."""
    t = clean(c.text)
    if not t:
        return "empty"
    if char_len(t) > MAX_CHARS:
        return "too_long"
    if mode == "repeat" and c.source == "llm":
        copied = example_copy(t)
        if copied:
            return "example_copy:" + copied[:40]
    hit = _lexicon_hit(t, c, a, mode)
    if not hit and mode == "repeat":
        hit = count_hit(t)
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
    folded = unicodedata.normalize("NFKC", t)
    for pid in a.order:
        letters, digits = pid[:1], pid[1:]
        pattern = (r"(?<![A-Za-z0-9])" + re.escape(letters) + r"\s*" + re.escape(digits)
                   + r"(?![0-9])")
        if re.search(pattern, folded, re.IGNORECASE):
            return "photo_id:" + pid
    names = _names()
    for word in names["marker_words"]:
        if word in folded:
            return "marker:" + word
    for pattern in names["marker_patterns"]:
        if re.search(pattern, folded):
            return "marker:" + pattern
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
         raw_count: int, when: When, mode: str = "life_fragment",
         f0: Optional[str] = None) -> BodyPick:
    """Check every candidate, score the survivors, return the best one; F0
    (the text `f0` when given) when none survives. Model candidates are
    expected to have passed call C already; the hard checks run again here
    for templates and as a guard."""
    rejected: List[Tuple[str, str]] = []
    alive: List[Tuple[float, int, int, int, Candidate, str]] = []
    for order, c in enumerate(list(ai) + list(templates)):
        reason = reject_reason(c, a, mode)
        if reason:
            rejected.append((c.text, reason))
            continue
        t = clean(c.text)
        s = _score(c, t, a)
        # max by: score, AI before template, shorter, earlier
        alive.append((s, 1 if c.source == "llm" else 0, -char_len(t), -order, c, t))
    if not alive:
        return BodyPick(text=f0 if f0 is not None else f0_text(raw_count, when),
                        source="fallback_F0", rejected=rejected)
    # Peak first (user decision 2026-10-31): when a surviving model sentence
    # cites the cover photo, the pick is made among those only; otherwise
    # among every survivor (model sentences and templates).
    on_peak = [x for x in alive if x[4].source == "llm" and a.peak in x[4].photo_ids]
    best = max(on_peak or alive, key=lambda x: x[:4])
    return BodyPick(
        text=best[5], source=best[4].source, rejected=rejected,
        photo_ids=list(best[4].photo_ids), score=best[0], style=best[4].style,
    )

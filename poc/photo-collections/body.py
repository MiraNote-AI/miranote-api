"""The detail sentence: templates T1 / T2 / F0 and the program pick (AI doc
2.1 and 4). Every Chinese word list and template comes from prompts/.

Order of checks for each candidate (first failure is the logged reason):
length -> lexicon -> (model-written only) uses -> nouns -> numbers -> lianpai.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import jieba
import jieba.posseg as pseg

import texts
from check_a import AResult

jieba.setLogLevel(logging.WARNING)
_jieba_lock = threading.Lock()

MAX_CHARS = 16
NOUN_FLAGS = frozenset({"n", "nr", "ns", "nz"})
SOURCES = ("llm", "template_T1", "template_T2", "fallback_F0")


@dataclass
class Candidate:
    text: str
    source: str  # llm | template_T1 | template_T2
    photo_ids: List[str] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)
    rank: Optional[int] = None  # the model's order, 0-based; None for templates


@dataclass
class BodyPick:
    text: str
    source: str
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    photo_ids: List[str] = field(default_factory=list)
    score: Optional[float] = None


@dataclass
class _Lexicon:
    category: str
    words: List[str]
    allowed: List[str]
    skip_t2: bool


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
        ))
    return out


def _clean(text: str) -> str:
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


def _details(a: AResult) -> List[str]:
    return [d for pid in a.selected for d in a.photos[pid].details]


def program_counts(a: AResult) -> Dict[str, int]:
    """Subject name -> number of selected photos it appears in."""
    counts: Dict[str, int] = {}
    for pid in a.selected:
        for name in {s.name for s in a.photos[pid].subjects}:
            counts[name] = counts.get(name, 0) + 1
    return counts


# ---------- templates ----------

def templates(a: AResult, meta: dict) -> List[Candidate]:
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
            # duplicates), not only the selected ones: "you took n" must be
            # true of the whole group.
            with_x = [pid for pid in a.order
                      if any(s.name == name for s in a.photos[pid].subjects)]
            n = sum(represents.get(pid, 1) for pid in with_x)
            sel_with_x = [pid for pid in a.selected if pid in with_x]
            out.append(Candidate(
                text=names["t1"].format(x=name, n=n),
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
            uses=[x for x in c.get("uses") or [] if isinstance(x, str)],
            rank=i,
        ))
    return out


# ---------- checks ----------

def _spans(text: str, needle: str) -> Iterable[Tuple[int, int]]:
    if not needle:
        return
    start = text.find(needle)
    while start >= 0:
        yield start, start + len(needle)
        start = text.find(needle, start + 1)


def _lexicon_hit(t: str, c: Candidate, a: AResult) -> Optional[str]:
    """The first lexicon word in t that is not inside a subject name, a
    photo's text, or its category's allowed words."""
    shields = _subject_names(a) + _texts(a)
    for cat in _lexicon():
        if cat.skip_t2 and c.source == "template_T2":
            continue
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


def _squeeze(s: str, strip: List[str]) -> str:
    for ch in strip:
        s = s.replace(ch, "")
    return re.sub(r"\s+", "", s)


def _uses_ok(t: str, uses: List[str]) -> bool:
    """Some uses fragment, without the particles de/le/zhe, is in t.

    Particles are dropped on both sides (a fragment keeping "zhe" must still
    match a sentence keeping it), and the labels the B prompt itself puts
    around facts -- a "(type)" suffix, a "text:" prefix -- are removed first:
    they are our formatting echoed back, not words the model used."""
    names = _names()
    strip = names["uses_strip"]
    labels = names["b_fact_labels"]
    target = _squeeze(t, strip)
    for u in uses:
        frag = re.sub(r"[(\uff08][^()\uff08\uff09]*[)\uff09]\s*$", "", u.strip())
        for label in labels:
            frag = re.sub("^" + re.escape(label) + r"\s*[:\uff1a]\s*", "", frag)
        frag = _squeeze(frag, strip)
        if frag and frag in target:
            return True
    return False


def _nouns(text: str) -> List[str]:
    with _jieba_lock:
        return [w.word for w in pseg.cut(text) if w.flag in NOUN_FLAGS]


def _unknown_noun(t: str, a: AResult) -> Optional[str]:
    """A noun (jieba n/nr/ns/nz) in t that the facts do not contain.

    Spans that are literally a subject, detail or text string are removed
    first; a remaining noun passes when it is a substring of any fact string
    (jieba segments the same phrase differently in different contexts, e.g.
    "guang luo" vs "guang"), or a people-count word."""
    facts = _subject_names(a) + _details(a) + _texts(a)
    rest = t
    for f in sorted(facts, key=len, reverse=True):
        if f:
            rest = rest.replace(f, "\uff0c")
    person = set(_names()["person_nouns"])
    for noun in _nouns(rest):
        if noun in person:
            continue
        if any(noun in f for f in facts):
            continue
        return noun
    return None


def _numeral_value(token: str, numerals: Dict[str, int]) -> Optional[int]:
    if token.isdigit():
        return int(token)
    total, cur = 0, 0
    for ch in token:
        v = numerals.get(ch)
        if v is None:
            return None
        if v in (10, 100):
            total += (cur or 1) * v
            cur = 0
        else:
            cur = v
    return total + cur


def _number_tokens(t: str) -> List[str]:
    numerals = "".join(_names()["numerals"].keys())
    return re.findall(r"[0-9]+|[" + re.escape(numerals) + r"]+", t)


def _bad_number(t: str, a: AResult, counts: Dict[str, int]) -> Optional[str]:
    numerals = _names()["numerals"]
    allowed_values: Set[int] = set(counts.values())
    # Subject names count as a source too (user decision 2026-10-31): the
    # subject "three people" grounds "three" in the sentence.
    sources = _details(a) + _texts(a) + _subject_names(a)
    for token in _number_tokens(t):
        if any(token in s for s in sources):
            continue
        if _numeral_value(token, numerals) in allowed_values:
            continue
        return token
    return None


def _reject_reason(c: Candidate, t: str, a: AResult, counts: Dict[str, int]) -> Optional[str]:
    if not t:
        return "empty"
    if char_len(t) > MAX_CHARS:
        return "too_long"
    hit = _lexicon_hit(t, c, a)
    if hit:
        return "lexicon:" + hit
    if c.source == "llm":
        # Not an AI doc 4 rule: a sanity check on broken output. A photo id
        # leaked into the sentence is never copy a person should see.
        for pid in a.order:
            if re.search(r"(?<![A-Za-z0-9])" + re.escape(pid) + r"(?![A-Za-z0-9])", t):
                return "photo_id:" + pid
        if not _uses_ok(t, c.uses):
            return "uses_not_in_text"
        noun = _unknown_noun(t, a)
        if noun:
            return "unknown_noun:" + noun
        number = _bad_number(t, a, counts)
        if number:
            return "number:" + number
        if _names()["lianpai"] in t:
            return "lianpai"
    return None


# ---------- scoring ----------

def _has_feature(t: str, a: AResult) -> bool:
    colors = texts.words("feature_words")
    if any(col in t for col in colors):
        return True
    if _number_tokens(t):
        return True
    return any(x and x in t for x in _texts(a))


def _score(c: Candidate, t: str, a: AResult, counts: Dict[str, int]) -> float:
    score = 0.0
    # Only the model cites photos; templates are fallback levels (AI doc 6),
    # so the peak bonus is the model's alone.
    if c.source == "llm" and a.peak in c.photo_ids:
        score += 2
    feature = _has_feature(t, a)
    if feature:
        score += 1
    mentioned = [n for n in _subject_names(a) if n in t]
    if any(counts.get(n, 0) >= 2 for n in mentioned):
        score += 1
    if not mentioned:
        mentioned = _nouns(t)
    commons = set(texts.words("common_objects"))
    if mentioned and all(m in commons for m in mentioned) and not feature:
        score -= 2
    if c.source == "llm" and c.rank is not None:
        score += {0: 1.0, 1: 0.5}.get(c.rank, 0.0)
    return score


def f0_text(raw_count: int) -> str:
    return _names()["f0"].format(n=raw_count)


def pick(ai: List[Candidate], templates: List[Candidate], a: AResult,
         counts: Dict[str, int], *, raw_count: int) -> BodyPick:
    """Check every candidate, score the survivors, return the best one; F0
    when none survives."""
    rejected: List[Tuple[str, str]] = []
    alive: List[Tuple[float, int, int, int, Candidate, str]] = []
    for order, c in enumerate(list(ai) + list(templates)):
        t = _clean(c.text)
        reason = _reject_reason(c, t, a, counts)
        if reason:
            rejected.append((c.text, reason))
            continue
        s = _score(c, t, a, counts)
        # max by: score, AI before template, shorter, earlier
        alive.append((s, 1 if c.source == "llm" else 0, -char_len(t), -order, c, t))
    if not alive:
        return BodyPick(text=f0_text(raw_count), source="fallback_F0", rejected=rejected)
    # Peak first (user decision 2026-10-31): when a surviving model sentence
    # cites the cover photo, the pick is made among those only; otherwise
    # among every survivor (model sentences and templates), as before.
    on_peak = [x for x in alive if x[4].source == "llm" and a.peak in x[4].photo_ids]
    best = max(on_peak or alive, key=lambda x: x[:4])
    return BodyPick(
        text=best[5], source=best[4].source, rejected=rejected,
        photo_ids=list(best[4].photo_ids), score=best[0],
    )

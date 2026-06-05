"""LLM reranker: pick the best matches from candidates with one-line whys.

Single LLM call; structured JSON output. Picks are indexed 1..N matching
the order of candidates passed in. Hallucination is structurally
impossible -- the LLM never sees the corpus, only the K candidates.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

SYSTEM = """You re-rank candidate quotes for emotional / semantic fit with a user's text.

Rules:
- Pick at most {max_picks} of the {n} candidates that truly fit the user's text.
- For each pick, write a one-sentence "why" in the SAME LANGUAGE as the user's text.
- If none of the candidates feels right, return an empty array.
- NEVER invent quotes. NEVER edit text or author. Only choose from the given list.

Output strictly as JSON: [{{"id": <int 1..{n}>, "why": "<one sentence>"}}].
No prose, no markdown fences, no preamble."""

USER_TEMPLATE = """Text: {user_text}

Candidates:
{candidates_block}"""


def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        author = c.get("author", "anon")
        source = c.get("source", "")
        suffix = f" -- {author}" + (f", {source}" if source else "")
        lines.append(f"{i}. [{c.get('lang', '?')}] {c['text']}{suffix}")
    return "\n".join(lines)


def rerank(
    client: Any,
    model: str,
    user_text: str,
    candidates: List[Dict[str, Any]],
    max_picks: int = 3,
) -> List[Dict[str, Any]]:
    """Return a list of {"index": <1..N>, "why": <str>}.

    Raises ValueError if the LLM emits malformed JSON, an out-of-range
    index, or a row missing the expected keys.
    """
    n = len(candidates)
    if n == 0:
        return []
    system = SYSTEM.format(max_picks=max_picks, n=n)
    user = USER_TEMPLATE.format(
        user_text=user_text,
        candidates_block=_format_candidates(candidates),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"reranker emitted invalid JSON: {raw[:200]}") from e
    if not isinstance(parsed, list):
        raise ValueError(f"reranker emitted unexpected schema (not a list): {raw[:200]}")
    out: List[Dict[str, Any]] = []
    for item in parsed[:max_picks]:
        if not isinstance(item, dict) or "id" not in item or "why" not in item:
            raise ValueError(f"reranker emitted unexpected schema: {raw[:200]}")
        idx = item["id"]
        if not isinstance(idx, int) or idx < 1 or idx > n:
            raise ValueError(f"reranker pick out of range (got {idx}, n={n}): {raw[:200]}")
        out.append({"index": idx, "why": str(item["why"])})
    return out

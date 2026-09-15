"""Prompt building and error classification for /generate.

Named for the Nano Banana fallback it used to implement: /generate once
led with Imagen and dropped to a Gemini image model on a 404. There is no
fallback any more -- ``config.MODEL_ID`` is called directly -- but the pure
helpers kept here are still what lets the call path be unit-tested without
torch or a live client.
"""
from __future__ import annotations


def is_model_unavailable(error: Exception) -> bool:
    """Vertex signals a gated/missing publisher model with a 404."""
    text = str(error)
    return "NOT_FOUND" in text or "404" in text


def is_rate_limited(error: Exception) -> bool:
    """Vertex signals an exhausted quota with a 429.

    Kept separate from is_model_unavailable because the two need opposite
    handling: a gated model is permanent for the process, a throttled one
    recovers on its own.
    """
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "429" in text


def build_prompt(prompt: str, aspect_ratio: str) -> str:
    return (
        f"Generate one image. {prompt}\n"
        f"Aspect ratio {aspect_ratio}. Return only the image, no words."
    )


# A refusal names itself in finish_reason or prompt_feedback. Matched on
# substrings of the name rather than against enum members, because the SDK
# stringifies FinishReason as "FinishReason.IMAGE_SAFETY" and because the set
# grows -- the IMAGE_* variants postdate the originals. Covers every refusing
# member of google.genai.types.FinishReason as of SDK 2026-09: SAFETY,
# RECITATION, BLOCKLIST, PROHIBITED_CONTENT, SPII and the three IMAGE_* forms.
#
# Deliberately excluded, because a retry of the same prompt can still succeed:
# NO_IMAGE (the model simply produced none), IMAGE_OTHER, OTHER, MAX_TOKENS.
_REFUSAL_MARKERS = ("SAFETY", "PROHIBITED", "BLOCK", "RECITATION", "SPII")


def _candidates(response) -> list:
    """Candidates from anything, including objects that are not responses.

    Total by design: this runs on the failure path, where the caller already
    has one problem and must not be handed a second one from the diagnostics.
    """
    return list(getattr(response, "candidates", None) or [])


def empty_reason(response) -> str:
    """Why a response carried no image, compact enough for one log line.

    /stylize and /border get this from shared.vertex_client, which raises with
    the same fields. /generate cannot raise there -- image_parts returning []
    is normal for one of several concurrent calls -- so it reports instead.
    """
    bits: list[str] = []
    candidates = _candidates(response)
    if not candidates:
        bits.append("no candidates")
    for candidate in candidates:
        finish = getattr(candidate, "finish_reason", None)
        if finish:
            bits.append(f"finish_reason={finish}")
        safety = getattr(candidate, "safety_ratings", None)
        if safety:
            bits.append(f"safety={safety}")
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                bits.append(f"text={text}")
    feedback = getattr(response, "prompt_feedback", None)
    if feedback:
        bits.append(f"prompt_feedback={feedback}")
    if not bits:
        bits.append("no image part and no diagnostics")
    return "; ".join(str(bit) for bit in bits)[:600]


def is_safety_refusal(response) -> bool:
    """Whether the model declined, as opposed to simply returning nothing.

    Retrying a refusal spends a second call from a two-per-minute bucket to be
    refused again, so the two cases must not be conflated.
    """
    signals = [
        str(getattr(candidate, "finish_reason", "") or "")
        for candidate in _candidates(response)
    ]
    signals.append(str(getattr(response, "prompt_feedback", "") or ""))
    joined = " ".join(signals).upper()
    return any(marker in joined for marker in _REFUSAL_MARKERS)


def image_parts(response) -> list[bytes]:
    """The image bytes from a generate_content response, in order."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    parts = getattr(candidates[0].content, "parts", None) or []
    out: list[bytes] = []
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            out.append(inline.data)
    return out

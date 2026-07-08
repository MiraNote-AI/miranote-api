"""Shared Vertex AI genai client singleton + response helpers.

Every pipeline that calls a Google model (Gemini bbox/seg detection, image
stylize, border generation, prompt expansion) goes through this one lazily-built
client, so they all reuse a single Vertex connection instead of each holding
their own. Small helpers shared by more than one image path (e.g. pulling the
inline image out of a Gemini image response) also live here.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["PROJECT_ID"],
            location=os.environ["LOCATION"],
        )
    return _client


def _extract_image_bytes(response) -> bytes:
    """Return the first inline image part from a Gemini image response.

    Shared by /stylize and /border. On failure it surfaces finish_reason /
    prompt_feedback / any text part so the cause (refusal, safety block,
    unsupported text-to-image, ...) is visible instead of a bare error.
    """
    texts = []
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                return inline.data
            if getattr(part, "text", None):
                texts.append(part.text)
    diag = []
    if not (response.candidates or []):
        diag.append("no candidates")
    for c in response.candidates or []:
        fr = getattr(c, "finish_reason", None)
        if fr:
            diag.append(f"finish_reason={fr}")
        sr = getattr(c, "safety_ratings", None)
        if sr:
            diag.append(f"safety={sr}")
    pf = getattr(response, "prompt_feedback", None)
    if pf:
        diag.append(f"prompt_feedback={pf}")
    if texts:
        diag.append("text=" + " | ".join(texts))
    detail = (" [" + "; ".join(str(d) for d in diag) + "]")[:600] if diag else ""
    raise RuntimeError("Gemini image response contained no inline image data." + detail)

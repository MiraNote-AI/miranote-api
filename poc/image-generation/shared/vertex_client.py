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
from google.genai import types

load_dotenv()

_client = None

# google-genai 2.30 added automatic continuation, on by default: when a reply
# ends with finish_reason CONTINUATION and carries a continuation_token,
# generate_content sends the request again, with no cap, and each hop is
# billed. That is an automatic retry inside the SDK, under the image buckets
# config.py describes (1/min per base model), so every call turns it off.
# Checked rather than assumed: google-genai 2.12.1, on beta until its venv is
# upgraded, has no such field and rejects it with a ValidationError.
_HAS_AUTOMATIC_CONTINUATION = (
    "automatic_continuation" in types.GenerateContentConfig.model_fields
)


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["PROJECT_ID"],
            location=os.environ["LOCATION"],
        )
    return _client


def generate_config(**fields) -> types.GenerateContentConfig | None:
    """The config for a generate_content call through this client: `fields`,
    with automatic continuation off where the installed SDK has it.

    None when there is nothing to set, so a call that passed no config before
    still passes none on an SDK without the field.
    """
    if _HAS_AUTOMATIC_CONTINUATION:
        fields["automatic_continuation"] = False
    return types.GenerateContentConfig(**fields) if fields else None


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

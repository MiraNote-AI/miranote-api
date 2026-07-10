"""Nano Banana fallback for /generate.

Imagen access is gated per GCP project; when Vertex answers 404 for the
configured Imagen model, /generate falls back to a Gemini image model
(``gemini-2.5-flash-image``) via ``generate_content``. This module keeps
the fallback's pure parts import-light so they are unit-testable without
torch or a live client.
"""
from __future__ import annotations


def is_model_unavailable(error: Exception) -> bool:
    """Vertex signals a gated/missing publisher model with a 404."""
    text = str(error)
    return "NOT_FOUND" in text or "404" in text


def build_prompt(prompt: str, aspect_ratio: str) -> str:
    return (
        f"Generate one image. {prompt}\n"
        f"Aspect ratio {aspect_ratio}. Return only the image, no words."
    )


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

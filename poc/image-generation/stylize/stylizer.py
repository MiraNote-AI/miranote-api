"""
Image-to-image style transfer via a Gemini image model (config.STYLE_MODEL).

Unlike the other Gemini calls in this package (which read response.text), the
image model returns the result as an inline image part, so we pull the bytes out
of the response via vertex_client._extract_image_bytes and normalize them to PNG.

Normalising to PNG changes the container, not the content: a model that answers
with JPEG has already discarded any alpha, and re-encoding gives an opaque PNG.
That matters for the "Edit sticker" path, whose input is a transparent cutout --
it is why that path runs the result back through /cutout afterwards.
"""

import io

from google.genai import types
from PIL import Image

from shared.vertex_client import _get_client, _extract_image_bytes


def stylize(image_bytes: bytes, instruction: str, model: str, temperature: float = 0) -> bytes:
    """Restyle a photo with the given instruction; return PNG bytes.

    Low temperature + fixed seed keep the restyle faithful to the original photo
    (the default ~1.0 lets the model reinvent content); callers can raise
    temperature for more creative reinterpretation.
    """
    mime = "image/png" if image_bytes[:8].startswith(b"\x89PNG") else "image/jpeg"
    response = _get_client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            instruction,
        ],
        config=types.GenerateContentConfig(
            # TEXT as well as IMAGE: Gemini 3.x image models expect both and
            # reject an IMAGE-only request. border.py has carried this since it
            # first called a 3.x model; this path only needed it once
            # STYLE_MODEL moved off 2.5. _extract_image_bytes drops any text
            # part and returns the inline image, so the return value is
            # unchanged on either generation of the model.
            response_modalities=["TEXT", "IMAGE"],
            temperature=temperature,
            seed=0,
        ),
    )
    raw = _extract_image_bytes(response)
    # Normalize to PNG so the output format matches the rest of the pipeline.
    img = Image.open(io.BytesIO(raw))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

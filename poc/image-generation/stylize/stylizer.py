"""
Image-to-image style transfer via Gemini 2.5 Flash Image (Nano Banana).

Unlike the other Gemini calls in this package (which read response.text), the
image model returns the result as an inline image part, so we pull the bytes out
of the response via vertex_client._extract_image_bytes and normalize them to PNG.
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
            response_modalities=["IMAGE"],
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

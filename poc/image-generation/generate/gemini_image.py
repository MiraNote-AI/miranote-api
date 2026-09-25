"""Nano Banana text-to-image, lifted out of main._call_model unchanged.

This is the same code /generate has always run for its fallback path; it moved
here so the model benchmark can call the real production path for its Gemini
baseline instead of a copy that could drift away from it. main._call_model still
owns the Imagen-first attempt and the "Imagen is gated, stop retrying" flag --
only the fallback body lives here.

Unlike every other provider, Nano Banana takes no size argument: the aspect
ratio goes into the prompt as prose (fallback.build_prompt) and the model
answers with raw bytes rather than a URL.
"""

from concurrent.futures import ThreadPoolExecutor

from shared.vertex_client import _get_client
from generate import fallback


def generate(model: str, prompt: str, aspect_ratio: str, n: int) -> list[bytes]:
    client = _get_client()

    def _one() -> bytes | None:
        response = client.models.generate_content(
            model=model,
            contents=fallback.build_prompt(prompt, aspect_ratio),
        )
        parts = fallback.image_parts(response)
        return parts[0] if parts else None

    # The images are independent; generate them concurrently so the whole
    # request stays comfortably inside client timeouts.
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda _: _one(), range(n)))
    images = [image for image in results if image]
    if not images:
        # Plain RuntimeError, not HTTPException: main.py turns this into a 502,
        # and keeping FastAPI out of the provider modules is what lets the
        # benchmark and the unit tests import them.
        raise RuntimeError("image generation returned no image")
    return images

"""HTTP client for the image-generation POC.

Canvas mode's look_at_page delegates to that service's /describe
endpoint rather than holding a second vision model here. Same shape as
text_client.py and retrieval_client.py: synchronous httpx, called from
the dispatcher inside run_turn's thread pool.

The default timeout is deliberately well under the app's turn budget
(60s): a stuck look must fail fast enough that the model still has room
to answer from the page map. A failed look is not a failed turn.
"""
from __future__ import annotations

import httpx


class ImageClient:
    def __init__(self, base_url: str, timeout: float = 25.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def describe(self, image: bytes, prompt: str) -> str:
        response = httpx.post(
            self._base_url + "/describe",
            files={"file": ("page.jpg", image, "image/jpeg")},
            params={"prompt": prompt},
            timeout=self._timeout,
        )
        response.raise_for_status()
        description = (response.json() or {}).get("description", "")
        if not description:
            raise ValueError("the image service returned no description")
        return description

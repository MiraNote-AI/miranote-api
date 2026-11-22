"""HTTP client for the image-generation POC.

Canvas mode's look_at_page delegates to that service's /describe
endpoint rather than holding a second vision model here. Same shape as
text_client.py and retrieval_client.py: synchronous httpx, called from
the dispatcher inside run_turn's thread pool.

The default timeout is deliberately well under the app's turn budget
(60s): a stuck look must fail fast enough that the model still has room
to answer from the page map. A failed look is not a failed turn.

The image service sits behind ai_auth, like /chat, and answers 401 to a
call without a token. /chat hands each canvas turn a client for its caller
(for_caller), as it does with text_client.py.
"""
from __future__ import annotations

import uuid
from typing import Dict, Optional

import httpx


class ImageClient:
    def __init__(self, base_url: str, timeout: float = 25.0, authorization: Optional[str] = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._authorization = authorization

    def for_caller(self, authorization: Optional[str]) -> "ImageClient":
        """A client that sends one caller's Authorization header. A new
        object: the shared client never holds a token, so concurrent turns
        never send each other's."""
        return ImageClient(self._base_url, self._timeout, authorization)

    def describe(self, image: bytes, prompt: str) -> str:
        # Its own X-Request-ID per call, never the chat turn's (see
        # text_client.py).
        headers: Dict[str, str] = {"X-Request-ID": str(uuid.uuid4())}
        if self._authorization:
            headers["Authorization"] = self._authorization
        response = httpx.post(
            self._base_url + "/describe",
            files={"file": ("page.jpg", image, "image/jpeg")},
            params={"prompt": prompt},
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        description = (response.json() or {}).get("description", "")
        if not description:
            raise ValueError("the image service returned no description")
        return description

"""HTTP client for the text-clean-expand POC.

The chatbot's six text-transformation tools (clean, expand, polish, shorten,
keywords, caption) delegate to text-clean-expand over HTTP instead of
duplicating prompts. The text service is the source of truth for prompts;
this client is a thin shim that builds URLs and payloads.

Synchronous (httpx.post). The dispatcher in tools.py calls into these
methods from a sync context inside run_turn's thread pool, so no event-loop
interaction is needed here.

The text service sits behind ai_auth, like /chat, and answers 401 to a call
without a token. /chat hands each turn a client for its caller
(for_caller), so a text tool runs as the person chatting.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class TextClient:
    def __init__(self, base_url: str, timeout: float = 30.0, authorization: Optional[str] = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._authorization = authorization

    def for_caller(self, authorization: Optional[str]) -> "TextClient":
        """A client that sends one caller's Authorization header.

        The beta token goes through as the app sent it; a signed-in
        writer's token makes each text call rate-limited and metered to
        that writer, the same as the app calling the route itself. A new
        object: the shared client never holds a token, so concurrent turns
        never send each other's.
        """
        return TextClient(self._base_url, self._timeout, authorization)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._base_url + path
        headers: Dict[str, str] = {}
        if self._authorization:
            headers["Authorization"] = self._authorization
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self._timeout)
        except httpx.RequestError as e:
            raise RuntimeError(
                f"text service unreachable at {self._base_url}: {e}"
            ) from e
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise RuntimeError(
                f"text service returned {resp.status_code}: {detail}"
            )
        return resp.json()

    def clean(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if context is not None:
            payload["context"] = context
        return self._post("/clean", payload)

    def expand(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if context is not None:
            payload["context"] = context
        return self._post("/expand", payload)

    def polish(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if context is not None:
            payload["context"] = context
        return self._post("/polish", payload)

    def shorten(self, text: str, target: str = "50%") -> Dict[str, Any]:
        return self._post("/shorten", {"text": text, "target": target})

    def keywords(self, text: str, max_hits: int = 10) -> Dict[str, Any]:
        return self._post("/keywords", {"text": text, "max": int(max_hits)})

    def caption(self, text: str, style: str = "instagram") -> Dict[str, Any]:
        return self._post("/caption", {"text": text, "style": style})

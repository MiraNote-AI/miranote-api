"""HTTP client for the retrieval POC.

Mirrors text_client.py: thin synchronous httpx wrapper. The chatbot
dispatcher binds this to a tool ('find_quote') so the agent can call
the retrieval service when the user wants a quote suggestion.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class RetrievalClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._base_url + path
        try:
            resp = httpx.post(url, json=payload, timeout=self._timeout)
        except httpx.RequestError as e:
            raise RuntimeError(
                f"retrieval service unreachable at {self._base_url}: {e}"
            ) from e
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise RuntimeError(
                f"retrieval service returned {resp.status_code}: {detail}"
            )
        return resp.json()

    def quotes(self, text: str, max_picks: int = 3, lang: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text, "max": int(max_picks)}
        if lang is not None:
            payload["lang"] = lang
        return self._post("/quotes", payload)

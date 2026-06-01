"""In-memory session store keyed by uuid4, with TTL eviction.

`clock` is injected so tests can advance time deterministically.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class SessionStore:
    def __init__(self, ttl_seconds: int, clock: Optional[Callable[[], float]] = None):
        self._ttl = ttl_seconds
        self._clock = clock or time.time
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._last_touched: Dict[str, float] = {}

    def create(self, seed: List[Dict[str, Any]]) -> str:
        sid = uuid.uuid4().hex
        self._data[sid] = list(seed)
        self._last_touched[sid] = self._clock()
        return sid

    def get(self, sid: str) -> List[Dict[str, Any]]:
        self._evict_expired()
        if sid not in self._data:
            raise KeyError(sid)
        self._last_touched[sid] = self._clock()
        return self._data[sid]

    def append(self, sid: str, msg: Dict[str, Any]) -> None:
        self.get(sid).append(msg)

    def replace(self, sid: str, msgs: List[Dict[str, Any]]) -> None:
        self.get(sid)  # also refreshes TTL / asserts exists
        self._data[sid] = list(msgs)

    def delete(self, sid: str) -> None:
        self._data.pop(sid, None)
        self._last_touched.pop(sid, None)

    def _evict_expired(self) -> None:
        now = self._clock()
        dead = [sid for sid, t in self._last_touched.items() if now - t > self._ttl]
        for sid in dead:
            self.delete(sid)

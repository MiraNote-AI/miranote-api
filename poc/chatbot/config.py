"""Mutable runtime config for the chatbot.

docs_root can be changed at runtime via POST /config or the set_docs_root
tool. Other fields are read once at startup. A Lock guards docs_root
writes so concurrent requests don't tear the assignment; reads rely on
Python attribute-access atomicity.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict


class ChatbotConfig:
    def __init__(
        self,
        docs_root: Path,
        model: str,
        max_tool_iterations: int,
        max_history_messages: int,
        session_ttl_seconds: int,
    ):
        self._lock = Lock()
        self.docs_root = docs_root.resolve()
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.max_history_messages = max_history_messages
        self.session_ttl_seconds = session_ttl_seconds

    def set_docs_root(self, new_path: str) -> Dict[str, object]:
        """Validate and update docs_root.

        Raises ValueError if the path does not exist or is not a directory.
        Returns {"docs_root": "<resolved-absolute-path>"} on success.
        """
        candidate = Path(new_path).expanduser().resolve()
        if not candidate.exists():
            raise ValueError(f"path does not exist: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"path is not a directory: {candidate}")
        with self._lock:
            self.docs_root = candidate
        return {"docs_root": str(candidate)}

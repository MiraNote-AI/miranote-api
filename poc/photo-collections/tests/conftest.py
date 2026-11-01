"""Shared fixtures. No test here reaches the network or a real model.

Every model call in the service goes through llm.complete, so one fake
replaces it: tests script the replies of call "a" (look) and call "b"
(write) in order, and an unscripted call fails loudly.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import pytest

POC_DIR = Path(__file__).resolve().parent.parent
if str(POC_DIR) not in sys.path:
    sys.path.insert(0, str(POC_DIR))

import llm  # noqa: E402


class FakeModel:
    """Stands in for llm.complete. Script replies per call, in order."""

    def __init__(self):
        self.calls: List[dict] = []
        self.unexpected: List[str] = []
        self._script: Dict[str, list] = {"a": [], "b": []}
        self._lock = threading.Lock()

    def reply(self, call: str, text: str, prompt_tokens: int = 1000,
              completion_tokens: int = 500, gate: Optional[threading.Event] = None):
        self._script[call].append(("reply", text, prompt_tokens, completion_tokens, gate))
        return self

    def fail(self, call: str, error: Exception):
        self._script[call].append(("fail", error, 0, 0, None))
        return self

    def respond(self, call: str, fn, times: int = 1):
        """Reply with fn(messages) -- for concurrent jobs, where the order
        of calls between threads is not fixed."""
        for _ in range(times):
            self._script[call].append(("fn", fn, 1000, 500, None))
        return self

    def __call__(self, messages, *, call: str, timeout: float):
        self.calls.append({"call": call, "messages": messages, "timeout": timeout})
        with self._lock:
            if not self._script.get(call):
                self.unexpected.append(call)
                raise AssertionError("FakeModel: no scripted reply for call " + call)
            kind, payload, pt, ct, gate = self._script[call].pop(0)
        if gate is not None:
            gate.wait(10)
        if kind == "fail":
            raise payload
        if kind == "fn":
            payload = payload(messages)
        return llm.LlmReply(
            text=payload,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=5,
            cost_cny=llm.config.cost_cny(pt, ct),
        )

    def count(self, call: str) -> int:
        return sum(1 for c in self.calls if c["call"] == call)


@pytest.fixture
def fake_model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(llm, "complete", fake)
    monkeypatch.setattr(llm, "probe", lambda: "ok")
    return fake


@pytest.fixture
def tmp_log(monkeypatch, tmp_path):
    """Send the compose log line to a temporary file."""
    import config

    path = tmp_path / "compose.jsonl"
    monkeypatch.setattr(config, "LOG_PATH", path)
    return path


@pytest.fixture
def load_main(fake_model, tmp_log, monkeypatch):
    """Load main.py fresh (the directory name has a hyphen, so importlib)."""
    monkeypatch.setenv("BETA_TOKENS", "test-token")

    def _load():
        main_path = POC_DIR / "main.py"
        spec = importlib.util.spec_from_file_location("photo_collections_main", main_path)
        main = importlib.util.module_from_spec(spec)
        sys.modules["photo_collections_main"] = main
        spec.loader.exec_module(main)
        return main

    return _load


@pytest.fixture
def client(load_main, fake_model):
    from fastapi.testclient import TestClient

    main = load_main()
    with TestClient(main.app, headers={"Authorization": "Bearer test-token"}) as c:
        c.main = main
        yield c

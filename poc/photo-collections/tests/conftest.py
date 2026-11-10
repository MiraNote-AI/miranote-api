"""Shared fixtures. No test here reaches the network or a real model.

Every model call in the service goes through llm.complete, so one fake
replaces it: tests script the replies of call "a" (look), "b" (write) and
"c" (check against the photos) in order, and an unscripted call fails
loudly -- except call "c", which by default approves every sentence and
question it is shown (see FakeModel.auto_c).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

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
        self._script: Dict[str, list] = {"a": [], "b": [], "c": []}
        self._lock = threading.Lock()
        # Unscripted call C: every sentence grounded, every question grounded
        # and safe; a sentence mentions the cover when c_cover(text) is true.
        self.c_cover: Callable[[str], bool] = lambda text: True
        self.c_ungrounded: set = set()
        self.c_unsafe: set = set()
        self.c_not_fluent: set = set()

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
            if call == "c" and not self._script["c"]:
                return self.auto_c(messages)
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

    def auto_c(self, messages) -> llm.LlmReply:
        """Call C's verdicts for the S / Q lines in its prompt."""
        task = messages[1]["content"][-1]["text"]
        sentences = re.findall(r"^(S\d+)\uff5c(.*)$", task, re.M)
        qs = re.findall(r"^(Q\d+)\uff5c(.*)$", task, re.M)
        out = {
            "sentences": [{"id": i, "grounded": t not in self.c_ungrounded,
                           "mentions_cover": bool(self.c_cover(t)),
                           "fluent": t not in self.c_not_fluent, "reason": ""}
                          for i, t in sentences],
            "questions": [{"id": i, "grounded": t not in self.c_ungrounded,
                           "safe": t not in self.c_unsafe, "reason": ""} for i, t in qs],
        }
        return llm.LlmReply(text=json.dumps(out, ensure_ascii=False), prompt_tokens=1000,
                            completion_tokens=500, latency_ms=5,
                            cost_cny=llm.config.cost_cny(1000, 500))

    def count(self, call: str) -> int:
        return sum(1 for c in self.calls if c["call"] == call)


@pytest.fixture(scope="session", autouse=True)
def _warm_jieba():
    """Build jieba's dictionary once, up front: done lazily inside the first
    job it can take seconds on a loaded machine and time out a poll."""
    import jieba

    jieba.initialize()


@pytest.fixture
def fake_model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(llm, "complete", fake)
    monkeypatch.setattr(llm, "probe", lambda: "ok")
    yield fake
    # The pipeline turns unexpected exceptions into model_error, so an
    # unscripted call would otherwise pass silently.
    assert fake.unexpected == [], "unscripted model calls: {}".format(fake.unexpected)


@pytest.fixture(scope="session", autouse=True)
def _no_real_log(tmp_path_factory):
    """The whole run writes its compose log to a temporary file. Jobs that
    outlive their test (worker threads) used to write rows into the real
    logs/photo-collections/compose.jsonl once a test's tmp_log was undone
    (full check 2026-11-05); per-test tmp_log now restores to this path."""
    import config

    real = config.LOG_PATH
    config.LOG_PATH = tmp_path_factory.mktemp("logs") / "compose.jsonl"
    yield
    config.LOG_PATH = real


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

    import beta_auth

    beta_auth._limiter._hits.clear()  # the gate's rate limit is per process
    main = load_main()
    with TestClient(main.app, headers={"Authorization": "Bearer test-token"}) as c:
        c.main = main
        yield c
    # Wait for this test's compose jobs: a worker still running after the
    # test would call the next test's FakeModel and fail that test instead.
    c.main.STORE._pool.shutdown(wait=True)

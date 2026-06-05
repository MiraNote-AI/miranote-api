from __future__ import annotations
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest


class FakeChatCompletions:
    def __init__(self):
        self.scripted: List[str] = []
        self.calls: list = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("FakeChatCompletions: no more scripted responses")
        content = self.scripted.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())

    def reply_with(self, *responses: str):
        """Queue scripted responses for the next N calls."""
        self.chat.completions.scripted.extend(responses)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the OpenAI class so main.py's client = OpenAI(...) returns the fake."""
    os.environ.setdefault("LLM_API_KEY", "fake-key-for-tests")
    fake = FakeOpenAI()
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake)
    return fake


@pytest.fixture
def client(fake_llm):
    """FastAPI TestClient with the FakeOpenAI already installed.

    main.py is loaded via importlib because the POC directory name
    (text-clean-expand) has hyphens and isn't a valid Python identifier.
    """
    from fastapi.testclient import TestClient

    main_path = Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("text_clean_expand_main", main_path)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    return TestClient(main.app), fake_llm

"""Local-dev config is not reachable from the served app.

POST /config moves docs_root for every user of the process, and the beta
token that reaches it ships inside the app. It answers only when the
server was started with CHATBOT_ALLOW_CONFIG=1; GET /health, which needs
no token at all, no longer shows the docs path.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("BETA_TOKENS", "test-token")

from poc.chatbot import main  # noqa: E402

BETA = {"Authorization": "Bearer test-token"}


@pytest.fixture
def svc(monkeypatch, tmp_path):
    import beta_auth

    beta_auth._limiter._hits.clear()
    monkeypatch.setenv("BETA_TOKENS", "test-token")
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    monkeypatch.delenv("CHATBOT_ALLOW_CONFIG", raising=False)
    monkeypatch.setattr(main.config, "docs_root", main.config.docs_root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    return TestClient(main.app), elsewhere


@pytest.mark.parametrize("flag", [None, "", "0", "true"])
@pytest.mark.parametrize("body", [{"docs_root": "<elsewhere>"}, {}])
def test_config_is_not_found_unless_the_flag_is_set(svc, monkeypatch, flag, body):
    http, elsewhere = svc
    if flag is not None:
        monkeypatch.setenv("CHATBOT_ALLOW_CONFIG", flag)
    root = main.config.docs_root
    body = {k: str(elsewhere) for k in body}
    r = http.post("/config", json=body, headers=BETA)
    assert r.status_code == 404
    assert r.json() == {"detail": "Not Found"}
    assert main.config.docs_root == root


def test_config_works_as_before_with_the_flag(svc, monkeypatch):
    http, elsewhere = svc
    monkeypatch.setenv("CHATBOT_ALLOW_CONFIG", "1")
    r = http.post("/config", json={"docs_root": str(elsewhere)}, headers=BETA)
    assert r.status_code == 200
    assert r.json() == {"docs_root": str(elsewhere.resolve())}
    assert main.config.docs_root == elsewhere.resolve()

    r = http.post("/config", json={"docs_root": str(elsewhere / "missing")}, headers=BETA)
    assert r.status_code == 400 and "does not exist" in r.json()["detail"]
    assert main.config.docs_root == elsewhere.resolve()


def test_config_still_needs_a_token_with_the_flag(svc, monkeypatch):
    http, elsewhere = svc
    monkeypatch.setenv("CHATBOT_ALLOW_CONFIG", "1")
    root = main.config.docs_root
    assert http.post("/config", json={"docs_root": str(elsewhere)}).status_code == 401
    assert main.config.docs_root == root


def test_health_does_not_show_the_docs_path(svc):
    http, _ = svc
    r = http.get("/health")
    assert r.status_code == 200
    assert "docs_root" not in r.json()
    assert str(main.config.docs_root) not in r.text
    assert "set_docs_root" not in r.json()["tools"]

"""Concurrency and retention are configuration (phase 2, task 1)."""

from __future__ import annotations

import importlib
import time

import pytest

import config
import jobs
from tests.test_api import _Clock, _files, _submit
from tests.test_compose import _a_json, _b_json, _meta


@pytest.fixture
def reload_config(monkeypatch):
    """Reload config under the env set by the test; restore it afterwards.
    A reload resets LOG_PATH to the real log, so it is put back to the
    run's temporary log each time (conftest._no_real_log)."""
    log_path = config.LOG_PATH

    def _reload():
        cfg = importlib.reload(config)
        cfg.LOG_PATH = log_path
        return cfg

    yield _reload
    monkeypatch.undo()
    _reload()


def test_workers_default_40_and_llm_pool_80(monkeypatch, reload_config):
    for name in ("PHOTO_WORKERS", "PHOTO_LLM_WORKERS"):
        monkeypatch.setenv(name, "")  # beats dotenv; "" falls back to the default
    cfg = reload_config()
    assert cfg.WORKERS == 40
    assert cfg.LLM_WORKERS == 80


def test_workers_read_from_env(monkeypatch, reload_config):
    monkeypatch.setenv("PHOTO_WORKERS", "7")
    monkeypatch.setenv("PHOTO_LLM_WORKERS", "9")
    cfg = reload_config()
    assert (cfg.WORKERS, cfg.LLM_WORKERS) == (7, 9)


def test_llm_pool_uses_configured_size():
    import llm

    assert llm._calls._max_workers == config.LLM_WORKERS


def test_job_ttl_is_one_hour():
    assert config.JOB_TTL_SECONDS == 3600


def test_expired_after_one_hour_reads_404_expired(client, fake_model):
    clock = _Clock()
    client.main.STORE = jobs.JobStore(clock=clock)
    store = client.main.STORE
    job_id = store.submit(lambda deadline: {"status": "ok", "memory": {}})
    end = time.monotonic() + 2
    while store.get(job_id) == {"status": "running"} and time.monotonic() < end:
        time.sleep(0.01)
    clock.now += 3599
    assert client.get("/collections/jobs/" + job_id).status_code == 200
    clock.now += 2
    r = client.get("/collections/jobs/" + job_id)
    assert (r.status_code, r.json()) == (404, {"status": "expired"})


def test_thirty_five_groups_finish_together(client, fake_model, tmp_log):
    def slow(reply):
        def fn(messages):
            time.sleep(2)
            return reply
        return fn

    n = 35
    fake_model.respond("a", slow(_a_json()), times=n)
    fake_model.respond("b", slow(_b_json()), times=n)
    started = time.monotonic()
    ids = []
    for i in range(n):
        r = _submit(client, _meta(group_key="g{}".format(i)))
        assert r.status_code == 202
        ids.append(r.json()["job_id"])
    pending = set(ids)
    end = time.monotonic() + 30
    while pending and time.monotonic() < end:
        for job_id in list(pending):
            # Straight from the store: the beta gate rate-limits HTTP polls.
            body = client.main.STORE.get(job_id)
            if body["status"] != "running":
                assert body["status"] == "ok"
                pending.discard(job_id)
        time.sleep(0.2)
    elapsed = time.monotonic() - started
    assert not pending
    # One job takes about 4 s (two 2 s calls); serial would be 140 s.
    assert elapsed < 12


def test_small_llm_pool_warns_at_startup(load_main, monkeypatch, capsys):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "WORKERS", 40)
    monkeypatch.setattr(config, "LLM_WORKERS", 10)
    main = load_main()
    with TestClient(main.app):
        pass
    assert "PHOTO_LLM_WORKERS=10" in capsys.readouterr().out

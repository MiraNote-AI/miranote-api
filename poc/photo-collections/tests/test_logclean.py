"""The compose log is kept 7 days; stats skip test entries (phase 2, task 4)."""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import compose
import logclean

NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=timezone.utc)


def _line(days_ago, **extra):
    ts = (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")
    return json.dumps(dict({"group_key": "g", "ts": ts, "status": "ok"}, **extra)) + "\n"


def test_purge_drops_lines_older_than_7_days(tmp_path):
    path = tmp_path / "compose.jsonl"
    old, edge, fresh = _line(8, n="old"), _line(7, n="edge"), _line(1, n="fresh")
    path.write_text(old + edge + fresh, encoding="utf-8")
    assert logclean.purge(path, NOW) == 1
    assert path.read_text(encoding="utf-8") == edge + fresh


def test_purge_keeps_undatable_lines(tmp_path):
    path = tmp_path / "compose.jsonl"
    keep = [
        '{"group_key": "no-ts"}\n',
        '{"ts": "not a time"}\n',
        '{"ts": 12345}\n',
        "not json at all\n",
        "[1, 2]\n",
        "\n",
    ]
    path.write_text("".join(keep) + _line(90), encoding="utf-8")
    assert logclean.purge(path, NOW) == 1
    assert path.read_text(encoding="utf-8") == "".join(keep)


def test_purge_splits_on_newline_only(tmp_path):
    path = tmp_path / "compose.jsonl"
    odd = "a\u2028b\u2029c\x85d\x0be\x0cf\x1cg"
    old = _line(90, msg=odd).replace("\\u2028", "\u2028")
    fresh = _line(1, msg=odd).replace("\\u2028", "\u2028")
    assert "\u2028" in old and "\u2028" in fresh
    path.write_text(old + fresh, encoding="utf-8")
    assert logclean.purge(path, NOW) == 1
    assert path.read_text(encoding="utf-8") == fresh


def test_purge_missing_file_is_zero(tmp_path):
    assert logclean.purge(tmp_path / "nope.jsonl", NOW) == 0


def test_purge_nothing_old_leaves_file_untouched(tmp_path):
    path = tmp_path / "compose.jsonl"
    path.write_text(_line(1) + _line(2), encoding="utf-8")
    before = path.stat().st_mtime_ns
    assert logclean.purge(path, NOW) == 0
    assert path.stat().st_mtime_ns == before
    assert [p.name for p in tmp_path.iterdir()] == ["compose.jsonl"]


def test_purge_takes_the_log_lock(tmp_path):
    path = tmp_path / "compose.jsonl"
    path.write_text(_line(90), encoding="utf-8")
    done = threading.Event()

    def run():
        logclean.purge(path, NOW)
        done.set()

    with compose.LOG_LOCK:
        t = threading.Thread(target=run)
        t.start()
        assert not done.wait(0.3)  # blocked while a writer holds the lock
    assert done.wait(2)
    t.join()


def test_purge_and_writes_lose_no_line(tmp_path, monkeypatch):
    path = tmp_path / "compose.jsonl"
    monkeypatch.setattr(compose.config, "LOG_PATH", path)
    path.write_text("".join(_line(90) for _ in range(50)), encoding="utf-8")
    fresh_ts = NOW.isoformat(timespec="seconds")

    def writer(k):
        for i in range(40):
            compose._write_log({"group_key": "w{}_{}".format(k, i), "ts": fresh_ts})

    def purger():
        for _ in range(10):
            logclean.purge(path, NOW)

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(4)]
    threads.append(threading.Thread(target=purger))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logclean.purge(path, NOW)
    keys = [json.loads(x)["group_key"] for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(keys) == 160 and len(set(keys)) == 160


class _Stop(Exception):
    pass


def test_daily_loop_purges_again_after_a_day(tmp_path):
    path = tmp_path / "compose.jsonl"
    path.write_text(_line(40, n=1) + _line(1, n=2), encoding="utf-8")
    now = {"t": NOW}
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            # a day passes; a line written meanwhile is now 31 days old
            now["t"] = NOW + timedelta(days=1)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(_line(31, n=3))
            return
        raise _Stop

    with pytest.raises(_Stop):
        logclean.run_daily(path, lambda: now["t"], sleep)
    assert sleeps == [24 * 3600, 24 * 3600]
    # n=2 was 1 day old, now 2 days: kept; n=1 purged on the first pass,
    # n=3 (31 days at the clock of the second pass) purged on the second.
    rows = [json.loads(x)["n"] for x in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [2]


def test_daily_loop_survives_a_purge_error(tmp_path, monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise OSError("disk")

    monkeypatch.setattr(logclean, "purge", boom)

    def sleep(seconds):
        if len(calls) >= 2:
            raise _Stop

    with pytest.raises(_Stop):
        logclean.run_daily(tmp_path / "x", lambda: NOW, sleep)
    assert len(calls) == 2


def _load_stats():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    try:
        import log_stats
    finally:
        sys.path.pop(0)
    return log_stats


def test_stats_skip_cc_entries(tmp_path):
    log_stats = _load_stats()
    path = tmp_path / "compose.jsonl"
    rows = [
        {"group_key": "2026-10-18_A", "status": "ok", "reason": "", "latencyMs": 1000, "costCny": 0.01},
        {"group_key": "2026-10-18_B", "status": "rejected", "reason": "sensitive",
         "latencyMs": 3000, "costCny": 0.03},
        {"group_key": "cc_test1", "status": "error", "reason": "timeout", "latencyMs": 99999, "costCny": 5.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    s = log_stats.summarize(path)
    assert s["count"] == 2
    assert s["skipped_cc"] == 1
    assert s["by_status"] == {"ok": 1, "rejected": 1}
    assert s["by_reason"] == {"sensitive": 1}
    assert s["mean_latency_ms"] == 2000
    assert s["total_cost_cny"] == pytest.approx(0.04)


def test_stats_keep_lines_with_unicode_line_separator(tmp_path):
    log_stats = _load_stats()
    path = tmp_path / "compose.jsonl"
    rows = [
        {"group_key": "g1", "status": "ok", "reason": "a\u2028b", "latencyMs": 1000, "costCny": 0.01},
        {"group_key": "g2", "status": "ok", "reason": "", "latencyMs": 3000, "costCny": 0.03},
    ]
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    assert "\u2028" in text
    path.write_text(text, encoding="utf-8")
    s = log_stats.summarize(path)
    assert s["count"] == 2
    assert s["total_cost_cny"] == pytest.approx(0.04)


def test_startup_purges_the_log(tmp_log, load_main, monkeypatch):
    from fastapi.testclient import TestClient

    real_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(timespec="seconds")
    tmp_log.write_text(
        json.dumps({"ts": old_ts, "n": "old"}) + "\n" + json.dumps({"ts": real_now, "n": "new"}) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(logclean, "DAY_SECONDS", 3600)
    main = load_main()
    with TestClient(main.app):
        import time

        end = time.monotonic() + 3
        while time.monotonic() < end and "old" in tmp_log.read_text(encoding="utf-8"):
            time.sleep(0.02)
    text = tmp_log.read_text(encoding="utf-8")
    assert '"old"' not in text and '"new"' in text


def test_tests_never_write_the_real_log():
    # Full check 2026-11-05: test rows were found in the real log.
    import config

    real = config.REPO_ROOT / "logs" / "photo-collections" / "compose.jsonl"
    assert config.LOG_PATH != real

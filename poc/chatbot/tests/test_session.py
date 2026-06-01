from __future__ import annotations
import pytest

from poc.chatbot.session import SessionStore


def test_create_returns_uuid():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    assert isinstance(sid, str) and len(sid) >= 32
    assert store.get(sid) == [{"role": "system", "content": "hi"}]


def test_append_extends_history():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    store.append(sid, {"role": "user", "content": "hello"})
    msgs = store.get(sid)
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_replace_overwrites_history():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    new = [{"role": "system", "content": "hi"}, {"role": "user", "content": "x"}]
    store.replace(sid, new)
    assert store.get(sid) == new


def test_delete_removes_session():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[])
    store.delete(sid)
    with pytest.raises(KeyError):
        store.get(sid)


def test_unknown_session_raises():
    store = SessionStore(ttl_seconds=60)
    with pytest.raises(KeyError):
        store.get("bogus")


def test_ttl_evicts_old_sessions():
    clock = [0.0]
    store = SessionStore(ttl_seconds=10, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    clock[0] = 5
    assert store.get(sid)  # still alive
    clock[0] = 20  # past TTL
    with pytest.raises(KeyError):
        store.get(sid)


def test_access_refreshes_ttl():
    clock = [0.0]
    store = SessionStore(ttl_seconds=10, clock=lambda: clock[0])
    sid = store.create(seed=[])
    clock[0] = 8
    store.get(sid)        # refresh
    clock[0] = 15         # 7 since last touch -> alive
    assert store.get(sid) == []

"""Taste events (design 4.2), their switch, and their deletion, against a
real Postgres."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from fastapi.testclient import TestClient

import main
import taste
from tests.conftest import run_sql
from tests.test_delete import run_purge
from tests.test_usage import new_user

NOW = dt.datetime.now(dt.timezone.utc)


@pytest.fixture
def client(clean_db, monkeypatch):
    monkeypatch.setenv("TASTE_EVENTS_ENABLED", "1")
    with TestClient(main.app) as c:
        yield c


def auth(client, user_id):
    return {"Authorization": "Bearer " + client.app.state.signer.issue(user_id, NOW)}


def event(**kw):
    e = {
        "event_id": str(uuid.uuid4()),
        "session_id": "00000000-0000-0000-0000-0000000000aa",
        "occurred_at": NOW.isoformat(),
        "app_version": "1.0.2",
        "build": "42",
        "tz_offset_minutes": -480,
        "name": "font_choice",
        "surface": "font_picker",
        "item_type": "font",
        "item_id": "serif-02",
        "position": 3,
        "action": "selected",
    }
    e.update(kw)
    return e


def post(client, user_id, *events, version=taste.EVENT_VERSION):
    return client.post(
        "/events", headers=auth(client, user_id), json={"event_version": version, "events": list(events)}
    )


def stored(user_id=None):
    if user_id is None:
        return run_sql("SELECT count(*) FROM taste_events")[0][0]
    return run_sql("SELECT count(*) FROM taste_events WHERE user_id = :u", u=user_id)[0][0]


def add_event(user_id, ago):
    run_sql(
        "INSERT INTO taste_events (event_id, user_id, session_id, occurred_at, app_version, build, "
        "tz_offset_minutes, event_version, name, surface, item_type, item_id, action) "
        "VALUES (gen_random_uuid(), :u, gen_random_uuid(), now() - CAST(:ago AS interval), '1.0', '1', "
        "0, 1, 'font_choice', 'font_picker', 'font', 'serif-02', 'shown')",
        u=user_id, ago=ago,
    )


# --- POST /events ------------------------------------------------------------


def test_a_batch_is_stored_with_every_field(client):
    user = new_user()
    request_id = uuid.uuid4()
    e = event(name="image_style_choice", item_type="image_style", item_id="watercolor",
              action="regenerated", request_id=str(request_id))
    r = post(client, user, e, event(position=None, action="shown"))
    assert r.status_code == 200, r.text
    assert r.json() == {"stored": 2, "duplicates": 0}
    row = run_sql(
        "SELECT user_id, name, item_type, item_id, action, request_id, event_version, tz_offset_minutes, "
        "app_version, build, received_at IS NOT NULL FROM taste_events WHERE event_id = :e",
        e=e["event_id"],
    )[0]
    assert tuple(row) == (user, "image_style_choice", "image_style", "watercolor", "regenerated",
                          request_id, 1, -480, "1.0.2", "42", True)


def test_a_resent_event_id_is_stored_once(client):
    user = new_user()
    e = event()
    assert post(client, user, e).json() == {"stored": 1, "duplicates": 0}
    assert post(client, user, e, event()).json() == {"stored": 1, "duplicates": 1}
    # The same id twice in one batch, too.
    again = event()
    assert post(client, user, again, again).json() == {"stored": 1, "duplicates": 1}
    assert stored(user) == 3


def test_every_registered_name_is_accepted(client):
    user = new_user()
    events = [event(name=name, item_type=item_type) for name, item_type in taste.EVENTS.items()]
    assert post(client, user, *events).json() == {"stored": len(taste.EVENTS), "duplicates": 0}


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "page_text"},  # not registered
        {"item_type": "color"},  # not the font_choice item type
        {"action": "liked"},
        {"text": "my day was great"},  # any field outside the schema
        {"prompt": "a cat"},
        {"item_id": "a whole sentence about my day"},  # catalog ids only
        {"surface": "Font Picker"},
        {"item_id": "x" * 65},
        {"app_version": "one"},
        {"occurred_at": "2026-11-17T10:00:00"},  # no time zone
        {"occurred_at": (NOW + dt.timedelta(days=2)).isoformat()},
        {"tz_offset_minutes": 900},
        {"position": -1},
    ],
)
def test_a_bad_event_refuses_the_whole_batch(client, bad):
    user = new_user()
    r = post(client, user, event(), event(**bad))
    assert r.status_code == 422, r.text
    assert stored() == 0


def test_batch_shape_is_checked(client):
    user = new_user()
    assert post(client, user).status_code == 422  # empty
    assert post(client, user, *[event() for _ in range(taste.MAX_BATCH + 1)]).status_code == 422
    assert post(client, user, event(), version=2).status_code == 422
    r = client.post("/events", headers=auth(client, user),
                    json={"event_version": 1, "events": [event()], "note": "hi"})
    assert r.status_code == 422
    assert stored() == 0
    assert post(client, user, *[event() for _ in range(taste.MAX_BATCH)]).status_code == 200


def test_signed_out_or_deleting_is_401(client):
    assert client.post("/events", json={"event_version": 1, "events": [event()]}).status_code == 401
    assert post(client, new_user(deleting=True), event()).status_code == 401
    assert stored() == 0


def test_the_endpoint_is_404_unless_turned_on(clean_db, monkeypatch):
    monkeypatch.delenv("TASTE_EVENTS_ENABLED", raising=False)
    user = new_user()
    with TestClient(main.app) as c:
        assert post(c, user, event()).status_code == 404
        # The switch still works, so a user can always turn it off.
        assert c.put("/me/preferences", headers=auth(c, user),
                     json={"taste_events_enabled": False}).status_code == 200
    assert stored() == 0


# --- the switch: GET / PUT /me/preferences ------------------------------------


def test_the_switch_is_on_by_default(client):
    user = new_user()
    r = client.get("/me/preferences", headers=auth(client, user))
    assert r.status_code == 200 and r.json() == {"taste_events_enabled": True}
    assert client.get("/me/preferences").status_code == 401


def test_turning_the_switch_off_deletes_that_users_events_and_stops_collection(client):
    user, other = new_user(), new_user()
    post(client, user, event(), event())
    post(client, other, event())

    r = client.put("/me/preferences", headers=auth(client, user), json={"taste_events_enabled": False})
    assert r.status_code == 200 and r.json() == {"taste_events_enabled": False}
    assert stored(user) == 0 and stored(other) == 1
    assert client.get("/me/preferences", headers=auth(client, user)).json() == {"taste_events_enabled": False}

    r = post(client, user, event())
    assert r.status_code == 403
    assert stored(user) == 0

    # On again: collection resumes; nothing old comes back.
    client.put("/me/preferences", headers=auth(client, user), json={"taste_events_enabled": True})
    assert post(client, user, event()).json() == {"stored": 1, "duplicates": 0}
    assert stored(user) == 1


def test_preferences_take_only_the_switch(client):
    user = new_user()
    for body in ({}, {"taste_events_enabled": "maybe"}, {"taste_events_enabled": True, "x": 1}):
        r = client.put("/me/preferences", headers=auth(client, user), json=body)
        assert r.status_code == 422, body


# --- deletion (retention: test_delete.py) ---------------------------------------------------


def test_delete_me_deletes_the_events_at_once(client):
    user, other = new_user(), new_user()
    post(client, user, event())
    post(client, other, event())
    # No Sign in with Apple key here: 503, but our data is deleted.
    r = client.request("DELETE", "/me", headers=auth(client, user), json={"authorization_code": "c"})
    assert r.json()["deleted"] is True
    assert stored(user) == 0 and stored(other) == 1
    run_purge()
    assert run_sql("SELECT count(*) FROM users") == [(1,)]


def test_purging_an_account_cascades_to_its_events(client):
    user = new_user(deleting=True)
    add_event(user, dt.timedelta(hours=1))
    assert run_purge()["users"] == 1
    assert stored() == 0


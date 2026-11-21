"""Repeat-theme judge rounds: round 1 (name a pile), round 2 (check push
candidates) and shorten. The model is the FakeModel; nothing is metered."""

from __future__ import annotations

import json

import pytest

from fake_accounts import FakeAccounts
from tests.test_compose import jpeg

WHERE = {"daypart": "evening", "daypart_share": 0.8, "hour_from": 18, "hour_to": 20,
         "weekpart": "weekday", "places": 1}
REPLY = json.dumps({"sensitive": False, "pile_name": "x"})


def _group(gid, where=None):
    return {"id": gid, "days": 6, "where": where or dict(WHERE)}


def _meta1(n=3, **over):
    meta = {
        "pile": {"photo_count": 51, "day_count": 27},
        "split_a": [_group("A1"), _group("A2")],
        "split_b": [_group("B1")],
        "photos": [{"n": i, "a": "A1", "b": "B1", "labels": ["soup", "bowl"]} for i in range(1, n + 1)],
        "old": None,
    }
    meta.update(over)
    return meta


def _meta2(n=3, **over):
    meta = {"theme": {"name": "THEME_NAME", "noun": "晚饭", "same_part": "傍晚在家"},
            "photos": [{"n": i} for i in range(1, n + 1)], "pushed": ["PUSHED_ONE", "PUSHED_TWO"],
            "blocked": ["BLOCKED_ONE"]}
    meta.update(over)
    return meta


def _files(n=3, px=8):
    return [("photos", ("{}.jpg".format(i), jpeg(px, px), "image/jpeg")) for i in range(1, n + 1)]


def _post1(client, meta=None, files=None, **kw):
    return client.post("/collections/rounds/1", data={"meta": json.dumps(meta or _meta1())},
                       files=files if files is not None else _files(), **kw)


def _post2(client, meta=None, files=None, **kw):
    return client.post("/collections/rounds/2", data={"meta": json.dumps(meta or _meta2())},
                       files=files if files is not None else _files(), **kw)


def _parts(call):
    return call["messages"][-1]["content"]


def _text(call):
    return _parts(call)[0]["text"]


def test_round1_sends_tagged_images_and_rendered_text(client, fake_model):
    fake_model.reply("r1", REPLY)
    r = _post1(client)
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert fake_model.count("r1") == 1 and len(fake_model.calls) == 1
    parts = _parts(fake_model.calls[0])
    assert parts[0]["type"] == "text"
    for needle in ("共 51 张", "27 个不同的日子", "下面给你其中 3 张", "A1、A2", "B1", "图1 A1/B1", "soup, bowl"):
        assert needle in parts[0]["text"], needle
    assert "<整堆的名字>" in parts[0]["text"]
    rest = parts[1:]
    assert [p["type"] for p in rest] == ["text", "image_url"] * 3
    assert [rest[i]["text"] for i in (0, 2, 4)] == ["图 1", "图 2", "图 3"]
    assert rest[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_round1_renders_where_lines(client, fake_model):
    fake_model.reply("r1", REPLY)
    other = {"daypart": None, "daypart_share": 0, "hour_from": None, "hour_to": None,
             "weekpart": "mixed", "places": None}
    meta = _meta1(split_a=[_group("A1"), _group("A2", other)],
                  split_b=[_group("B1", dict(WHERE, places=3, weekpart="weekend"))])
    where = _post1(client, meta).json()["where"]
    assert where["A1"] == "多在傍晚，18–20 点，工作日，同一个地点"
    assert where["A2"] == ""
    assert where["B1"] == "多在傍晚，18–20 点，周末，3 个地点"
    assert "A1：多在傍晚，18–20 点，工作日，同一个地点" in _text(fake_model.calls[0])


def test_round1_reask_prepends_old_names(client, fake_model):
    import texts
    fake_model.reply("r1", REPLY)
    _post1(client, _meta1(old={"pile_name": "每天这一顿", "groups": ["一碗汤", "面"]}))
    head = texts.render("repeat_round1_reask", pile_name="每天这一顿", groups="一碗汤」「面")
    assert _text(fake_model.calls[0]).startswith(head)
    assert "「一碗汤」「面」" in head


def test_round1_without_old_has_no_reask(client, fake_model):
    fake_model.reply("r1", REPLY)
    _post1(client)
    assert "以前起过名字" not in _text(fake_model.calls[0])


def test_round1_returns_unparseable_reply_verbatim(client, fake_model):
    fake_model.reply("r1", "not json")
    r = _post1(client)
    assert r.status_code == 200 and r.json()["reply"] == "not json"


def test_round1_usage_in_response(client, fake_model):
    fake_model.reply("r1", REPLY, prompt_tokens=2000, completion_tokens=250)
    usage = _post1(client).json()["usage"]
    assert usage["prompt_tokens"] == 2000 and usage["completion_tokens"] == 250
    assert usage["model"] == "qwen3.7-flash" and usage["cost_cny"] > 0


def test_round1_rejects_unknown_photo_number(client):
    r = _post1(client, files=_files(2) + [("photos", ("9.jpg", jpeg(), "image/jpeg"))])
    assert r.status_code == 400 and r.json()["reason"] == "bad_request"
    assert _post1(client, _meta1(photos=[{"n": 1, "a": "A1", "b": "B1"}, {"n": 3, "a": "A1", "b": "B1"}]),
                  files=_files(2)).status_code == 400


def test_round1_rejects_oversized_image(client):
    assert _post1(client, files=_files(3, px=337)).status_code == 400


def test_round1_rejects_21_photos(client):
    r = _post1(client, _meta1(n=21), files=_files(21))
    assert r.status_code == 400


def test_round1_rejects_bad_meta(client):
    assert _post1(client, _meta1(split_a=[_group("A7")])).status_code == 400
    assert _post1(client, _meta1(pile={"photo_count": 0, "day_count": 1})).status_code == 400
    bad = [{"n": 1, "a": "B1", "b": "B1"}] + [{"n": i, "a": "A1", "b": "B1"} for i in (2, 3)]
    assert _post1(client, _meta1(photos=bad)).status_code == 400
    assert _post1(client, _meta1(old={"pile_name": 5, "groups": []})).status_code == 400
    r = client.post("/collections/rounds/1", data={"meta": "nope"}, files=_files())
    assert r.status_code == 400


def test_round2_rejects_16_photos(client):
    assert _post2(client, _meta2(n=16), files=_files(16)).status_code == 400


def test_rounds_require_token(client, fake_model):
    client.headers.pop("Authorization")
    for r in (_post1(client), _post2(client),
              client.post("/collections/rounds/shorten", json={"items": []})):
        assert r.status_code == 401
    assert fake_model.calls == []


def test_round2_lists_pushed_and_blocked_names(client, fake_model):
    fake_model.reply("r2", json.dumps({"photos": [], "similar_to": []}))
    r = _post2(client)
    assert r.status_code == 200 and "where" not in r.json()
    text = _text(fake_model.calls[0])
    for needle in ("THEME_NAME", "晚饭", "傍晚在家", "3 张", "PUSHED_ONE、PUSHED_TWO", "BLOCKED_ONE",
                   "<名单里的名字>"):
        assert needle in text, needle
    assert "工作日午饭" not in text
    assert [p["type"] for p in _parts(fake_model.calls[0])[1:]] == ["text", "image_url"] * 3


def test_round2_timeout_answers_502(client, fake_model):
    import llm
    fake_model.fail("r2", llm.ModelTimeout("slow"))
    r = _post2(client)
    assert r.status_code == 502
    body = r.json()
    assert body["status"] == "error" and body["reason"] == "timeout"
    assert body["usage"]["prompt_tokens"] == 0 and body["usage"]["cost_cny"] == 0


def test_round1_model_error_answers_502(client, fake_model):
    import llm
    fake_model.fail("r1", llm.ModelError("boom"))
    r = _post1(client)
    assert r.status_code == 502 and r.json()["reason"] == "model_error"


def test_shorten_is_text_only(client, fake_model):
    fake_model.reply("rs", '{"items": []}')
    items = [{"key": "g1", "name": "N" * 14, "same_part": "S" * 30}]
    r = client.post("/collections/rounds/shorten", json={"items": items})
    assert r.status_code == 200 and r.json()["reply"] == '{"items": []}'
    assert fake_model.count("rs") == 1
    parts = _parts(fake_model.calls[0])
    assert all(p["type"] == "text" for p in parts)
    assert "N" * 14 in parts[0]["text"] and "g1" in parts[0]["text"]


def test_shorten_rejects_bad_items(client):
    for body in ({"items": []}, {"items": [{"key": "a", "name": "b"}]},
                 {"items": [{"key": "k%d" % i, "name": "n", "same_part": "s"} for i in range(8)]}, []):
        assert client.post("/collections/rounds/shorten", json=body).status_code == 400
    assert client.post("/collections/rounds/shorten", content=b"{", headers={
        "Content-Type": "application/json"}).status_code == 400


@pytest.mark.parametrize("kind", ["round1", "round2", "shorten"])
def test_round_logged_without_images(client, fake_model, tmp_log, kind):
    reply = '{"ok": "REPLY_TEXT"}'
    if kind == "round1":
        fake_model.reply("r1", reply, prompt_tokens=1234, completion_tokens=56)
        _post1(client)
    elif kind == "round2":
        fake_model.reply("r2", reply, prompt_tokens=1234, completion_tokens=56)
        _post2(client)
    else:
        fake_model.reply("rs", reply, prompt_tokens=1234, completion_tokens=56)
        client.post("/collections/rounds/shorten",
                    json={"items": [{"key": "a", "name": "b", "same_part": "c"}]})
    lines = tmp_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["kind"] == kind and entry["ts"]
    assert entry["promptTokens"] == 1234 and entry["completionTokens"] == 56 and entry["costCny"] > 0
    assert entry["reply"] == reply
    assert "base64" not in lines[0] and "data:image" not in lines[0]


def test_round_log_is_purged_by_logclean(client, fake_model, tmp_log):
    from datetime import datetime, timedelta, timezone

    import logclean
    fake_model.reply("r1", REPLY)
    _post1(client)
    later = datetime.now(timezone.utc) + timedelta(days=8)
    assert logclean.purge(tmp_log, later) == 1


@pytest.fixture
def accounts(monkeypatch):
    monkeypatch.delenv("ACCEPT_BETA_TOKENS", raising=False)
    fake = FakeAccounts(caps={"photo_push": 1})
    fake.install(monkeypatch)
    return fake


def test_rounds_not_metered(client, fake_model, accounts):
    fake_model.reply("r1", REPLY).reply("r2", REPLY).reply("rs", REPLY)
    headers = accounts.headers()
    assert _post1(client, headers=headers).status_code == 200
    assert _post2(client, headers=headers).status_code == 200
    r = client.post("/collections/rounds/shorten", headers=headers,
                    json={"items": [{"key": "a", "name": "b", "same_part": "c"}]})
    assert r.status_code == 200
    assert [c["call"] for c in fake_model.calls] == ["r1", "r2", "rs"]
    assert accounts.rows == {}

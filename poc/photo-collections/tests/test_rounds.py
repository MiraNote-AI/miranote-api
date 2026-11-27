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
    assert where["A1"] == "多在傍晚，18–20 点，工作日，一个固定的地点"
    assert where["A2"] == ""
    assert where["B1"] == "多在傍晚，18–20 点，周末，3 个地点"
    assert "A1：多在傍晚，18–20 点，工作日，一个固定的地点" in _text(fake_model.calls[0])


def test_round1_renders_group_tags(client, fake_model):
    fake_model.reply("r1", REPLY)
    a1 = dict(_group("A1"), tags=[{"label": "cat", "in": 1.0, "out": 0.0},
                                  {"label": "outdoor", "in": 0.95, "out": 0.8}])
    _post1(client, _meta1(split_a=[a1, _group("A2")]))
    lines = _text(fake_model.calls[0]).split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("  · A1："))
    assert lines[i + 1] == "      cat 组里100% / 组外0%"
    assert lines[i + 2] == "      outdoor 组里95% / 组外80%"
    assert lines[i + 3].startswith("  · A2：")


@pytest.mark.parametrize("tags", [
    [{"label": "cat", "in": 1.2, "out": 0.0}],
    [{"label": "", "in": 0.5, "out": 0.0}],
    [{"label": "t{}".format(i), "in": 0.5, "out": 0.1} for i in range(9)],
    "x",
])
def test_round1_rejects_bad_tags(client, tags):
    meta = _meta1(split_a=[dict(_group("A1"), tags=tags)])
    assert _post1(client, meta).status_code == 400


def test_round1_ignores_tags_on_split_b(client, fake_model):
    fake_model.reply("r1", REPLY)
    b1 = dict(_group("B1"), tags=[{"label": "cat", "in": 1.0, "out": 0.0}])
    _post1(client, _meta1(split_b=[b1]))
    assert " / 组外" not in _text(fake_model.calls[0])


def test_round1_without_tags_renders_no_tag_lines(client, fake_model):
    fake_model.reply("r1", REPLY)
    _post1(client)
    assert " / 组外" not in _text(fake_model.calls[0])


def test_round1_no_groups_tells_model_to_leave_groups_empty(client, fake_model):
    import texts
    fake_model.reply("r1", REPLY)
    _post1(client, _meta1(split_a=[], split_b=[]))
    text = _text(fake_model.calls[0])
    line = texts.json("names")["round1_no_groups"]
    assert line in text
    assert text.index("共 51 张") < text.index(line) < text.index("【每张图的组号】")
    fake_model.reply("r1", REPLY)
    _post1(client)
    assert line not in _text(fake_model.calls[1])


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


# The third-version prompts of the draft (feature-photo-trigger/repeat themes,
# Qwen judge prompt draft), written out so drift is caught. The draft lives
# outside the repo. The example JSON follows the prompt body.
ROUND1_PROMPT = """你是一个相册整理助手。下面的照片来自同一个人的相册，被手机归成了一堆，
分布在很多个不同的日子里。手机还用两种方法把这一堆分成了小组：
分法 A 按拍的是什么，分法 B 按什么时候、在哪儿拍。每张图都标了它在 A、B 里属于哪个组。

请按顺序做：

1. 敏感：这一堆是不是药、医院、体检单、体重或身体、证件这类内容？是的话，
   sensitive 填 true，后面几步都不用做。

2. 整堆的名字：2–10 个字，具体、口语，像这个人自己给相册合集起的名字。
   名字要看着这些照片起；下面和回答格式里的例子只是说明写法，不要照抄。
   名字只写这一堆里大多数照片都有的东西；只在一两张里出现的东西不写进名字。那一两张如果拍的主要不是大多数照片都有的东西，才放进第 5 步「不属于这一堆」。
   不评价、不说教，不写「同一个」，不写次数。
   再给一个 2–4 个字的短名词，用在「留住这些{短名词}」这句话里。
   例：窗外的晚霞 → 晚霞；家里的晚饭 → 晚饭；各种拉花 → 拉花。

3. 比较分法 A 和 B，选一种：
   - 选「一组里的共同点一眼看得出、组和组的区别明显、名字说得出口」的那种；
   - 都说不通，就选 none，整堆就是一个主题。

4. 对选中的那种分法，每个组：
   - 起名（规则同第 2 步），并给一个短名词。名字只写这一组大多数照片都有的东西；分法 B 的组是按时间和地点分的，名字写这个场合（「工作日的晚饭」「周末中午的便当」），不要缩成某一样菜或东西（不写「汤饭」），除非这一组几乎每张都是它；
   - 两组的名字其实是一回事，就合并；看不出共同点的，标「去掉」；
   - 只留「像一种习惯或一种收集」的组：同一个时段、地点、器具反复出现，
     或者看得出这个人在特意拍这类东西。只是那天碰巧不一样的（自己带的饭今天海鲜、
     明天肉），标「去掉」。拿不准就不拆。
   - 每个留下的组写一句「每次都一样的是什么」，不超过 20 个字（含标点），
     只写这一组大多数照片都有的。
     例：「傍晚在家，推开窗的时候」「端上来、还没喝的那一刻」。
   - 只有选了分法 A 时，才给每个组选 1–3 个证据标签：只能从这个组下面列的标签里选，选「组里占比高、组外占比低」的（比如组里 100%、组外 0%）；组外也很常见的标签（组里 95%、组外 80%）不要选。没有列占比时，从图片后面的手机标签里选「有这个标签，基本就是这个组」的那几个。选不出就留空。
   组号为 0 的照片不算任何组。
   用组号（A1、B2 这样）指代组，不要自己按图重新分组，也不要把照片挪到别的组。
   一个组号只能出现在一个组里，不能把一个组拆成两个。
   drop_groups 只列选中那种分法里你要去掉的组。

5. 指出哪几张图明显不属于这一堆，或者拍的主要是只在一两张里出现的别的东西。

6. 如果你觉得这一堆其实是两类不相干的东西（比如一半是饭、一半是书桌），
   two_kinds 填 true。

只用 JSON 回答，不要解释。回答格式见下面的例子：groups 的 from 和 drop_groups 填组号（A1、B2），
只有 not_in_pile 填图号。"""

ROUND2_PROMPT = """下面是一个人相册里的一个主题，以及这个主题的 15 张候选照片，每张来自不同的一天。
请逐张检查：
1. 属于这个主题吗？主题写的是一种场合（时段、地点）时，看图号后面括号里标的时间、地点来判断，不要只看画面猜；拍的是这个场合里的东西就算属于，不要求每张都拍到名字里的每一样东西。
2. 和另一张几乎一样吗？是的话写出是哪一张。
3. 太糊、太暗，或者看不清主体吗？
4. 是不是药、医院、体检单、身体、证件、病历这类敏感内容？

然后把这个主题和下面两份名单里的每个名字比一比：是同一种东西、很像，还是不同。
只列出「同一种」或「很像」的，没有就留空。

只用 JSON 回答，不要解释。"""


def _body(name):
    import texts
    return texts.render(name, example="@@EXAMPLE@@", input="@@INPUT@@").split("@@EXAMPLE@@")[0].strip()


def test_round1_prompt_body_matches_the_draft():
    assert _body("repeat_round1_user") == ROUND1_PROMPT


def test_round2_prompt_body_matches_the_draft():
    assert _body("repeat_round2_user") == ROUND2_PROMPT


def test_round2_accepts_pile_theme_without_noun_or_same_part(client, fake_model):
    fake_model.reply("r2", REPLY)
    meta = _meta2()
    meta["theme"] = {"name": "THEME_NAME", "noun": "", "same_part": ""}
    assert _post2(client, meta).status_code == 200
    text = _text(fake_model.calls[0])
    assert "THEME_NAME" in text and "短名词：无" in text and "是：无。" in text


def test_round2_still_requires_theme_name(client):
    meta = _meta2()
    meta["theme"] = {"name": "", "noun": "", "same_part": ""}
    assert _post2(client, meta).status_code == 400


def test_shorten_accepts_empty_same_part(client, fake_model):
    fake_model.reply("rs", '{"items": []}')
    r = client.post("/collections/rounds/shorten",
                    json={"items": [{"key": "pile", "name": "N" * 14, "same_part": ""}]})
    assert r.status_code == 200
    for bad in ({"key": "", "name": "n", "same_part": ""}, {"key": "k", "name": "", "same_part": ""}):
        assert client.post("/collections/rounds/shorten", json={"items": [bad]}).status_code == 400


def test_round1_log_keeps_only_validated_pile_fields(client, fake_model, tmp_log):
    fake_model.reply("r1", REPLY)
    meta = _meta1()
    meta["pile"]["extra_secret"] = "LEAKED_VALUE"
    _post1(client, meta)
    logged = tmp_log.read_text(encoding="utf-8")
    assert "photo_count" in logged and "LEAKED_VALUE" not in logged


def _tag_texts(call):
    return [p["text"] for p in _parts(call)[1:] if p["type"] == "text"]


def test_round2_renders_photo_time_and_place(client, fake_model):
    fake_model.reply("r2", REPLY)
    meta = _meta2(n=2)
    meta["photos"] = [{"n": 1, "daypart": "evening", "hour": 17, "at_place": True},
                      {"n": 2, "daypart": "night", "hour": 20, "at_place": False}]
    assert _post2(client, meta, files=_files(2)).status_code == 200
    assert _tag_texts(fake_model.calls[0]) == ["图 1（傍晚 17 点，在主题的地点）",
                                               "图 2（夜晚 20 点，不在主题的地点）"]


def test_round2_renders_each_fact_on_its_own(client, fake_model):
    fake_model.reply("r2", REPLY)
    meta = _meta2(n=3)
    meta["photos"] = [{"n": 1, "daypart": "morning"}, {"n": 2, "hour": 9}, {"n": 3, "at_place": True}]
    assert _post2(client, meta).status_code == 200
    assert _tag_texts(fake_model.calls[0]) == ["图 1（上午）", "图 2（9 点）", "图 3（在主题的地点）"]


def test_round2_without_facts_renders_plain_tags(client, fake_model):
    fake_model.reply("r2", REPLY)
    assert _post2(client, _meta2(n=1), files=_files(1)).status_code == 200
    assert _tag_texts(fake_model.calls[0]) == ["图 1"]


@pytest.mark.parametrize("bad", [{"daypart": "dawn"}, {"hour": 24}, {"hour": -1}, {"hour": True},
                                 {"hour": 7.5}, {"at_place": "yes"}, {"at_place": 1}])
def test_round2_rejects_bad_facts(client, bad):
    meta = _meta2(n=1)
    meta["photos"] = [dict({"n": 1}, **bad)]
    assert _post2(client, meta, files=_files(1)).status_code == 400

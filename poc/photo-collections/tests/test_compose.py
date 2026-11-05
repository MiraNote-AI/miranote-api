"""The pipeline: scripted model calls A, B and C, checks, retries, deadline,
memory assembly and the log line (handoff 5.2, AI doc 0 and 6, copy refine
handoff 2 and 5.7)."""

from __future__ import annotations

import base64
import json
import time

import pytest

import compose
import config
import llm
import texts


def _meta(**over):
    meta = {
        "group_key": "2026-10-18_8F3A",
        "date": "2026-10-18",
        "weekday": 7,
        "start_time": "14:03",
        "end_time": "17:40",
        "place": "达拉斯植物园",
        "raw_count": 23,
        "covers_all": False,
        "recent_question_types": ["sound", "order"],
        "bucket": "event",
        "photos": [
            {"id": "P1", "taken_at": "2026-10-18T14:03:00-05:00", "labels": ["plant"], "faces": 0,
             "favorite": True, "edited": False, "represents": 1},
            {"id": "P2", "taken_at": "2026-10-18T14:05:00-05:00", "labels": ["plant", "animal"], "faces": 0,
             "favorite": False, "edited": False, "represents": 3},
            {"id": "P3", "taken_at": "2026-10-18T16:30:00-05:00", "labels": ["event"], "faces": 3,
             "favorite": False, "edited": False, "represents": 1},
            {"id": "P4", "taken_at": "2026-10-18T17:13:00-05:00", "labels": ["event"], "faces": 0,
             "favorite": False, "edited": False, "represents": 2},
            {"id": "P5", "taken_at": "2026-10-18T17:14:00-05:00", "labels": ["event"], "faces": 0,
             "favorite": False, "edited": False, "represents": 1},
        ],
    }
    meta.update(over)
    return meta


IMAGES = {pid: b"\xff\xd8\xff\xe0fake-jpeg-" + pid.encode() for pid in ["P1", "P2", "P3", "P4", "P5"]}


def _img(pid):
    return "data:image/jpeg;base64," + base64.b64encode(IMAGES[pid]).decode()


def _photo(pid, subjects, description, people="0", face=False, scene="park_nature", keep=True,
           drop="", flags=None, text=None):
    return {
        "id": pid,
        "subjects": [{"name": n, "type": t} for n, t in subjects],
        "description": description,
        "text": text or [],
        "people": {"count": people, "face_prominent": face},
        "scene": scene,
        "flags": flags or [],
        "keep": keep,
        "drop_reason": drop,
    }


def _a_json(**over):
    raw = {
        "photos": [
            _photo("P1", [("花丛", "scenery")], "红色的花开满花坛，下午的阳光从左边照过来。"),
            _photo("P2", [("长椅", "object"), ("橘猫", "animal")], "一只橘猫睡在木长椅上，背景是草地。"),
            _photo("P3", [("舞台", "object"), ("灯光", "object")], "两个人站在舞台上，一束光落在台中央。",
                   people="2-4", scene="venue_event"),
            _photo("P4", [("舞台", "object")], "一束光落在台中央，四周很暗，背景是蓝色幕布。",
                   scene="venue_event"),
            _photo("P5", [("舞台", "object")], "一束光落在台中央。", keep=False, drop="duplicate",
                   scene="venue_event"),
        ],
        "selected_ids": ["P1", "P2", "P3", "P4"],
        "peak_id": "P4",
        "skip": False,
        "skip_reason": "",
    }
    raw.update(over)
    return json.dumps(raw, ensure_ascii=False)


Q1 = {"text": "台中央那束光，是几点亮起来的？", "type": "zoom", "why": "封面的光"}
Q2 = {"text": "那天看完演出，你去了哪儿？", "type": "route", "why": "演出之后"}
Q3 = {"text": "那只橘猫，后来醒了吗？", "type": "order", "why": "长椅上的猫"}


def _b_json(candidates=None, ranked=None, cannot_write=False):
    if candidates is None:
        candidates = [{"text": "台中央，落下一束光", "photo_ids": ["P4"]}]
    if ranked is None:
        ranked = [Q1]
    return json.dumps({"cannot_write": cannot_write, "candidates": candidates,
                       "question": {"ranked": ranked}}, ensure_ascii=False)


def _run(meta=None, deadline=None):
    return compose.run(meta or _meta(), dict(IMAGES), deadline or time.monotonic() + 300)


def _log(tmp_log):
    return json.loads(tmp_log.read_text(encoding="utf-8").splitlines()[-1])


def _texts(content):
    return [p["text"] for p in content if p["type"] == "text"]


def _images(content):
    return [p["image_url"]["url"] for p in content if p["type"] == "image_url"]


def test_happy_path_matches_handoff_example_shape(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    out = _run()
    assert out["status"] == "ok"
    m = out["memory"]
    assert set(m) == {"id", "mode", "date", "daypart", "bucket", "push", "story", "ending",
                      "facts", "generation"}
    assert m["id"] == "2026-10-18_8F3A"
    assert m["mode"] == "life_fragment"
    assert m["date"] == "2026-10-18"
    assert m["daypart"] == "evening"
    assert m["bucket"] == "活动演出"
    assert m["push"] == {"title": "10.18 · 达拉斯植物园 · 傍晚", "body": "台中央，落下一束光",
                         "coverPhotoId": "P4"}
    assert m["story"]["firstFrameLabel"] == "10.18 · 达拉斯植物园"
    assert m["story"]["photos"] == [
        {"id": "P4", "takenAt": "2026-10-18T17:13:00-05:00", "isPeak": True},
        {"id": "P1", "takenAt": "2026-10-18T14:03:00-05:00", "isPeak": False},
        {"id": "P2", "takenAt": "2026-10-18T14:05:00-05:00", "isPeak": False},
        {"id": "P3", "takenAt": "2026-10-18T16:30:00-05:00", "isPeak": False},
    ]
    assert m["ending"] == {
        "factLine": "10.18 · 达拉斯植物园",
        "detail": "台中央，落下一束光",
        "questions": [{"text": "台中央那束光，是几点亮起来的？", "type": "zoom", "source": "llm"}],
        "ctaLabel": "留住这个傍晚",
    }
    g = m["generation"]
    assert set(g) == {"bodySource", "questionSource", "copyVersion", "model", "costCny",
                      "latencyMs", "peakSwitched", "peakRewrite"}
    assert g["copyVersion"] == "copy_v2"
    assert (g["peakSwitched"], g["peakRewrite"]) == (False, False)
    assert (g["bodySource"], g["questionSource"], g["model"]) == ("llm", "llm", config.LLM_MODEL)
    assert isinstance(g["latencyMs"], int)
    assert [c["call"] for c in fake_model.calls] == ["a", "b", "c"]


def test_call_a_gets_images_and_labels_in_time_order(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    meta = _meta()
    meta["photos"] = list(reversed(meta["photos"]))
    _run(meta)
    a_call = fake_model.calls[0]
    assert a_call["timeout"] == pytest.approx(config.CALL_A_TIMEOUT, abs=1)
    content = a_call["messages"][1]["content"]
    assert _images(content) == [_img(pid) for pid in ["P1", "P2", "P3", "P4", "P5"]]
    labels = _texts(content)
    assert any(t.startswith("P1｜14:03｜") and "收藏" in t for t in labels)
    assert any(t.startswith("P2｜14:05｜") and "代表 3 张（连拍或近似重复合并）" in t for t in labels)


def test_call_b_sees_the_selected_photos_with_descriptions_and_cover(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    _run()
    b_call = fake_model.calls[1]
    assert b_call["timeout"] == pytest.approx(config.CALL_B_TIMEOUT, abs=1)
    content = b_call["messages"][1]["content"]
    assert _images(content) == [_img(pid) for pid in ["P1", "P2", "P3", "P4"]]
    lines = _texts(content)
    assert "P4｜17:13｜封面｜描述：一束光落在台中央，四周很暗，背景是蓝色幕布。｜人数：0" in lines
    assert "P3｜16:30｜描述：两个人站在舞台上，一束光落在台中央。｜人数：2-4" in lines
    assert "日期：2026-10-18（周日）；时段：傍晚；地点：达拉斯植物园" in lines[0]
    assert "舞台出现在 2 张里" in lines[0]
    task = lines[-1]
    assert "  order｜先后顺序｜那桌菜里，哪一样最先吃完？" in task
    assert "  people｜有人在场（照片里有人才能用）｜合照，是谁提议要拍的？" in task
    assert "尽量避开这些类型（前后几晚已经用过）：声音、先后顺序" in task


def test_unknown_recent_types_are_ignored_and_none_reads_none(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    _run(_meta(recent_question_types=["nope", "B1"]))
    assert "前后几晚已经用过）：无" in _texts(fake_model.calls[1]["messages"][1]["content"])[-1]


def test_call_c_sees_the_selected_photos_and_what_passed(fake_model, tmp_log):
    cands = [{"text": "台中央，落下一束光", "photo_ids": ["P4"]},
             {"text": "和朋友度过的温馨夜晚", "photo_ids": ["P3"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=cands, ranked=[Q1, Q2]))
    _run()
    c_call = fake_model.calls[2]
    assert c_call["call"] == "c"
    assert c_call["timeout"] == pytest.approx(config.CALL_C_TIMEOUT, abs=1)
    content = c_call["messages"][1]["content"]
    assert _images(content) == [_img(pid) for pid in ["P1", "P2", "P3", "P4"]]
    assert _texts(content)[:4] == ["P1", "P2", "P3", "P4｜封面"]
    task = _texts(content)[-1]
    assert "S1｜台中央，落下一束光" in task
    assert "温馨" not in task  # rejected by the lexicon before C
    assert "Q1｜台中央那束光，是几点亮起来的？" in task and "Q2｜那天看完演出，你去了哪儿？" in task


def test_daypart_from_peak_time_consistent_in_title_question_cta(fake_model, tmp_log):
    meta = _meta()
    meta["photos"][3]["taken_at"] = "2026-10-18T15:13:00-05:00"
    fake_model.reply("a", _a_json()).reply("b", _b_json(ranked=[]))
    m = _run(meta)["memory"]
    assert m["daypart"] == "afternoon"
    assert m["push"]["title"].endswith("· 下午")
    assert m["ending"]["questions"][0] == {"text": "那个下午的三个关键词：＿＿、＿＿、＿＿",
                                           "type": "one_word", "source": "fallback_B1"}
    assert m["ending"]["ctaLabel"] == "留住这个下午"


def test_daypart_uses_photo_offset_not_server_tz(fake_model, tmp_log, monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    if hasattr(time, "tzset"):
        time.tzset()
    meta = _meta()
    meta["photos"][3]["taken_at"] = "2026-10-19T09:13:00+09:00"  # 00:13 UTC
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    try:
        assert _run(meta)["memory"]["daypart"] == "morning"
    finally:
        monkeypatch.undo()
        if hasattr(time, "tzset"):
            time.tzset()


@pytest.mark.parametrize("hour,part", [(4, "night"), (5, "morning"), (10, "morning"), (11, "noon"),
                                       (14, "afternoon"), (17, "evening"), (18, "evening"),
                                       (19, "night"), (23, "night")])
def test_daypart_boundaries(hour, part):
    assert compose.daypart_of(hour) == part


def test_date_label():
    assert compose.date_label("2026-10-18") == "10.18"
    assert compose.date_label("2026-09-05") == "9.05"


def test_no_place_omits_place(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    m = _run(_meta(place=None))["memory"]
    assert m["push"]["title"] == "10.18 · 傍晚"
    assert m["ending"]["factLine"] == "10.18"
    assert m["story"]["firstFrameLabel"] == "10.18"


def test_empty_place_is_missing_and_braces_safe(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    assert _run(_meta(place="  "))["memory"]["push"]["title"] == "10.18 · 傍晚"
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    assert _run(_meta(place="{x} 公园 {0}"))["memory"]["push"]["title"] == "10.18 · {x} 公园 {0} · 傍晚"


def test_f0_empties_detail_and_names_the_cover_weekday(fake_model, tmp_log):
    # First and last selected share a subject (no T2), covers_all false (no T1),
    # and the model cannot write: F0, by the cover's weekday and daypart.
    a = json.loads(_a_json())
    a["photos"][0]["subjects"] = [{"name": "舞台", "type": "object"}]
    fake_model.reply("a", json.dumps(a, ensure_ascii=False)).reply(
        "b", _b_json(candidates=[], cannot_write=True)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))  # cover rewrite
    m = _run()["memory"]
    assert m["push"]["body"] == "23 张照片，一个周日的傍晚"
    assert m["ending"]["detail"] == ""
    assert m["generation"]["bodySource"] == "fallback_F0"


def test_t1_new_shape(fake_model, tmp_log):
    a = json.loads(_a_json())
    a["photos"][0]["subjects"] = [{"name": "舞台", "type": "object"}]  # no T2 either
    fake_model.reply("a", json.dumps(a, ensure_ascii=False)).reply(
        "b", _b_json(candidates=[], cannot_write=True)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run(_meta(covers_all=True))["memory"]
    # P1, P3, P4, P5 show the stage; P4 represents 2: 1 + 1 + 2 + 1.
    assert m["push"]["body"] == "5 张舞台，一个周日的傍晚"
    assert m["generation"]["bodySource"] == "template_T1"


def test_a_bad_json_twice_rejects_check_failed_and_skips_b(fake_model, tmp_log):
    fake_model.reply("a", "不是 JSON").reply("a", '{"photos": []}')
    out = _run()
    assert out == {"status": "rejected", "reason": "check_failed", "message": "AI 的结果没通过检查"}
    assert fake_model.count("a") == 2 and fake_model.count("b") == 0


def test_a_timeout_twice_is_timeout_error(fake_model, tmp_log):
    fake_model.fail("a", llm.ModelTimeout("slow")).fail("a", llm.ModelTimeout("slow"))
    assert _run() == {"status": "error", "reason": "timeout", "message": "AI 超时"}


def test_a_model_error_twice_is_model_error(fake_model, tmp_log):
    fake_model.fail("a", llm.ModelError("500")).fail("a", llm.ModelError("500"))
    assert _run()["reason"] == "model_error"


def test_a_retry_then_ok(fake_model, tmp_log):
    fake_model.fail("a", llm.ModelTimeout("slow")).reply("a", _a_json()).reply("b", _b_json())
    assert _run()["status"] == "ok"
    assert fake_model.count("a") == 2


def test_b_fails_twice_uses_templates_and_fallbacks(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", "坏的").fail("b", llm.ModelError("500")).fail(
        "b", llm.ModelError("500"))  # the cover rewrite fails too
    m = _run()["memory"]
    assert m["push"]["body"] == "从花丛，到舞台"
    assert m["generation"]["bodySource"] == "template_T2"
    assert m["ending"]["questions"] == [
        {"text": "那个傍晚的三个关键词：＿＿、＿＿、＿＿", "type": "one_word", "source": "fallback_B1"},
        {"text": "满分 10 分，那个傍晚几分？", "type": "one_word", "source": "fallback_B2"},
        {"text": "如果给那个傍晚配一首歌，会是哪首？", "type": "imagine", "source": "fallback_B3"},
    ]
    assert m["generation"]["questionSource"] == "fallback_B1"
    assert fake_model.count("c") == 0  # nothing model-written to check


def test_b_wrong_shape_is_retried(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", '{"candidates": "x"}').reply("b", _b_json())
    m = _run()["memory"]
    assert m["generation"]["bodySource"] == "llm"
    assert fake_model.count("b") == 2


def test_deadline_passed_is_timeout(fake_model, tmp_log):
    out = _run(deadline=time.monotonic() - 1)
    assert out["status"] == "error" and out["reason"] == "timeout"
    assert fake_model.calls == []


def test_call_timeout_shrinks_to_the_deadline(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    _run(deadline=time.monotonic() + 50)
    assert fake_model.calls[0]["timeout"] <= 50


def test_sensitive_rejects_without_calling_b(fake_model, tmp_log):
    a = json.loads(_a_json())
    a["photos"][2]["flags"] = ["hospital"]
    fake_model.reply("a", json.dumps(a, ensure_ascii=False))
    out = _run()
    assert (out["status"], out["reason"]) == ("rejected", "sensitive")
    assert fake_model.count("b") == 0


def test_too_few_selected_rejects(fake_model, tmp_log):
    fake_model.reply("a", _a_json(selected_ids=["P1", "P4"]))
    assert _run() == {"status": "rejected", "reason": "too_few_selected", "message": "精选不到 3 张"}


BUCKET_NAMES = {"food": "美食饮品", "scenery": "自然风光", "animal": "宠物与动物",
                "city": "城市街头", "indoor": "室内居家", "event": "活动演出", "other": "其他"}


@pytest.mark.parametrize("bucket_id", list(BUCKET_NAMES))
def test_memory_bucket_is_chinese_name_of_meta_bucket(fake_model, tmp_log, bucket_id):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    out = _run(_meta(bucket=bucket_id))
    assert out["status"] == "ok"
    assert out["memory"]["bucket"] == BUCKET_NAMES[bucket_id]
    assert texts.json("names")["buckets"] == BUCKET_NAMES


def test_labels_do_not_decide_the_bucket(fake_model, tmp_log):
    # The phone's bucket wins whatever the labels say.
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    assert _run(_meta(bucket="food"))["memory"]["bucket"] == "美食饮品"


def test_facts_only_selected(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    facts = _run()["memory"]["facts"]
    assert set(facts) == {"P1", "P2", "P3", "P4"}
    assert facts["P3"] == "主体：舞台、灯光；描述：两个人站在舞台上，一束光落在台中央。；人数：2-4"
    assert facts["P2"] == "主体：长椅、橘猫；描述：一只橘猫睡在木长椅上，背景是草地。；人数：0"


def test_log_line_has_no_image_bytes_or_key(fake_model, tmp_log, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-very-secret-key")
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=[
        {"text": "和朋友度过的温馨午后", "photo_ids": ["P1"]},
        {"text": "台中央，落下一束光", "photo_ids": ["P4"]},
    ]))
    _run()
    lines = tmp_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    raw = lines[0]
    assert "sk-very-secret-key" not in raw
    for data in IMAGES.values():
        assert base64.b64encode(data).decode() not in raw
    assert "fake-jpeg" not in raw
    entry = json.loads(raw)
    assert entry["group_key"] == "2026-10-18_8F3A"
    assert entry["bodySource"] == "llm" and entry["questionSource"] == "llm"
    assert ["和朋友度过的温馨午后", "lexicon:温馨"] in entry["body"]["rejected"]
    assert entry["body"]["chosen"] == "台中央，落下一束光"
    assert entry["question"]["final"] == [{"text": Q1["text"], "type": "zoom", "source": "llm"}]
    assert entry["question"]["ranked"] == [Q1]
    assert entry["c"]["status"] == "ok"
    assert entry["c"]["sentences"]["S1"]["text"] == "台中央，落下一束光"
    assert [c["call"] for c in entry["calls"]] == ["a", "b", "c"]
    assert entry["calls"][0]["raw"].startswith("{")
    assert {"latency_ms", "prompt_tokens", "completion_tokens", "cost_cny"} <= set(entry["calls"][0])


def test_rejected_job_is_logged_too(fake_model, tmp_log):
    fake_model.reply("a", "x").reply("a", "y")
    _run()
    entry = _log(tmp_log)
    assert entry["status"] == "rejected" and entry["reason"] == "check_failed"
    assert len(entry["calls"]) == 2 and entry["calls"][0]["error"]


def test_cost_sums_retries(fake_model, tmp_log):
    fake_model.reply("a", "x", 1000, 500).reply("a", _a_json(), 1000, 500).reply("b", _b_json(), 1000, 500)
    cost = _run()["memory"]["generation"]["costCny"]
    assert cost == round(4 * config.cost_cny(1000, 500), 4)  # A twice, B, C


# ---------- call C: checked against the photos (copy refine handoff 2.4) ----------

def test_c_drops_ungrounded_sentences_and_bad_questions(fake_model, tmp_log):
    cands = [{"text": "台中央，落下一束光", "photo_ids": ["P4"]},
             {"text": "台上的灯，一盏接一盏亮了", "photo_ids": ["P4", "P3"]}]
    fake_model.c_ungrounded = {"台中央，落下一束光", Q2["text"]}
    fake_model.c_unsafe = {Q1["text"]}
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=cands, ranked=[Q1, Q2, Q3]))
    m = _run()["memory"]
    assert m["push"]["body"] == "台上的灯，一盏接一盏亮了"
    assert m["ending"]["questions"] == [{"text": Q3["text"], "type": "order", "source": "llm"}]
    entry = _log(tmp_log)
    assert ["台中央，落下一束光", "c_ungrounded:"] in entry["body"]["rejected"]
    assert [Q1["text"], "c_unsafe:"] in entry["question"]["rejected"]
    assert [Q2["text"], "c_ungrounded:"] in entry["question"]["rejected"]


def test_up_to_three_questions_in_model_order(fake_model, tmp_log):
    q4 = {"text": "那天的晚饭，吃的是什么？", "type": "zoom", "why": "x"}
    fake_model.reply("a", _a_json()).reply("b", _b_json(ranked=[Q2, Q1, Q3, q4]))
    qs = _run()["memory"]["ending"]["questions"]
    assert [q["text"] for q in qs] == [Q2["text"], Q1["text"], Q3["text"]]
    assert [q["type"] for q in qs] == ["route", "zoom", "order"]


def test_c_fails_twice_uses_templates_and_fallbacks_only(fake_model, tmp_log):
    # Handoff 2.4 / R5: nothing unchecked is shown, and no cover rewrite.
    fake_model.reply("a", _a_json()).reply("b", _b_json(ranked=[Q1, Q2])).fail(
        "c", llm.ModelTimeout("slow")).fail("c", llm.ModelTimeout("slow"))
    m = _run()["memory"]
    assert m["generation"]["bodySource"] == "template_T2"
    assert m["push"]["body"] == "从花丛，到舞台"
    assert [q["source"] for q in m["ending"]["questions"]] == [
        "fallback_B1", "fallback_B2", "fallback_B3"]
    assert m["generation"]["peakRewrite"] is False
    assert fake_model.count("b") == 1 and fake_model.count("c") == 2
    assert _log(tmp_log)["c"] == {"failure": "timeout", "status": "failed"}


def test_c_bad_reply_is_retried(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json()).reply("c", '{"sentences": []}')
    m = _run()["memory"]  # the second call C is the automatic approval
    assert m["generation"]["bodySource"] == "llm"
    assert fake_model.count("c") == 2


def test_c_decides_whether_a_sentence_reaches_the_cover(fake_model, tmp_log):
    # photo_ids leave the cover out, but C says the sentence shows it.
    cands = [{"text": "蓝色幕布前，一束光", "photo_ids": ["P3"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=cands))
    m = _run()["memory"]
    assert m["push"]["body"] == "蓝色幕布前，一束光"
    assert m["push"]["coverPhotoId"] == "P4"
    assert m["generation"]["peakRewrite"] is False


# ---------- the body describes the cover (user decision 2026-10-31) ----------

NON_PEAK = [{"text": "长椅上，猫睡着", "photo_ids": ["P2"]}]
ON_PEAK = [{"text": "台中央，落下一束光", "photo_ids": ["P4"]}]


def _cover_is(*texts_):
    return lambda text: text in texts_


def test_peak_rewrite_used_when_first_round_has_none(fake_model, tmp_log):
    fake_model.c_cover = _cover_is("台中央，落下一束光")
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=NON_PEAK)).reply(
        "b", _b_json(candidates=ON_PEAK, ranked=[]))
    m = _run()["memory"]
    assert m["push"]["body"] == "台中央，落下一束光"
    assert m["push"]["coverPhotoId"] == "P4"
    assert (m["generation"]["peakRewrite"], m["generation"]["peakSwitched"]) == (True, False)
    assert [c["call"] for c in fake_model.calls] == ["a", "b", "c", "b", "c"]
    rewrite = fake_model.calls[3]["messages"][1]["content"]
    assert _images(rewrite) == [_img("P4")]
    assert "P4｜17:13｜封面｜描述：一束光落在台中央，四周很暗，背景是蓝色幕布。｜人数：0" in _texts(rewrite)
    c_rewrite = fake_model.calls[4]["messages"][1]["content"]
    assert _images(c_rewrite) == [_img("P4")] and _texts(c_rewrite)[0] == "P4｜封面"
    entry = _log(tmp_log)
    assert (entry["peakRewrite"], entry["peakSwitched"]) == (True, False)
    assert entry["peak_rewrite"]["c"]["status"] == "ok"


def test_rewrite_is_shown_the_cover_sentences_cut_for_length(fake_model, tmp_log):
    long_cover = "台中央落下一束很亮很亮的光，四周的幕布都暗了下去"  # 24
    long_other = "长椅上那只橘猫睡得很沉很沉，一下午都没有醒过来呢"
    cands = [{"text": long_cover, "photo_ids": ["P4"]}, {"text": long_other, "photo_ids": ["P2"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=cands)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True)).reply(  # shorten round
        "b", _b_json(candidates=ON_PEAK, ranked=[]))
    m = _run()["memory"]
    assert m["push"]["body"] == "台中央，落下一束光"
    task = _texts(fake_model.calls[4]["messages"][1]["content"])[-1]
    assert "- 台中央落下一束很亮很亮的光，四周的幕布都暗了下去（24 个字）" in task
    assert long_other not in task


def test_shorten_round_rescues_sentences_cut_for_length(fake_model, tmp_log):
    # User decision 2026-11-05: cut only for length -> one text-only call
    # to shorten them, then the usual checks and call C.
    long_cover = "台中央落下一束很亮很亮的光，四周的幕布都暗了下去"  # 24
    cands = [{"text": long_cover, "photo_ids": ["P4"]}]
    short = [{"text": "台中央落下一束光，幕布暗了", "photo_ids": ["P9"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=cands)).reply(
        "b", _b_json(candidates=short, ranked=[]))
    m = _run()["memory"]
    assert m["push"]["body"] == "台中央落下一束光，幕布暗了"
    assert m["push"]["coverPhotoId"] == "P4"  # the source's photos, not the reply's
    assert [c["call"] for c in fake_model.calls] == ["a", "b", "c", "b", "c"]
    shorten = fake_model.calls[3]["messages"][1]["content"]
    assert isinstance(shorten, str)  # text only, no photos
    assert "1. 台中央落下一束很亮很亮的光，四周的幕布都暗了下去（24 个字）｜photo_ids：P4" in shorten
    entry = _log(tmp_log)
    assert entry["shortened"] is True and entry["peakRewrite"] is False
    assert entry["shorten"]["asked"] == [long_cover]


def test_no_shorten_round_without_long_sentences(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    _run()
    assert fake_model.count("b") == 1 and _log(tmp_log)["shortened"] is False


def test_shrink_caps_the_long_side_and_keeps_what_it_cannot_read():
    import io
    from PIL import Image

    def png(w, h):
        out = io.BytesIO()
        Image.new("RGB", (w, h), (200, 120, 80)).save(out, "PNG")
        return out.getvalue()

    big = compose.shrink(png(1000, 800), 512)
    assert Image.open(io.BytesIO(big)).size == (512, 410)
    assert compose.image_mime(big) == "image/jpeg"
    small = png(400, 300)
    assert compose.shrink(small, 512) == small
    assert compose.shrink(b"not an image", 512) == b"not an image"
    assert config.SMALL_IMAGE_PX == 512


def test_rewrite_without_long_cover_sentences_says_none(fake_model, tmp_log):
    fake_model.c_cover = _cover_is("台中央，落下一束光")
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=NON_PEAK)).reply(
        "b", _b_json(candidates=ON_PEAK, ranked=[]))
    _run()
    task = _texts(fake_model.calls[3]["messages"][1]["content"])[-1]
    assert "改短，只留最要紧的画面）：\n无\n" in task


def test_rewrite_failing_call_c_is_discarded(fake_model, tmp_log):
    fake_model.c_cover = _cover_is("台中央，落下一束光")
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=NON_PEAK)).reply(
        "b", _b_json(candidates=ON_PEAK, ranked=[]))
    fake_model.respond("c", lambda m: fake_model.auto_c(m).text)  # first round
    fake_model.fail("c", llm.ModelError("500")).fail("c", llm.ModelError("500"))
    m = _run()["memory"]
    assert m["push"]["body"] == "长椅上，猫睡着"  # the checked first-round sentence
    assert m["push"]["coverPhotoId"] == "P2"
    assert _log(tmp_log)["peak_rewrite"]["c"]["status"] == "failed"


def test_cover_switches_to_body_photo_when_rewrite_fails(fake_model, tmp_log):
    fake_model.c_cover = _cover_is()
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=NON_PEAK)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["push"]["body"] == "长椅上，猫睡着"
    assert m["push"]["coverPhotoId"] == "P2"
    assert [(p["id"], p["isPeak"]) for p in m["story"]["photos"]] == [
        ("P2", True), ("P1", False), ("P3", False), ("P4", False)]
    assert (m["generation"]["peakRewrite"], m["generation"]["peakSwitched"]) == (True, True)


def test_daypart_follows_switched_peak(fake_model, tmp_log):
    fake_model.c_cover = _cover_is()
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=NON_PEAK, ranked=[])).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]  # old peak P4 at 17:13 (evening); P2 at 14:05
    assert m["daypart"] == "afternoon"
    assert m["push"]["title"] == "10.18 · 达拉斯植物园 · 下午"
    assert m["ending"]["ctaLabel"] == "留住这个下午"
    assert m["ending"]["questions"][1]["text"] == "满分 10 分，那个下午几分？"


def test_template_winner_keeps_peak(fake_model, tmp_log):
    bad = [{"text": "和朋友度过的温馨午后", "photo_ids": ["P2"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=bad)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["generation"]["bodySource"] == "template_T2"
    assert m["push"]["coverPhotoId"] == "P4" and m["daypart"] == "evening"
    assert (m["generation"]["peakRewrite"], m["generation"]["peakSwitched"]) == (True, False)


def test_rewrite_counts_cost_and_respects_deadline(fake_model, tmp_log, monkeypatch):
    # The fake model ignores the timeout it is given, so the sub-second
    # deadlines here need no margin (the margin is tested at real scale in
    # test_deadline_budget.py).
    monkeypatch.setattr(compose, "CALL_MARGIN_SECONDS", 0.0)
    fake_model.c_cover = _cover_is("台中央，落下一束光")
    fake_model.reply("a", _a_json(), 1000, 500).reply("b", _b_json(candidates=NON_PEAK), 1000, 500).reply(
        "b", _b_json(candidates=ON_PEAK, ranked=[]), 1000, 500)
    m = _run(deadline=time.monotonic() + 20)["memory"]
    assert m["generation"]["costCny"] == round(5 * config.cost_cny(1000, 500), 4)
    assert fake_model.calls[3]["timeout"] <= 20
    # Call B comes back after the deadline: the job answers a timeout
    # without call C (and no cover rewrite).
    fake_model.reply("a", _a_json())

    def slow_b(messages):
        time.sleep(0.4)
        return _b_json(candidates=NON_PEAK)

    fake_model.respond("b", slow_b)
    out = _run(deadline=time.monotonic() + 0.3)
    assert (out["status"], out["reason"]) == ("error", "timeout")
    assert fake_model.count("b") == 3 and fake_model.count("c") == 2  # none for the second job


def test_time_limits():
    # Call A 180 s and the job 300 s are a user decision (2026-11-01); jobs
    # are kept an hour (phase 2). Calls B and C: see config (handoff 2.2, 2.4).
    assert config.CALL_A_TIMEOUT == 180
    assert (config.CALL_B_TIMEOUT, config.CALL_C_TIMEOUT) == (60, 60)
    assert config.JOB_CAP_SECONDS == 300
    assert config.JOB_TTL_SECONDS == 3600
    assert compose.ATTEMPTS == 2


def test_unusable_photo_ids_fall_back_to_survivors_not_f0(fake_model, tmp_log):
    # The winning model sentence cites no usable photo (photo_ids empty, C
    # says not the cover), the cover rewrite cannot write, a valid T2 exists.
    fake_model.c_cover = _cover_is()
    no_ids = [{"text": "长椅上，猫睡着", "photo_ids": []}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=no_ids)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["generation"]["bodySource"] == "template_T2"
    assert m["push"]["body"] == "从花丛，到舞台"
    assert m["push"]["coverPhotoId"] == "P4"
    assert m["generation"]["peakSwitched"] is False


def test_unusable_winner_gives_way_to_next_model_sentence(fake_model, tmp_log):
    # The first (AI rank 1: +2) would win, but P9 is not a photo of this group.
    fake_model.c_cover = _cover_is()
    cands = [{"text": "长椅上，一只猫睡着", "photo_ids": ["P9"]},
             {"text": "红色的花开满花坛", "photo_ids": ["P1"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=cands)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["push"]["body"] == "红色的花开满花坛"
    assert m["push"]["coverPhotoId"] == "P1"
    assert m["generation"]["peakSwitched"] is True


def test_unexpected_exception_still_logs_and_answers_model_error(fake_model, tmp_log, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("bug in the pick")

    monkeypatch.setattr(compose.body, "pick", boom)
    fake_model.reply("a", _a_json(), 1000, 500).reply("b", _b_json(), 1000, 500)
    assert _run() == {"status": "error", "reason": "model_error", "message": "AI 出错了"}
    entry = _log(tmp_log)
    assert (entry["status"], entry["reason"]) == ("error", "model_error")
    assert "RuntimeError" in entry["exception"]
    assert entry["calls"][0]["raw"].startswith("{")
    assert entry["costCny"] == pytest.approx(3 * config.cost_cny(1000, 500))


def test_result_after_deadline_is_a_logged_timeout(fake_model, tmp_log, monkeypatch):
    monkeypatch.setattr(compose, "CALL_MARGIN_SECONDS", 0.0)  # see above
    def slow_c(messages):
        time.sleep(0.4)
        return fake_model.auto_c(messages).text

    fake_model.reply("a", _a_json()).reply("b", _b_json()).respond("c", slow_c)
    out = _run(deadline=time.monotonic() + 0.3)
    assert out == {"status": "error", "reason": "timeout", "message": "AI 超时"}
    entry = _log(tmp_log)
    assert (entry["status"], entry["reason"], entry["lateResult"], entry["lateStatus"]) == (
        "error", "timeout", True, "ok")


def test_one_unsure_verdict_keeps_the_rest_of_call_c(fake_model, tmp_log):
    # Full check 2026-11-05: an "unsure" safety answer on one question threw
    # every checked sentence away. Now only that question is dropped.
    reply = json.dumps({"sentences": [{"id": "S1", "grounded": True, "mentions_cover": True}],
                        "questions": [{"id": "Q1", "grounded": True, "safe": "unsure"}]},
                       ensure_ascii=False)
    fake_model.reply("a", _a_json()).reply("b", _b_json()).reply("c", reply).reply("c", reply)
    m = _run()["memory"]
    assert (m["push"]["body"], m["generation"]["bodySource"]) == ("台中央，落下一束光", "llm")
    assert m["ending"]["questions"][0]["source"] == "fallback_B1"

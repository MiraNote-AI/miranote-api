"""The pipeline: two scripted model calls, checks, retries, deadline,
memory assembly and the log line (handoff 5.2, AI doc 0 and 6)."""

from __future__ import annotations

import base64
import json
import time

import pytest

import compose
import config
import llm


def _meta(**over):
    meta = {
        "group_key": "2026-10-18_8F3A",
        "date": "2026-10-18",
        "weekday": 6,
        "start_time": "14:03",
        "end_time": "17:40",
        "place": "达拉斯植物园",
        "raw_count": 23,
        "covers_all": False,
        "recent_question_ids": ["B4", "A2"],
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


def _photo(pid, subjects, details, people="0", face=False, scene="park_nature", keep=True,
           drop="", flags=None, text=None):
    return {
        "id": pid,
        "subjects": [{"name": n, "type": t} for n, t in subjects],
        "details": details,
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
            _photo("P1", [("花丛", "scenery")], ["红色的花开满花坛"]),
            _photo("P2", [("长椅", "object"), ("橘猫", "animal")], ["猫睡在长椅上"]),
            _photo("P3", [("舞台", "object"), ("灯光", "object")], ["一束光落在台中央"], people="2-4",
                   scene="venue_event"),
            _photo("P4", [("舞台", "object")], ["一束光落在台中央"], scene="venue_event"),
            _photo("P5", [("舞台", "object")], ["一束光落在台中央"], keep=False, drop="duplicate",
                   scene="venue_event"),
        ],
        "selected_ids": ["P1", "P2", "P3", "P4"],
        "peak_id": "P4",
        "skip": False,
        "skip_reason": "",
    }
    raw.update(over)
    return json.dumps(raw, ensure_ascii=False)


def _b_json(candidates=None, ranked=None, cannot_write=False):
    if candidates is None:
        candidates = [{"text": "台中央，落下一束光", "photo_ids": ["P4"], "uses": ["一束光", "台中央"]}]
    if ranked is None:
        ranked = [{"id": "A1", "fill": "一束光落在台中央", "why": "峰值照的画面"}]
    return json.dumps({"cannot_write": cannot_write, "candidates": candidates,
                       "question": {"ranked": ranked}}, ensure_ascii=False)


def _run(meta=None, deadline=None):
    return compose.run(meta or _meta(), dict(IMAGES), deadline or time.monotonic() + 300)


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
    assert m["bucket"] == "活动与演出"
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
        "question": "一束光落在台中央的时候，旁边在发生什么？",
        "questionId": "A1",
        "ctaLabel": "留住这个傍晚",
    }
    g = m["generation"]
    assert set(g) == {"bodySource", "questionSource", "model", "costCny", "latencyMs",
                      "peakSwitched", "peakRewrite"}
    assert (g["peakSwitched"], g["peakRewrite"]) == (False, False)
    assert (g["bodySource"], g["questionSource"], g["model"]) == ("llm", "llm", config.LLM_MODEL)
    assert isinstance(g["latencyMs"], int)


def test_call_a_gets_images_and_labels_in_time_order(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    meta = _meta()
    meta["photos"] = list(reversed(meta["photos"]))
    _run(meta)
    a_call = fake_model.calls[0]
    assert a_call["timeout"] == pytest.approx(config.CALL_A_TIMEOUT, abs=1)
    content = a_call["messages"][1]["content"]
    images = [p["image_url"]["url"] for p in content if p["type"] == "image_url"]
    assert images == ["data:image/jpeg;base64," + base64.b64encode(IMAGES[pid]).decode()
                      for pid in ["P1", "P2", "P3", "P4", "P5"]]
    labels = [p["text"] for p in content if p["type"] == "text"]
    assert any(t.startswith("P1｜14:03｜") and "收藏" in t for t in labels)
    assert any(t.startswith("P2｜14:05｜") and "代表 3 张（连拍或近似重复合并）" in t for t in labels)
    b_call = fake_model.calls[1]
    assert b_call["timeout"] == pytest.approx(config.CALL_B_TIMEOUT, abs=1)
    b_user = b_call["messages"][1]["content"]
    assert "A1｜{画面}的时候，旁边在发生什么？｜菜单：一束光落在台中央" in b_user
    assert "B4｜" not in b_user and "A2｜" not in b_user  # recent ids not offered
    assert "舞台出现在 2 张里" in b_user


def test_daypart_from_peak_time_consistent_in_title_question_cta(fake_model, tmp_log):
    meta = _meta()
    meta["photos"][3]["taken_at"] = "2026-10-18T15:13:00-05:00"
    fake_model.reply("a", _a_json()).reply("b", _b_json(ranked=[{"id": "B1", "fill": "", "why": "x"}]))
    m = _run(meta)["memory"]
    assert m["daypart"] == "afternoon"
    assert m["push"]["title"].endswith("· 下午")
    assert m["ending"]["question"] == "那个下午的三个关键词：＿＿、＿＿、＿＿"
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


def test_f0_empties_detail(fake_model, tmp_log):
    # First and last selected share a subject (no T2), covers_all false (no T1),
    # and the model cannot write: F0.
    a = json.loads(_a_json())
    a["photos"][0]["subjects"] = [{"name": "舞台", "type": "object"}]
    fake_model.reply("a", json.dumps(a, ensure_ascii=False)).reply(
        "b", _b_json(candidates=[], cannot_write=True)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))  # cover rewrite
    m = _run()["memory"]
    assert m["push"]["body"] == "那天你拍了 23 张照片"
    assert m["ending"]["detail"] == ""
    assert m["generation"]["bodySource"] == "fallback_F0"


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


def test_b_fails_twice_uses_templates_and_b1(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", "坏的").fail("b", llm.ModelError("500")).fail(
        "b", llm.ModelError("500"))  # the cover rewrite fails too
    m = _run()["memory"]
    assert m["push"]["body"] == "从花丛拍到了舞台"
    assert m["generation"]["bodySource"] == "template_T2"
    assert m["ending"]["questionId"] == "B1"
    assert m["generation"]["questionSource"] == "fallback_B1"
    assert m["ending"]["question"] == "那个傍晚的三个关键词：＿＿、＿＿、＿＿"


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


def test_bucket_majority_and_tie_order():
    def photos(*labels):
        return [{"id": "P{}".format(i), "labels": list(ls)} for i, ls in enumerate(labels)]

    assert compose.bucket(photos(["food_drink"], ["sky"], ["sky"])) == "天空与日落"
    # tie between food (1) and city (1): food comes first in the list order
    assert compose.bucket(photos(["street"], ["food_drink"])) == "美食饮品"
    assert compose.bucket(photos(["plant"], ["water"], ["people"])) == "自然与植物"
    assert compose.bucket(photos(["document"], ["document"], ["sky"])) == "其他"
    assert compose.bucket(photos([], [])) == "其他"
    assert compose.bucket(photos(["interior", "tableware"], ["event"])) == "室内与居家"


def test_facts_only_selected(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    facts = _run()["memory"]["facts"]
    assert set(facts) == {"P1", "P2", "P3", "P4"}
    assert facts["P3"] == "主体：舞台、灯光；细节：一束光落在台中央；人数：2-4"
    assert facts["P2"] == "主体：长椅、橘猫；细节：猫睡在长椅上；人数：0"


def test_log_line_has_no_image_bytes_or_key(fake_model, tmp_log, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-very-secret-key")
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=[
        {"text": "和朋友度过的温馨午后", "photo_ids": ["P1"], "uses": ["朋友"]},
        {"text": "台中央，落下一束光", "photo_ids": ["P4"], "uses": ["一束光"]},
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
    assert entry["question"]["chosen_id"] == "A1"
    assert [c["call"] for c in entry["calls"]] == ["a", "b"]
    assert entry["calls"][0]["raw"].startswith("{")
    assert {"latency_ms", "prompt_tokens", "completion_tokens", "cost_cny"} <= set(entry["calls"][0])


def test_rejected_job_is_logged_too(fake_model, tmp_log):
    fake_model.reply("a", "x").reply("a", "y")
    _run()
    entry = json.loads(tmp_log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["status"] == "rejected" and entry["reason"] == "check_failed"
    assert len(entry["calls"]) == 2 and entry["calls"][0]["error"]


def test_cost_sums_retries(fake_model, tmp_log):
    fake_model.reply("a", "x", 1000, 500).reply("a", _a_json(), 1000, 500).reply("b", _b_json(), 1000, 500)
    cost = _run()["memory"]["generation"]["costCny"]
    assert cost == round(3 * config.cost_cny(1000, 500), 4)


# ---------- the body describes the cover (user decision 2026-10-31) ----------

NON_PEAK = [{"text": "长椅上，猫睡着", "photo_ids": ["P2"], "uses": ["长椅"]}]
ON_PEAK = [{"text": "台中央，落下一束光", "photo_ids": ["P4"], "uses": ["一束光"]}]


def test_peak_rewrite_used_when_first_round_has_none(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=NON_PEAK)).reply(
        "b", _b_json(candidates=ON_PEAK, ranked=[]))
    m = _run()["memory"]
    assert m["push"]["body"] == "台中央，落下一束光"
    assert m["push"]["coverPhotoId"] == "P4"
    assert (m["generation"]["peakRewrite"], m["generation"]["peakSwitched"]) == (True, False)
    assert fake_model.count("b") == 2
    rewrite_user = fake_model.calls[2]["messages"][1]["content"]
    assert "P4" in rewrite_user and "峰值照" in rewrite_user
    entry = json.loads(tmp_log.read_text(encoding="utf-8").splitlines()[-1])
    assert (entry["peakRewrite"], entry["peakSwitched"]) == (True, False)


def test_cover_switches_to_body_photo_when_rewrite_fails(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=NON_PEAK)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["push"]["body"] == "长椅上，猫睡着"
    assert m["push"]["coverPhotoId"] == "P2"
    assert [(p["id"], p["isPeak"]) for p in m["story"]["photos"]] == [
        ("P2", True), ("P1", False), ("P3", False), ("P4", False)]
    assert (m["generation"]["peakRewrite"], m["generation"]["peakSwitched"]) == (True, True)


def test_daypart_follows_switched_peak(fake_model, tmp_log):
    fake_model.reply("a", _a_json()).reply(
        "b", _b_json(candidates=NON_PEAK, ranked=[{"id": "B2", "fill": "", "why": "x"}])).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]  # old peak P4 at 17:13 (evening); P2 at 14:05
    assert m["daypart"] == "afternoon"
    assert m["push"]["title"] == "10.18 · 达拉斯植物园 · 下午"
    assert m["ending"]["ctaLabel"] == "留住这个下午"
    assert m["ending"]["question"] == "满分 10 分，那个下午几分？"


def test_template_winner_keeps_peak(fake_model, tmp_log):
    bad = [{"text": "和朋友度过的温馨午后", "photo_ids": ["P2"], "uses": ["朋友"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=bad)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["generation"]["bodySource"] == "template_T2"
    assert m["push"]["coverPhotoId"] == "P4" and m["daypart"] == "evening"
    assert (m["generation"]["peakRewrite"], m["generation"]["peakSwitched"]) == (True, False)


def test_rewrite_counts_cost_and_respects_deadline(fake_model, tmp_log):
    fake_model.reply("a", _a_json(), 1000, 500).reply("b", _b_json(candidates=NON_PEAK), 1000, 500).reply(
        "b", _b_json(candidates=ON_PEAK, ranked=[]), 1000, 500)
    m = _run(deadline=time.monotonic() + 20)["memory"]
    assert m["generation"]["costCny"] == round(3 * config.cost_cny(1000, 500), 4)
    assert fake_model.calls[2]["timeout"] <= 20
    # No time left after call B: the extra call is skipped, the job still answers.
    fake_model.reply("a", _a_json())

    def slow_b(messages):
        time.sleep(0.4)
        return _b_json(candidates=NON_PEAK)

    fake_model.respond("b", slow_b)
    out = _run(deadline=time.monotonic() + 0.3)
    assert out["status"] == "ok"
    assert fake_model.count("b") == 3  # 2 from the first job, 1 here: no rewrite
    assert out["memory"]["generation"]["peakRewrite"] is False
    assert out["memory"]["push"]["coverPhotoId"] == "P2"


def test_time_limits():
    # Call A 120 s and the job 300 s are a user decision (2026-11-01);
    # call B (and the cover rewrite) stays 30 s; jobs are kept 15 min.
    assert config.CALL_A_TIMEOUT == 120
    assert config.CALL_B_TIMEOUT == 30
    assert config.JOB_CAP_SECONDS == 300
    assert config.JOB_TTL_SECONDS == 900
    assert compose.ATTEMPTS == 2


def test_unusable_photo_ids_fall_back_to_survivors_not_f0(fake_model, tmp_log):
    # Code review repro: the winning model sentence cites no usable photo
    # (photo_ids empty), the cover rewrite cannot write, a valid T2 exists.
    no_ids = [{"text": "长椅上，猫睡着", "photo_ids": [], "uses": ["长椅"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=no_ids)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["generation"]["bodySource"] == "template_T2"
    assert m["push"]["body"] == "从花丛拍到了舞台"
    assert m["push"]["coverPhotoId"] == "P4"
    assert m["generation"]["peakSwitched"] is False


def test_unusable_winner_gives_way_to_next_model_sentence(fake_model, tmp_log):
    # The first scores 2 (rank 1 + number 1) and would win, but P9 is not a
    # photo of this group; the second scores 1.5 (rank 0.5 + colour 1).
    cands = [{"text": "长椅上，一只猫睡着", "photo_ids": ["P9"], "uses": ["长椅"]},
             {"text": "红色的花开满花坛", "photo_ids": ["P1"], "uses": ["花坛"]}]
    fake_model.reply("a", _a_json()).reply("b", _b_json(candidates=cands)).reply(
        "b", _b_json(candidates=[], ranked=[], cannot_write=True))
    m = _run()["memory"]
    assert m["push"]["body"] == "红色的花开满花坛"
    assert m["push"]["coverPhotoId"] == "P1"
    assert m["generation"]["peakSwitched"] is True

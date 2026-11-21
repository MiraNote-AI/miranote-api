"""Repeat-theme compose after call A: calls B and C, the repeat checks,
templates T1 / T2 / F0, fallback questions and the memory (handoff 6.9,
decisions 2026-11-20 H1-H4)."""

from __future__ import annotations

import json
import time

import pytest

import body
import compose
import llm
import questions
import repeat
import rounds
import texts
import verify
from check_a import AResult, PhotoFacts, Subject
from tests.test_compose import _log, _photo, _texts, jpeg

WHERE = {"daypart": "evening", "daypart_share": 0.8, "hour_from": 17, "hour_to": 19,
         "weekpart": "mixed", "places": 1}
THEME = {"id": "T7", "name": "窗外的晚霞", "noun": "晚霞", "same_part": "推开窗看到的晚霞",
         "where": WHERE}
TAKEN = {
    "P1": "2026-09-02T18:30:00-05:00",
    "P2": "2026-09-15T18:10:00-05:00",
    "P3": "2026-10-03T18:05:00-05:00",
    "P4": "2026-10-11T18:00:00-05:00",
    "P5": "2026-10-25T17:50:00-05:00",
    "P6": "2026-11-08T17:20:00-06:00",
    "P7": "2026-11-20T17:40:00-06:00",
}
IDS = sorted(TAKEN)
GOOD = "推开窗的那一下，天已经烧起来了"
GOOD_Q = {"text": "这些傍晚里，哪一晚最让你停下脚步？", "type": "which_most", "why": "挑一晚"}


def _meta(theme=None, order=None):
    photos = [{"id": pid, "taken_at": TAKEN[pid], "labels": ["sky"], "faces": 0,
               "favorite": False, "edited": False} for pid in (order or IDS)]
    return {"mode": "repeat", "group_key": "theme-T7", "tz": "America/Chicago",
            "theme": dict(theme or THEME), "recent_question_types": [], "photos": photos}


def _images():
    return {pid: jpeg(color=(30 * n, 90, 60)) for n, pid in enumerate(IDS, 1)}


def _a(peak="P4", bad=()):
    photos = [_photo(pid, [("晚霞", "scenery")], "窗外的天空被晚霞染成橙红色，楼顶在画面下方。",
                     scene="home_indoor", **(dict(keep=False, drop="unclear") if pid in bad else {}))
              for pid in IDS]
    return json.dumps({"photos": photos, "peak_id": peak, "skip": False, "skip_reason": ""},
                      ensure_ascii=False)


def _b(candidates=None, ranked=None):
    if candidates is None:
        candidates = [{"text": GOOD, "style": "诗意", "photo_ids": ["P4"]}]
    return json.dumps({"scene": "推开窗看晚霞", "cannot_write": False, "candidates": candidates,
                       "question": {"ranked": [GOOD_Q] if ranked is None else ranked}},
                      ensure_ascii=False)


def _run(meta=None):
    return repeat.run(meta or _meta(), _images(), time.monotonic() + 300)


def _facts():
    return AResult(photos={pid: PhotoFacts(id=pid, subjects=[Subject("晚霞", "scenery")],
                                           description="窗外的晚霞")
                           for pid in IDS},
                   selected=list(IDS), peak="P4", verdict="ok", order=list(IDS))


# ---------- memory ----------

def test_repeat_memory_titles_and_cta(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b())
    out = _run()
    assert out["status"] == "ok"
    m = out["memory"]
    assert m["push"]["title"] == m["story"]["firstFrameLabel"] == m["ending"]["factLine"] \
        == "窗外的晚霞 · 9月–11月"
    assert m["ending"]["ctaLabel"] == "留住这些晚霞"
    assert m["mode"] == "repeat" and m["bucket"] == "" and m["themeId"] == "T7"
    assert m["id"] == "theme-T7"
    assert m["date"] == "2026-11-20"  # the newest kept photo, local date
    assert m["daypart"] == "evening"  # the peak's local hour, 18:00
    assert m["push"]["body"] == m["ending"]["detail"] == GOOD
    assert m["push"]["coverPhotoId"] == "P4"
    assert m["ending"]["questions"] == [{"text": GOOD_Q["text"], "type": "which_most",
                                         "source": "llm"}]
    assert set(m["facts"]) == set(IDS)
    assert m["generation"]["copyVersion"] == "repeat_v1"
    assert m["generation"]["bodySource"] == "llm"
    assert m["generation"]["peakRewrite"] is False and m["generation"]["peakSwitched"] is False


def test_repeat_memory_without_noun_and_b_failing_is_f0(fake_model, tmp_log):
    theme = dict(THEME, noun="")
    fake_model.reply("a", _a())
    fake_model.fail("b", llm.ModelError("down")).fail("b", llm.ModelError("down"))
    m = _run(_meta(theme))["memory"]
    assert m["ending"]["ctaLabel"] == "留住这些瞬间"
    assert m["push"]["body"] == "7 张照片，7 个不同的日子"
    assert m["ending"]["detail"] == ""
    assert m["generation"]["bodySource"] == "fallback_F0"
    assert [q["text"] for q in m["ending"]["questions"]] == [
        "这些瞬间的三个关键词？", "这些瞬间里，最喜欢哪一张？", "给这些瞬间配一首歌？"]


def test_repeat_story_order(fake_model, tmp_log):
    fake_model.reply("a", _a(peak="P4", bad=("P6",))).reply("b", _b())
    shuffled = ["P5", "P2", "P7", "P1", "P4", "P6", "P3"]
    m = _run(_meta(order=shuffled))["memory"]
    story = m["story"]["photos"]
    assert [p["id"] for p in story] == ["P4", "P1", "P2", "P3", "P5", "P7"]
    assert [p["isPeak"] for p in story] == [True, False, False, False, False, False]
    assert story[0]["takenAt"] == TAKEN["P4"]
    assert set(m["facts"]) == {"P1", "P2", "P3", "P4", "P5", "P7"}


# ---------- call B ----------

def test_repeat_b_prompt_carries_theme_months_where(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b())
    _run()
    b_call = next(c for c in fake_model.calls if c["call"] == "b")
    text = "\n".join(_texts(b_call["messages"][1]["content"]))
    where = rounds.where_line(WHERE, texts.json("names"))
    assert where and where in text
    for part in ("窗外的晚霞", "晚霞", "推开窗看到的晚霞", "9月–11月", "哪一次最", "每次都一样的那一刻"):
        assert part in text
    assert "{" not in text.replace("{{", "")  # every placeholder filled


def test_repeat_b_prompt_without_where_says_none(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b())
    _run(_meta(dict(THEME, where=None)))
    b_call = next(c for c in fake_model.calls if c["call"] == "b")
    text = "\n".join(_texts(b_call["messages"][1]["content"]))
    assert "遇到它的那一刻" in text


# ---------- program checks ----------

BANNED = ["同一个", "同一只", "又是", "还是那", "总是", "坚持", "爱拍", "拍了 3 次", "拍了三次",
          "每天", "天天", "5 次", "十二次"]


@pytest.mark.parametrize("word", BANNED)
def test_repeat_banned_words_block(word):
    c = body.Candidate(text="窗外{}的晚霞".format(word), source="llm", photo_ids=["P4"])
    reason = body.reject_reason(c, _facts(), mode="repeat")
    assert reason is not None and reason.startswith("lexicon:")


@pytest.mark.parametrize("word", BANNED)
def test_repeat_words_keep_life_fragment_verdicts(word):
    c = body.Candidate(text="窗外{}的晚霞".format(word), source="llm", photo_ids=["P4"])
    reason = body.reject_reason(c, _facts())
    assert reason == ("lexicon:又" if "又" in word else None)


@pytest.mark.parametrize("text", ["每一次推开窗，天都烧起来了", "这一次的晚霞", "下一次推开窗"])
def test_repeat_patterns_leave_every_time_alone(text):
    c = body.Candidate(text=text, source="llm", photo_ids=["P4"])
    assert body.reject_reason(c, _facts(), mode="repeat") is None


def test_repeat_question_types_and_bans():
    a = _facts()
    which = questions.Question(text=GOOD_Q["text"], type="which_most", source="llm")
    assert questions.reject_reason(which, a, mode="repeat") is None
    assert questions.reject_reason(which, a) == "unknown_type:which_most"
    for qtype in ("which_most", "that_moment", "since_when", "hands_busy", "next_time"):
        assert qtype in questions.type_ids(mode="repeat")
    assert set(questions.type_ids()) <= set(questions.type_ids(mode="repeat"))
    daily = questions.Question(text="你每天都会推开窗看它？", type="that_moment", source="llm")
    assert questions.reject_reason(daily, a, mode="repeat").startswith("forbidden:")


# ---------- templates and fallbacks ----------

def test_repeat_templates_order():
    ids = list(IDS)
    out = body.templates_repeat(THEME, ids)
    assert [(c.source, c.text) for c in out] == [
        ("template_T1", "那些傍晚，和晚霞"),
        ("template_T2", "7 张晚霞，7 个不同的日子"),
        ("fallback_F0", "7 张照片，7 个不同的日子"),
    ]
    no_daypart = body.templates_repeat(dict(THEME, where=dict(WHERE, daypart=None)), ids)
    assert [c.source for c in no_daypart] == ["template_T2", "fallback_F0"]
    assert [c.source for c in body.templates_repeat(dict(THEME, where=None), ids)] \
        == ["template_T2", "fallback_F0"]
    no_noun = body.templates_repeat(dict(THEME, noun=""), ids)
    assert [(c.source, c.text) for c in no_noun] == [("fallback_F0", "7 张照片，7 个不同的日子")]


def test_repeat_template_used_when_no_model_sentence_survives(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b([{"text": "每天推开窗，都是晚霞", "photo_ids": ["P4"]}]))
    m = _run()["memory"]
    assert m["push"]["body"] == m["ending"]["detail"] == "那些傍晚，和晚霞"
    assert m["generation"]["bodySource"] == "template_T1"


def test_repeat_fallback_questions():
    for noun, shown in (("晚霞", "晚霞"), ("", "瞬间")):
        out = questions.final([], noun, mode="repeat")
        assert [q.text for q in out] == ["这些{}的三个关键词？".format(shown),
                                         "这些{}里，最喜欢哪一张？".format(shown),
                                         "给这些{}配一首歌？".format(shown)]
        assert all(q.type in questions.type_ids(mode="repeat") for q in out)
        assert [q.source for q in out] == ["fallback_B1", "fallback_B2", "fallback_B3"]


# ---------- call C ----------

def test_repeat_c_whole_theme_false_drops_sentence(fake_model, tmp_log):
    album = "这是你的一整个晚霞合集"
    fake_model.reply("a", _a()).reply("b", _b([
        {"text": album, "style": "自然", "photo_ids": ["P4"]},
        {"text": GOOD, "style": "诗意", "photo_ids": ["P4"]},
    ]))
    fake_model.reply("c", json.dumps({
        "sentences": [{"id": "S1", "grounded": True, "whole_theme": False, "fluent": True,
                       "reason": "合集说明"},
                      {"id": "S2", "grounded": True, "whole_theme": True, "fluent": True,
                       "reason": ""}],
        "questions": [{"id": "Q1", "grounded": True, "safe": True, "reason": ""}],
    }, ensure_ascii=False))
    m = _run()["memory"]
    assert m["push"]["body"] == GOOD
    log = _log(tmp_log)
    assert [album, "c_not_whole_theme:合集说明"] in log["body"]["rejected"]
    assert log["c"]["sentences"]["S1"]["whole_theme"] is False


def test_repeat_c_prompt_asks_whole_theme(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b())
    _run()
    c_call = next(c for c in fake_model.calls if c["call"] == "c")
    text = "\n".join(_texts(c_call["messages"][1]["content"]))
    assert "whole_theme" in text and "mentions_cover" not in text
    assert "窗外的晚霞" in text


def test_verify_parse_reads_whole_theme_only_in_repeat():
    reply = json.dumps({"sentences": [{"id": "S1", "grounded": True, "whole_theme": False,
                                       "fluent": True}]})
    assert verify.parse(reply, 1, 0, mode="repeat").sentences["S1"].whole_theme is False
    assert verify.parse(reply, 1, 0).sentences["S1"].whole_theme is True
    unsure = json.dumps({"sentences": [{"id": "S1", "grounded": True, "fluent": True}]})
    assert verify.parse(unsure, 1, 0, mode="repeat").sentences["S1"].whole_theme is True


# ---------- cover ----------

def test_repeat_no_cover_rewrite(fake_model, tmp_log):
    # A surviving sentence that does not cite the peak keeps the cover.
    other = "推开窗，楼顶后面的天刚刚粉了"
    fake_model.c_cover = lambda text: False
    fake_model.reply("a", _a()).reply("b", _b([{"text": other, "photo_ids": ["P2"]}]))
    m = _run()["memory"]
    assert fake_model.count("b") == 1
    assert m["push"]["body"] == other and m["push"]["coverPhotoId"] == "P4"
    assert m["story"]["photos"][0]["id"] == "P4"


def test_repeat_no_cover_rewrite_when_only_templates_left(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b([{"text": "又是那片晚霞", "photo_ids": ["P4"]}]))
    m = _run()["memory"]
    assert fake_model.count("b") == 1
    assert m["push"]["coverPhotoId"] == "P4"
    assert m["generation"]["peakRewrite"] is False and m["generation"]["peakSwitched"] is False
    log = _log(tmp_log)
    assert "peak_rewrite" not in log


def test_repeat_life_fragment_lexicon_unchanged_by_mode_default():
    a = _facts()
    c = body.Candidate(text="又一个傍晚", source="llm", photo_ids=["P4"])
    assert body.reject_reason(c, a) == body.reject_reason(c, a, mode="life_fragment") == "lexicon:又"


def test_compose_run_routes_repeat(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b())
    out = compose.run(_meta(), _images(), time.monotonic() + 300)
    assert out["status"] == "ok" and out["memory"]["mode"] == "repeat"


# ---------- system prompts ----------

def _systems(fake_model):
    return [(c["call"], c["messages"][0]["content"]) for c in fake_model.calls]


def test_repeat_calls_use_repeat_system_prompts(fake_model, tmp_log):
    long_s = "推开窗的那一下，楼顶后面的整片天空已经烧起来了，橙红色铺满窗框"
    long_q = {"text": "推开窗看到楼顶后面那一整片烧起来的天空，你的第一反应是什么？",
              "type": "that_moment", "why": "那一刻"}
    fake_model.reply("a", _a())
    fake_model.reply("b", _b([{"text": long_s, "style": "诗意", "photo_ids": ["P4"]}], [long_q]))
    fake_model.reply("b", _b([], [{"text": "推开窗看到它，第一反应是什么？", "type": "that_moment",
                                   "why": "改短"}]))
    fake_model.reply("b", _b([{"text": GOOD, "style": "诗意", "photo_ids": ["P4"]}], []))
    m = _run()["memory"]
    assert m["push"]["body"] == GOOD
    want = {"a": texts.raw("repeat_a_system").strip(), "b": texts.raw("repeat_b_system").strip(),
            "c": texts.raw("repeat_verify_c_system").strip()}
    seen = _systems(fake_model)
    assert [c for c, _ in seen] == ["a", "b", "b", "c", "b", "c"]
    assert all(system == want[call] for call, system in seen)


def test_life_fragment_calls_keep_v1_system_prompts(fake_model, tmp_log):
    from tests.test_compose import IMAGES, _a_json, _b_json
    from tests.test_compose import _meta as _lf_meta
    fake_model.reply("a", _a_json()).reply("b", _b_json())
    assert compose.run(_lf_meta(), dict(IMAGES), time.monotonic() + 300)["status"] == "ok"
    want = {"a": texts.raw("collect_a_system").strip(), "b": texts.raw("collect_b_system").strip(),
            "c": texts.raw("verify_c_system").strip()}
    seen = _systems(fake_model)
    assert {c for c, _ in seen} == {"a", "b", "c"}
    assert all(system == want[call] for call, system in seen)

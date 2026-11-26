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
GOOD = "楼下那排自行车，停得整整齐齐"
GOOD_Q = {"text": "第一次注意到它，是在哪儿？", "type": "which_most", "why": "挑一晚"}


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
    assert m["generation"]["copyVersion"] == "repeat_v2"
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
    assert "集中在：无" in text


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


@pytest.mark.parametrize("text", ["有一次推开窗", "再一次推开窗", "印象最深的一次是？", "每一次", "这一次", "下一次", "哪一次"])
def test_repeat_patterns_leave_lone_once_alone(text):
    c = body.Candidate(text=text, source="llm", photo_ids=["P4"])
    assert body.reject_reason(c, _facts(), mode="repeat") is None


@pytest.mark.parametrize("text", ["拍了 3 次", "拍了三次", "两次推开窗", "十次推开窗", "每天推开窗", "天天推开窗"])
def test_repeat_patterns_still_block_counts(text):
    c = body.Candidate(text=text, source="llm", photo_ids=["P4"])
    assert body.reject_reason(c, _facts(), mode="repeat") is not None


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
                       "natural": True, "ominous": False, "reason": "合集说明"},
                      {"id": "S2", "grounded": True, "whole_theme": True, "fluent": True,
                       "natural": True, "ominous": False, "reason": ""}],
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


def test_repeat_c_ominous_drops_sentence(fake_model, tmp_log):
    bad = "它躺在那儿，像睡着了"
    fake_model.c_ominous = {bad}
    fake_model.reply("a", _a()).reply("b", _b([
        {"text": bad, "style": "自然", "photo_ids": ["P4"]},
        {"text": GOOD, "style": "诗意", "photo_ids": ["P4"]},
    ]))
    m = _run()["memory"]
    assert m["push"]["body"] == GOOD
    log = _log(tmp_log)
    assert [bad, "c_ominous:"] in log["body"]["rejected"]
    assert log["c"]["sentences"]["S1"]["ominous"] is True


def test_repeat_c_not_natural_drops_sentence(fake_model, tmp_log):
    bad = "天烧起来，撞进镜头里的慵懒"
    fake_model.c_unnatural = {bad}
    fake_model.reply("a", _a()).reply("b", _b([
        {"text": bad, "style": "自然", "photo_ids": ["P4"]},
        {"text": GOOD, "style": "诗意", "photo_ids": ["P4"]},
    ]))
    m = _run()["memory"]
    assert m["push"]["body"] == GOOD
    log = _log(tmp_log)
    assert [bad, "c_not_natural:"] in log["body"]["rejected"]
    assert log["c"]["sentences"]["S1"]["natural"] is False


def test_repeat_missing_natural_drops_only_that_sentence(fake_model, tmp_log):
    other = "推开窗，楼顶后面的天刚刚粉了"
    fake_model.reply("a", _a()).reply("b", _b([
        {"text": other, "style": "自然", "photo_ids": ["P4"]},
        {"text": GOOD, "style": "诗意", "photo_ids": ["P4"]},
    ]))
    fake_model.reply("c", json.dumps({
        "sentences": [{"id": "S1", "grounded": True, "whole_theme": True, "fluent": True,
                       "ominous": False, "reason": ""},
                      {"id": "S2", "grounded": True, "whole_theme": True, "fluent": True,
                       "natural": True, "ominous": False, "reason": ""}],
        "questions": [{"id": "Q1", "grounded": True, "safe": True, "reason": ""}],
    }, ensure_ascii=False))
    m = _run()["memory"]
    assert m["push"]["body"] == GOOD
    assert [other, "c_not_natural:"] in _log(tmp_log)["body"]["rejected"]


def test_verify_parse_reads_natural_and_ominous_only_in_repeat():
    unsure = json.dumps({"sentences": [{"id": "S1", "grounded": True, "fluent": "unsure",
                                        "natural": "unsure", "ominous": None}]})
    r = verify.parse(unsure, 1, 0, mode="repeat").sentences["S1"]
    assert r.natural is False and r.ominous is True and r.fluent is True
    flipped = json.dumps({"sentences": [{"id": "S1", "grounded": True, "natural": False,
                                         "ominous": True}]})
    life = verify.parse(flipped, 1, 0).sentences["S1"]
    assert life.natural is True and life.ominous is False
    ok = json.dumps({"sentences": [{"id": "S1", "grounded": True, "natural": True,
                                    "ominous": False}]})
    r = verify.parse(ok, 1, 0, mode="repeat").sentences["S1"]
    assert r.natural is True and r.ominous is False


def test_life_fragment_c_ignores_natural_and_ominous():
    reply = json.dumps({"sentences": [{"id": "S1", "grounded": True, "mentions_cover": True,
                                       "natural": False, "ominous": True}]})
    v = verify.parse(reply, 1, 0).sentences["S1"]
    assert (v.natural, v.ominous) == (True, False)


def test_repeat_c_prompt_asks_natural_and_ominous(fake_model, tmp_log):
    fake_model.reply("a", _a()).reply("b", _b())
    _run()
    c_call = next(c for c in fake_model.calls if c["call"] == "c")
    text = "\n".join(_texts(c_call["messages"][1]["content"]))
    assert "natural" in text and "ominous" in text
    assert "natural 算不自然（false），ominous 算有（true）" in text


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
    fake_model.reply("b", _b([], [{"text": "这扇窗，平时一般几点会推开？", "type": "that_moment",
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


# ---------- example copies, taboo words (2026-11-26) ----------

EXAMPLE_TEXT = "推开窗的那一下 天已经烧起来了！"


def _example_lines():
    out = []
    section = None
    for line in texts.raw("repeat_b_user").split("\n"):
        if line.startswith("好的例子"):
            section = "good"
        elif line.startswith("不好的例子"):
            section = "bad"
        elif line.startswith("## "):
            section = None
        if section == "good" and line.startswith("- "):
            out.append(line[2:].split("（")[0].strip())
        if section == "bad" and "✓ " in line:
            out.append(line.split("✓ ", 1)[1].split("（")[0].strip())
    return out


def test_repeat_examples_file_covers_every_prompt_example():
    listed = texts.json("repeat_examples")
    lines = _example_lines()
    assert len(lines) >= 20
    assert [t for t in lines if t not in listed] == []


def test_repeat_examples_file_covers_every_question_type_example():
    listed = texts.json("repeat_examples")
    shown = [t["example"] for t in texts.json("repeat_question_types")["types"]
             if t["example"] != "\u2014"]
    assert len(shown) >= 14
    assert [e for e in shown if e not in listed] == []
    assert len(listed) == len(set(listed))


def test_repeat_example_copy_rejected_with_punctuation_changes(fake_model, tmp_log):
    copy = "路过书店， 看一眼橱窗！"
    c = body.Candidate(text=copy, source="llm", photo_ids=["P4"])
    assert body.reject_reason(c, _facts(), mode="repeat").startswith("example_copy:")
    near = body.Candidate(text="路过书店，看了一眼橱窗", source="llm", photo_ids=["P4"])
    assert body.reject_reason(near, _facts(), mode="repeat").startswith("example_copy:")
    fake_model.reply("a", _a()).reply("b", _b([
        {"text": copy, "style": "自然", "photo_ids": ["P4"]},
        {"text": GOOD, "style": "自然", "photo_ids": ["P4"]},
    ]))
    m = _run()["memory"]
    assert m["push"]["body"] == GOOD
    assert [copy, "example_copy:路过书店，看一眼橱窗"] in _log(tmp_log)["body"]["rejected"]


def test_repeat_example_copy_leaves_other_sources_and_modes_alone():
    t = "路过书店，看一眼橱窗"
    a = _facts()
    assert body.reject_reason(body.Candidate(text=t, source="llm", photo_ids=["P4"]), a) is None
    tmpl = body.Candidate(text=t, source="template_T1", photo_ids=["P4"])
    assert body.reject_reason(tmpl, a, mode="repeat") is None
    assert body.example_copy(GOOD) is None


def test_life_fragment_ignores_example_guard():
    c = body.Candidate(text="阳台的多肉，挤满了一个盆", source="llm", photo_ids=["P4"])
    assert body.reject_reason(c, _facts()) is None


def test_repeat_question_copy_rejected(fake_model, tmp_log):
    copy = {"text": "下一盆，想养什么？", "type": "next_time", "why": "照抄"}
    q = questions.Question(text=copy["text"], type="next_time", source="llm")
    assert questions.reject_reason(q, _facts(), mode="repeat").startswith("example_copy:")
    q2 = questions.Question(text="下一盆 想养什么", type="next_time", source="llm")
    assert questions.reject_reason(q2, _facts(), mode="repeat").startswith("example_copy:")
    assert questions.reject_reason(q, _facts(), mode="life_fragment") is None
    fake_model.reply("a", _a()).reply("b", _b(ranked=[copy]))
    m = _run()["memory"]
    assert all(x["source"] != "llm" for x in m["ending"]["questions"])
    assert len(m["ending"]["questions"]) == 3


@pytest.mark.parametrize("text,rejected", [
    ("它们就趴在那儿不动了", True),
    ("一动不动地看着窗外", True),
    ("换着地方睡", False),
    ("晚霞不动声色地铺开", False),
])
def test_repeat_taboo_words_block_only_listed_phrases(text, rejected):
    c = body.Candidate(text=text, source="llm", photo_ids=["P4"])
    reason = body.reject_reason(c, _facts(), mode="repeat")
    assert (reason is not None and reason.startswith("lexicon:")) is rejected
    if not rejected:
        assert reason is None
    q = questions.Question(text=text + "？", type="that_moment", source="llm")
    assert (questions.reject_reason(q, _facts(), mode="repeat") is not None) is rejected


def test_repeat_copy_version_is_v2():
    import config
    assert config.REPEAT_COPY_VERSION == "repeat_v2"

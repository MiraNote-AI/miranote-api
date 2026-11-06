"""Ending questions: types, hard checks and fallbacks (copy refine handoff
2.3, 4.1, 4.2)."""

from __future__ import annotations

from typing import List

import pytest

import questions
import texts
from check_a import AResult, PhotoFacts, Subject


def _p(pid, people="0"):
    return PhotoFacts(id=pid, subjects=[Subject("拉面", "food")], description="一碗拉面",
                      people_count=people)


def _a(photos: List[PhotoFacts]) -> AResult:
    ids = [p.id for p in photos]
    return AResult(photos={p.id: p for p in photos}, selected=ids, peak=ids[-1],
                   verdict="ok", order=ids)


NOBODY = _a([_p("P1"), _p("P2"), _p("P3")])
SOMEONE = _a([_p("P1"), _p("P2", people="2-4"), _p("P3")])


def _q(text, type_="order"):
    return questions.Question(text=text, type=type_, source="llm")


def _b(*ranked):
    return {"question": {"ranked": list(ranked)}}


def test_type_table_matches_handoff_4_2():
    # Question refine 2026-11-05 added taste.
    assert questions.type_ids() == ["order", "sound", "route", "found", "zoom", "taste", "next_time",
                                    "people", "body", "imagine", "one_word", "other"]
    names = {t["id"]: t["name"] for t in questions.type_table()}
    assert names["order"] == "先后顺序" and names["other"] == "其他" and names["taste"] == "味道"


def test_prompt_types_lists_every_type():
    lines = questions.prompt_types().splitlines()
    assert len(lines) == 12
    # The narrowed meaning (question refine 2026-11-05) goes to call B.
    assert lines[0] == "  order｜先后顺序（有意义的先后：先吃完哪样、先到哪儿）｜那桌菜里，哪一样最先吃完？"
    assert "  taste｜味道｜那锅冬阴功，味道怎么样？" in lines
    # Open examples only (2026-11-05 evening): no yes/no or either/or left.
    assert not any("还是" in line or "吗？" in line for line in lines)


def test_prompt_recent_names_known_types_once():
    assert questions.prompt_recent(["sound", "sound", "zzz", "order"]) == "声音、先后顺序"
    assert questions.prompt_recent([]) == "无"


def test_from_b_keeps_model_order_and_skips_junk():
    qs = questions.from_b(_b({"text": " 那桌菜里，哪一样最先吃完？ ", "type": "order", "why": "x"},
                             "not an object",
                             {"text": "那晚台上说的话，你还记得哪一句？", "type": "sound"}))
    assert [(q.text, q.type, q.rank) for q in qs] == [
        ("那桌菜里，哪一样最先吃完？", "order", 0), ("那晚台上说的话，你还记得哪一句？", "sound", 2)]
    assert questions.from_b(None) == [] and questions.from_b({"question": {}}) == []


def test_good_question_passes():
    assert questions.reject_reason(_q("那桌菜里，哪一样最先吃完？"), NOBODY) is None


def test_over_22_chars_rejected():
    assert questions.reject_reason(_q("那" * 22 + "？"), NOBODY) == "too_long"
    assert questions.reject_reason(_q("那" * 21 + "？"), NOBODY) is None


@pytest.mark.parametrize("word", ["又", "一直没", "好久没", "别忘了", "该", "错过", "再也", "还没"])
def test_forbidden_words_rejected(word):
    assert questions.reject_reason(_q("那天" + word + "去了哪儿？"), NOBODY) == "forbidden:" + word


def test_pronouns_rejected_but_other_allowed():
    assert questions.reject_reason(_q("那天他说了什么？"), NOBODY) == "pronoun:他"
    assert questions.reject_reason(_q("她点的是哪一碗？"), NOBODY) == "pronoun:她"
    assert questions.reject_reason(_q("其他几碗，是什么面？", "zoom"), NOBODY) is None


def test_unknown_type_rejected():
    assert questions.reject_reason(_q("那碗面，是什么面？", "flavour"), NOBODY) == "unknown_type:flavour"
    assert questions.reject_reason(_q("那碗面，是什么面？", ""), NOBODY) == "unknown_type:"


def test_people_type_needs_people_in_the_photos():
    q = _q("合照，是谁提议要拍的？", "people")
    assert questions.reject_reason(q, NOBODY) == "people_without_people"
    assert questions.reject_reason(q, SOMEONE) is None


def test_hard_check_keeps_top_three_in_order_and_drops_repeats():
    qs = [_q("那桌菜里，哪一样最先吃完？"), _q("那天他说了什么？"), _q("那桌菜里，哪一样最先吃完？"),
          _q("这家店，是怎么找到的？", "found")]
    passed, rejected = questions.hard_check(qs, NOBODY)
    assert [q.text for q in passed] == ["那桌菜里，哪一样最先吃完？"]  # the 4th is past the top 3
    assert rejected == [("那天他说了什么？", "pronoun:他"), ("那桌菜里，哪一样最先吃完？", "duplicate")]


def test_fallbacks_are_b1_b2_b3_with_types_and_daypart():
    fb = questions.fallbacks("下午")
    assert [(q.text, q.type, q.source) for q in fb] == [
        ("那个下午的三个关键词：＿＿、＿＿、＿＿", "one_word", "fallback_B1"),
        ("满分 10 分，那个下午几分？", "one_word", "fallback_B2"),
        ("如果给那个下午配一首歌，会是哪首？", "imagine", "fallback_B3"),
    ]
    assert texts.json("question_bank")["fallback_types"] == {
        "B1": "one_word", "B2": "one_word", "B3": "imagine"}


def test_final_is_survivors_or_fallbacks():
    one = [_q("那桌菜里，哪一样最先吃完？")]
    assert questions.final(one, "下午") == one  # not topped up with fallbacks
    assert [q.source for q in questions.final([], "下午")] == [
        "fallback_B1", "fallback_B2", "fallback_B3"]
    assert questions.final(one, "下午")[0].payload() == {
        "text": "那桌菜里，哪一样最先吃完？", "type": "order", "source": "llm"}


def test_photo_ids_and_cover_marker_rejected():
    # Seen on real photos (2026-11-05): "P1 he P2 dou zai bai sha ...".
    assert questions.reject_reason(_q("P1和P2，哪一张先拍的？"), NOBODY) == "photo_id:P1"
    assert questions.reject_reason(_q("封面那碗面，是什么面？", "zoom"), NOBODY) == "marker:封面"
    assert questions.reject_reason(_q("那碗面，是什么面？", "zoom"), NOBODY) is None


def test_question_echoing_its_type_is_stripped():
    # The prompt's examples end with "（sound）"; a copied annotation is not
    # copy a person should see (full check 2026-11-05).
    qs = questions.from_b({"question": {"ranked": [
        {"text": "推车上，卖的是什么？（zoom）", "type": "zoom", "why": ""},
        {"text": "- 这家店，是怎么找到的？", "type": "found", "why": ""}]}})
    assert [q.text for q in qs] == ["推车上，卖的是什么？", "这家店，是怎么找到的？"]


def test_guitar_is_not_a_pronoun_in_questions():
    q = questions.Question(text="台上那把吉他，弹的是哪首歌？", type="sound", source="llm")
    assert questions.reject_reason(q, SOMEONE) is None


def test_taste_is_a_known_type():
    # Question refine 2026-11-05: the new taste type passes the type check,
    # and "味道" is not a banned word for questions (only for detail lines).
    assert questions.reject_reason(_q("那锅冬阴功，是什么味道？", "taste"), NOBODY) is None


LONG_Q = {"text": "从外面的街道一路走到这个舞台，中间经过了哪段路？", "type": "route", "why": "x"}


def test_too_long_lists_only_questions_cut_for_length():
    qs = questions.from_b(_b(LONG_Q, {"text": "那晚台上说的话，你还记得哪一句？", "type": "sound", "why": "y"}))
    assert [q.text for q in questions.too_long(qs, NOBODY)] == [LONG_Q["text"]]


def test_with_shortened_keeps_rank_and_type():
    qs = questions.from_b(_b({"text": "这家店，是怎么找到的？", "type": "found", "why": "y"}, LONG_Q))
    long_qs = questions.too_long(qs, NOBODY)
    reply = questions.from_b(_b({"text": "从街上到台下，去了哪儿？", "type": "zoom", "why": "改短"}))
    out = questions.with_shortened(qs, long_qs, reply)
    assert [(q.text, q.type, q.rank) for q in out] == [
        ("这家店，是怎么找到的？", "found", 0), ("从街上到台下，去了哪儿？", "route", 1)]


def test_with_shortened_ignores_a_reply_of_another_count():
    qs = questions.from_b(_b(LONG_Q))
    out = questions.with_shortened(qs, questions.too_long(qs, NOBODY), [])
    assert [q.text for q in out] == [LONG_Q["text"]]

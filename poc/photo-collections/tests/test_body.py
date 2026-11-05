"""Detail sentence hard checks, templates and pick (copy refine handoff 2.3,
2.5, 2.6, 3): one case per rule."""

from __future__ import annotations

from typing import List

import pytest

import body
import texts
from check_a import AResult, PhotoFacts, Subject

WHEN = body.When(weekday="周四", daypart="夜晚")


def _p(pid, subjects=(), description="", text=(), people="0"):
    return PhotoFacts(
        id=pid,
        subjects=[Subject(n, t) for n, t in subjects],
        description=description,
        text=list(text),
        people_count=people,
    )


def _a(photos: List[PhotoFacts], peak=None, selected=None) -> AResult:
    ids = [p.id for p in photos]
    return AResult(
        photos={p.id: p for p in photos},
        selected=selected or ids,
        peak=peak or ids[-1],
        verdict="ok",
        order=ids,
    )


STAGE = [
    _p("P1", [("长椅", "object"), ("橘猫", "animal")], "猫睡在长椅上"),
    _p("P2", [("舞台", "object"), ("灯光", "object")], "一束光落在台中央"),
    _p("P3", [("舞台", "object")], "蓝色的幕布", ["ABCDEFGHIJKLMNOPQRSTUV"]),
]


def _ai(text, photo_ids=("P2",)):
    return body.Candidate(text=text, source="llm", photo_ids=list(photo_ids))


def _pick(cands, a=None, templates=None, raw_count=23):
    a = a or _a(STAGE, peak="P2")
    for i, c in enumerate(cands):
        c.rank = i
    return body.pick(cands, templates or [], a, raw_count=raw_count, when=WHEN)


def _reason(text, a=None, source="llm"):
    a = a or _a(STAGE, peak="P2")
    return body.reject_reason(body.Candidate(text=text, source=source, photo_ids=["P2"]), a)


def test_char_len_strips_spaces_and_counts_everything_else_once():
    assert body.char_len("菜单上写着 Too Thai") == 12
    assert body.char_len("台中央，落下一束光") == 9


def test_22_chars_pass_23_rejected():
    assert body.MAX_CHARS == 22
    ok = "天" * 22
    assert _reason(ok) is None
    assert _reason(ok + "天") == "too_long"


def test_trailing_full_stop_stripped():
    pick = _pick([_ai("  台中央，落下一束光。")])
    assert pick.text == "台中央，落下一束光"
    assert pick.source == "llm"


@pytest.mark.parametrize("text,word", [
    ("台中央，温馨的一束光", "温馨"),        # mood
    ("那晚的灯，很好看", "好看"),            # vague praise
    ("蓝色幕布后，男子握着麦克风", "男子"),  # person words (new)
    ("台上站着一个女人", "女人"),
    ("两个小孩站在台前", "小孩"),
    ("拉面上，很热", "热"),                  # unseen senses
    ("台上好像有一束光", "好像"),            # guessing
    ("灯似乎亮了", "似乎"),
    ("那晚的时光，停在台上", "时光"),        # stock words, only two
    ("一束光，留在回忆里", "回忆"),
    ("旅行的最后一晚，台上亮着灯", "旅行"),  # event, still banned
    ("台上的灯又亮了", "又"),
])
def test_banned_words_rejected(text, word):
    assert _reason(text) == "lexicon:" + word


@pytest.mark.parametrize("text", [
    "灯串像一排低低的星星",          # metaphor words are free now
    "幕布仿佛一片蓝色的海",
    "先是黄昏，然后台上的灯亮了",    # time order words
    "后来，台上只剩一束光",
    "暖黄的光，落在台中央",          # light / colour feelings
    "温柔的灯光，铺满了台面",
    "那晚的桌子很满",
    "门外柴火堆得老高",
    "一张票根，一个故事的开头",      # 故事 / 旅程 / 曾经 released
])
def test_released_words_pass(text):
    assert _reason(text) is None


def test_person_words_not_shielded_by_subject_names():
    # Call A may still name a subject 男子; the sentence may not.
    a = _a([_p("P1", [("男子", "person")], "一个男子站在台上"), STAGE[1]], peak="P2")
    assert _reason("男子站在台上", a) == "lexicon:男子"


def test_person_words_shielded_by_photo_text():
    a = _a([_p("P1", [("招牌", "object")], "招牌", ["女人街"]), STAGE[1]], peak="P2")
    assert _reason("招牌上写着女人街", a) is None


def test_other_is_not_a_pronoun():
    assert _reason("其他的灯都暗了") is None
    assert _reason("他站在灯下") == "lexicon:他"


def test_lexicon_word_inside_subject_allowed():
    a = _a([_p("P1", [("香菜", "food")], "汤里有香菜"), STAGE[1]], peak="P2")
    assert _reason("汤面上，浮着几片香菜", a) is None


def test_lexicon_word_in_photo_text_allowed():
    a = _a([_p("P1", [("蛋糕", "food")], "蜡烛插在蛋糕上", ["生日快乐"]), STAGE[1]], peak="P2")
    assert _reason("蛋糕上写着生日快乐", a) is None


def test_hot_steam_allowed():
    assert _reason("拉面上，冒着热气") is None


@pytest.mark.parametrize("food", ["香菜", "香肠", "香菇", "香蕉"])
def test_visible_foods_named_with_xiang_allowed(food):
    assert _reason("盘子边上，几片" + food) is None
    assert _reason("那晚的饭很香") == "lexicon:香"


def test_lexicon_file_matches_handoff_section_3():
    words = {}
    for line in texts.lines("lexicon_push_body"):
        parts = line.split("|")
        words[parts[0]] = parts[1].split()
    assert words["陈词"] == ["时光", "回忆"]
    assert words["推测"] == ["好像", "似乎", "一定", "应该", "也许", "大概", "估计"]
    flat = {w for ws in words.values() for w in ws}
    for released in ["温暖", "暖", "温柔", "像", "仿佛", "宛如", "如同", "先", "然后", "后来",
                     "接着", "随后", "故事", "旅程", "曾经"]:
        assert released not in flat
    for added in ["男子", "女子", "男人", "女人", "男生", "女生", "小孩", "小朋友"]:
        assert added in words["人物身份关系表情"]


def test_unknown_nouns_and_numbers_are_left_to_call_c():
    # The word-by-word fact check is gone (handoff 2.3 item 6).
    assert _reason("桌上三只红灯笼，一锅冬阴功") is None


def test_lianpai_rejected():
    assert _reason("长椅上，连拍了橘猫") == "lianpai"


def test_cover_marker_rejected():
    assert _reason("封面里三人站在蓝幕前") == "marker:封面"


def test_photo_id_in_text_rejected():
    # Seen on the real model: "P1, P3, ..." A photo id is never copy.
    assert _reason("P1, P3, 橙黄色的晚霞") == "photo_id:P1"
    # Letters and digits that are not an id stay allowed.
    assert _reason("ABCDEFGHIJKLMNOP") is None


# ---------- templates (handoff 2.6) ----------

def _meta(covers_all):
    return {"raw_count": 9, "covers_all": covers_all, "photos": [
        {"id": "P1", "represents": 1}, {"id": "P2", "represents": 2}, {"id": "P3", "represents": 3}]}


def test_t1_only_when_covers_all():
    a = _a(STAGE, peak="P2")
    assert [c.source for c in body.templates(a, _meta(False), WHEN)] == ["template_T2"]
    assert "template_T1" in [c.source for c in body.templates(a, _meta(True), WHEN)]


def test_t1_new_shape_needs_subject_in_two_photos_and_sums_represents():
    a = _a(STAGE, peak="P2")
    t1 = [c for c in body.templates(a, _meta(True), WHEN) if c.source == "template_T1"]
    # Only the stage is in >= 2 selected photos (P2, P3): 2 + 3 = 5.
    assert [c.text for c in t1] == ["5 张舞台，一个周四的夜晚"]
    assert t1[0].photo_ids == ["P2", "P3"]


def test_t2_new_shape_needs_different_first_last_subjects():
    a = _a(STAGE, peak="P2")
    t2 = [c for c in body.templates(a, _meta(False), WHEN) if c.source == "template_T2"]
    assert [c.text for c in t2] == ["从长椅，到舞台"]
    same = _a([STAGE[1], STAGE[2]], peak="P2")
    assert [c for c in body.templates(same, _meta(False), WHEN) if c.source == "template_T2"] == []


def test_t2_not_checked_against_guilt_lexicon():
    text = "从长椅，到又一束光"
    t2 = body.Candidate(text=text, source="template_T2", photo_ids=["P1", "P3"])
    pick = _pick([], templates=[t2])
    assert pick.text == text and pick.source == "template_T2"
    assert _reason(text) == "lexicon:又"  # the model may not


def test_all_rejected_falls_back_to_f0():
    pick = _pick([_ai("和朋友度过的温馨午后")], raw_count=23)
    assert pick.text == "23 张照片，一个周四的夜晚"
    assert pick.source == "fallback_F0"


# ---------- scoring (handoff 2.5) ----------

def test_model_rank_decides_between_peak_sentences():
    first = _ai("台中央，落下一束光")
    second = _ai("灯光下，舞台空着")
    assert _pick([first, second]).text == "台中央，落下一束光"
    assert _pick([second, first]).text == "灯光下，舞台空着"


def test_several_photos_add_one():
    # Ranks 0 and 1 are rejected (too long); rank 2 and 3 get no rank bonus,
    # and the one tying two photos together wins.
    lone = _ai("台中央，一束光")
    two = _ai("台上的光，落在蓝幕布上", ["P2", "P3"])
    pick = _pick([_ai("y" * 30), _ai("z" * 30), lone, two])
    assert (pick.text, pick.score) == ("台上的光，落在蓝幕布上", 1)


def test_colour_and_number_no_longer_score():
    # Both score 0 now (no rank bonus, one photo): the shorter wins, where
    # the old "+1 for colour / number / text" would have picked the longer.
    a = _a(STAGE, peak="P2")
    plain = _ai("台中央，落下一束光")
    plain.rank = 3
    coloured = _ai("台中央，落下三束红色的光")
    coloured.rank = 4
    pick = body.pick([plain, coloured], [], a, raw_count=9, when=WHEN)
    assert (pick.text, pick.score) == ("台中央，落下一束光", 0)


def test_tie_prefers_ai_then_shorter():
    filler = [_ai("x" * 30), _ai("y" * 30)]  # rank 0, 1: rejected
    longer = _ai("长椅上，猫睡着了", ["P1"])
    shorter = _ai("长椅上，猫睡着", ["P1"])
    a = _a(STAGE, peak="P3")
    for i, c in enumerate(filler + [longer, shorter]):
        c.rank = i
    pick = body.pick(filler + [longer, shorter], [], a, raw_count=9, when=WHEN)
    assert pick.text == "长椅上，猫睡着"
    template = body.Candidate(text="长椅上，猫睡", source="template_T2", photo_ids=["P1"])
    ai = _ai("长椅上，猫睡着", ["P1"])
    ai.rank = 2
    pick = body.pick([ai], [template], a, raw_count=9, when=WHEN)
    assert pick.source == "llm"


def test_plain_sky_penalised():
    a = _a([_p("P1", [("天空", "scenery")], "天空里有云"),
            _p("P2", [("长椅", "object")], "猫睡在长椅上"), STAGE[2]], peak="P3")
    sky = _ai("那天的天空", ["P1"])
    cat = _ai("长椅上，猫睡着", ["P2"])
    assert _pick([sky, cat], a=a).text == "长椅上，猫睡着"


@pytest.mark.parametrize("sky", ["蓝色的天空", "天空里，最后一点光"])
def test_coloured_or_lit_sky_not_penalised(sky):
    a = _a([_p("P1", [("天空", "scenery")], "蓝色的天空"),
            _p("P2", [("长椅", "object")], "猫睡在长椅上"), STAGE[2]], peak="P3")
    cat = _ai("长椅上，猫睡着", ["P2"])
    assert _pick([_ai(sky, ["P1"]), cat], a=a).text == sky


def test_program_counts_counts_selected_photos():
    a = _a(STAGE, peak="P2", selected=["P2", "P3"])
    assert body.program_counts(a) == {"舞台": 2, "灯光": 1}


# ---------- peak first (user decision 2026-10-31) ----------

def test_peak_candidate_beats_higher_scoring_non_peak():
    a = _a(STAGE, peak="P2")
    strong = _ai("舞台，和蓝色的幕布", ["P3", "P1"])  # rank 0 +2, two photos +1
    weak = _ai("台中央，落下光", ["P2"])            # rank 3: 0
    t1 = body.Candidate(text="5 张舞台，一个周四的夜晚", source="template_T1", photo_ids=["P2", "P3"])
    cands = [strong, _ai("x" * 30), _ai("y" * 30), weak]
    for i, c in enumerate(cands):
        c.rank = i
    pick = body.pick(cands, [t1], a, raw_count=9, when=WHEN)
    assert (pick.text, pick.source) == ("台中央，落下光", "llm")


def test_falls_back_when_no_peak_candidate_survives():
    a = _a(STAGE, peak="P2")
    bad_peak = _ai("台中央，温馨的一束光", ["P2"])  # lexicon: rejected
    other = _ai("长椅上，猫睡着", ["P1"])          # rank 1: +1
    t2 = body.Candidate(text="从长椅，到舞台", source="template_T2", photo_ids=["P1", "P3"])
    cands = [bad_peak, other]
    for i, c in enumerate(cands):
        c.rank = i
    pick = body.pick(cands, [t2], a, raw_count=9, when=WHEN)
    assert dict(pick.rejected)["台中央，温馨的一束光"].startswith("lexicon")
    # other: 1 (rank 1); T2: 1 (two photos) -> tie: AI first
    assert pick.source == "llm" and pick.text == "长椅上，猫睡着"
    pick = body.pick([bad_peak], [], a, raw_count=9, when=WHEN)
    assert pick.source == "fallback_F0"


def test_photo_id_in_any_spelling_rejected():
    # Full check 2026-11-05: only the exact uppercase id was caught.
    for t in ("p1里的光落在台中央", "Ｐ１里的光落在台中央", "P 1 里的光落在台中央",
              "照片1里，光落在台中央", "第一张里，光落在台中央"):
        assert _reason(t) is not None, t


def test_prompt_annotation_echo_is_stripped():
    # Real shorten replies 2026-11-05 echoed the prompt's own line format.
    assert body.clean("红蓝绿菜单摊在木桌（18个字）") == "红蓝绿菜单摊在木桌"
    assert body.clean("蓝灰天色沉入玻璃幕墙（20 个字）｜photo_ids：P1") == "蓝灰天色沉入玻璃幕墙"
    assert body.clean("- 台中央，落下一束光（9 个字）") == "台中央，落下一束光"
    assert body.clean("1. 台中央，落下一束光") == "台中央，落下一束光"


def test_leftover_annotation_is_a_marker_leak():
    assert _reason("台中央的光，18个字") is not None
    # A sign's words are not an annotation.
    assert _reason("招牌上那四个字亮着") is None
    assert _reason("光落在台中央 photo_ids") is not None

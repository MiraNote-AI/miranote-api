"""Detail sentence selection (AI doc 2.1 + 4): one violating case per rule."""

from __future__ import annotations

from typing import List

import body
from check_a import AResult, PhotoFacts, Subject


def _p(pid, subjects=(), details=(), text=(), people="0"):
    return PhotoFacts(
        id=pid,
        subjects=[Subject(n, t) for n, t in subjects],
        details=list(details),
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
    _p("P1", [("长椅", "object"), ("橘猫", "animal")], ["猫睡在长椅上"]),
    _p("P2", [("舞台", "object"), ("灯光", "object")], ["一束光落在台中央"]),
    _p("P3", [("舞台", "object")], ["蓝色的幕布"], ["ABCDEFGHIJKLMNOP"]),
]


def _ai(text, photo_ids=("P1",), uses=()):
    return body.Candidate(text=text, source="llm", photo_ids=list(photo_ids), uses=list(uses))


def _pick(cands, a=None, templates=None, raw_count=23):
    a = a or _a(STAGE, peak="P2")
    for i, c in enumerate(cands):
        c.rank = i
    return body.pick(cands, templates or [], a, body.program_counts(a), raw_count=raw_count)


def _reason(pick, text):
    return dict(pick.rejected).get(text)


def test_char_len_strips_spaces_and_counts_everything_else_once():
    assert body.char_len("菜单上写着 Too Thai") == 12
    assert body.char_len("台中央，落下一束光") == 9


def test_over_16_chars_rejected():
    ok = _ai("ABCDEFGHIJKLMNOP", ["P3"], ["ABCDEFGHIJKLMNOP"])
    long = _ai("ABCDEFGHIJKLMNOPQ", ["P3"], ["ABCDEFGHIJKLMNOP"])
    pick = _pick([long, ok])
    assert _reason(pick, "ABCDEFGHIJKLMNOPQ") == "too_long"
    assert pick.text == "ABCDEFGHIJKLMNOP"


def test_trailing_full_stop_stripped():
    pick = _pick([_ai("  台中央，落下一束光。", ["P2"], ["一束光", "台中央"])])
    assert pick.text == "台中央，落下一束光"
    assert pick.source == "llm"


def test_lexicon_word_rejected():
    pick = _pick([_ai("台中央，温馨的一束光", ["P2"], ["一束光"])])
    assert _reason(pick, "台中央，温馨的一束光").startswith("lexicon")
    assert pick.source == "fallback_F0"


def test_lexicon_word_inside_subject_allowed():
    a = _a([_p("P1", [("雕像", "object")], ["一束光落在雕像前"]), STAGE[1], STAGE[2]], peak="P1")
    pick = _pick([_ai("雕像前，落下一束光", ["P1"], ["雕像", "一束光"])], a=a)
    assert pick.text == "雕像前，落下一束光"


def test_lexicon_word_in_photo_text_allowed():
    a = _a([_p("P1", [("蛋糕", "food")], ["蜡烛插在蛋糕上"], ["生日快乐"]), STAGE[1], STAGE[2]], peak="P1")
    pick = _pick([_ai("蛋糕上写着生日快乐", ["P1"], ["蛋糕", "生日快乐"])], a=a)
    assert pick.text == "蛋糕上写着生日快乐"


def test_hot_steam_allowed():
    a = _a([_p("P1", [("拉面", "food")], ["面上冒着热气"]), STAGE[1], STAGE[2]], peak="P1")
    pick = _pick([_ai("拉面上，冒着热气", ["P1"], ["拉面", "热气"])], a=a)
    assert pick.text == "拉面上，冒着热气"
    rejected = _pick([_ai("拉面上，很热", ["P1"], ["拉面"])], a=a)
    assert _reason(rejected, "拉面上，很热").startswith("lexicon")


def test_ai_uses_not_in_text_rejected():
    pick = _pick([_ai("台中央，落下一束光", ["P2"], ["橘猫"])])
    assert _reason(pick, "台中央，落下一束光") == "uses_not_in_text"


def test_uses_fragment_matches_after_dropping_particles():
    pick = _pick([_ai("台中央，落下一束光", ["P2"], ["落下了一束光"])])
    assert pick.text == "台中央，落下一束光"


def test_unknown_noun_rejected():
    pick = _pick([_ai("台中央，落下一束光和一只狗", ["P2"], ["一束光"])])
    assert _reason(pick, "台中央，落下一束光和一只狗") == "unknown_noun:狗"


def test_people_count_nouns_allowed():
    a = _a([
        _p("P1", [("玻璃楼", "object")], ["站在玻璃楼前"], people="2-4"),
        _p("P2", [("玻璃楼", "object")], ["玻璃楼映着晚霞"]),
        STAGE[2],
    ], peak="P1")
    pick = _pick([_ai("两个人，站在玻璃楼前", ["P1"], ["玻璃楼"])], a=a)
    assert pick.text == "两个人，站在玻璃楼前"


def test_number_not_in_counts_rejected():
    a = _a([_p("P1", [("碗筷", "object")], ["碗筷摆在桌上"]), STAGE[1], STAGE[2]], peak="P1")
    pick = _pick([_ai("桌上，六副碗筷摆好了", ["P1"], ["碗筷"])], a=a)
    assert _reason(pick, "桌上，六副碗筷摆好了") == "number:六"


def test_number_in_details_allowed():
    a = _a([_p("P1", [("碗筷", "object")], ["六副碗筷摆在桌上"]), STAGE[1], STAGE[2]], peak="P1")
    pick = _pick([_ai("桌上，六副碗筷摆好了", ["P1"], ["碗筷"])], a=a)
    assert pick.text == "桌上，六副碗筷摆好了"


def test_number_equal_to_a_program_count_allowed():
    # 舞台 appears in 2 selected photos, so "2" is a program count.
    pick = _pick([_ai("舞台拍了 2 张", ["P2", "P3"], ["舞台"])])
    assert pick.text == "舞台拍了 2 张"
    pick = _pick([_ai("舞台拍了 5 张", ["P2", "P3"], ["舞台"])])
    assert _reason(pick, "舞台拍了 5 张") == "number:5"


def test_lianpai_rejected():
    pick = _pick([_ai("长椅上，连拍了橘猫", ["P1"], ["橘猫", "长椅"])])
    assert _reason(pick, "长椅上，连拍了橘猫") == "lianpai"


def test_t1_only_when_covers_all():
    a = _a(STAGE, peak="P2")
    meta = {"raw_count": 9, "covers_all": False, "photos": [
        {"id": "P1", "represents": 1}, {"id": "P2", "represents": 2}, {"id": "P3", "represents": 3}]}
    assert [c.source for c in body.templates(a, meta)] == ["template_T2"]
    meta["covers_all"] = True
    assert "template_T1" in [c.source for c in body.templates(a, meta)]


def test_t1_needs_subject_in_two_photos_and_sums_represents():
    a = _a(STAGE, peak="P2")
    meta = {"raw_count": 9, "covers_all": True, "photos": [
        {"id": "P1", "represents": 1}, {"id": "P2", "represents": 2}, {"id": "P3", "represents": 3}]}
    t1 = [c for c in body.templates(a, meta) if c.source == "template_T1"]
    # Only 舞台 is in >= 2 selected photos (P2, P3): 2 + 3 = 5.
    assert [c.text for c in t1] == ["那天的舞台，你拍了 5 张"]
    assert t1[0].photo_ids == ["P2", "P3"]


def test_t2_needs_different_first_last_subjects():
    a = _a(STAGE, peak="P2")
    meta = {"raw_count": 9, "covers_all": False, "photos": []}
    t2 = [c for c in body.templates(a, meta) if c.source == "template_T2"]
    assert [c.text for c in t2] == ["从长椅拍到了舞台"]
    same = _a([STAGE[1], STAGE[2]], peak="P2")
    assert [c for c in body.templates(same, meta) if c.source == "template_T2"] == []


def test_t2_not_checked_against_narrative_lexicon():
    text = "从长椅拍到了又一束光"
    t2 = body.Candidate(text=text, source="template_T2", photo_ids=["P1", "P3"], uses=[])
    pick = _pick([], templates=[t2])
    assert pick.text == text and pick.source == "template_T2"
    # The same words written by the model are rejected.
    pick = _pick([_ai(text, ["P1"], ["长椅"])])
    assert _reason(pick, text).startswith("lexicon")


def test_score_prefers_peak_reference():
    first = _ai("长椅上，猫睡着", ["P1"], ["长椅"])
    second = _ai("台中央，落下一束光", ["P2"], ["一束光"])
    pick = _pick([first, second])  # peak is P2
    assert pick.text == "台中央，落下一束光"


def test_tie_prefers_ai_then_shorter():
    filler = [_ai("x" * 20, ["P1"], ["x"]), _ai("y" * 20, ["P1"], ["y"])]  # rank 0, 1: rejected
    longer = _ai("长椅上，猫睡着了", ["P1"], ["长椅"])
    shorter = _ai("长椅上，猫睡着", ["P1"], ["长椅"])
    pick = _pick(filler + [longer, shorter])
    assert pick.text == "长椅上，猫睡着"
    template = body.Candidate(text="长椅上，猫睡", source="template_T2", photo_ids=["P1"], uses=[])
    pick = _pick(filler + [_ai("长椅上，猫睡着", ["P1"], ["长椅"])], templates=[template])
    assert pick.source == "llm"


def test_all_rejected_falls_back_to_f0_with_raw_count():
    pick = _pick([_ai("和朋友度过的温馨午后", ["P1"], ["朋友"])], raw_count=23)
    assert pick.text == "那天你拍了 23 张照片"
    assert pick.source == "fallback_F0"


def test_plain_sky_penalised():
    a = _a([_p("P1", [("天空", "scenery")], ["天空里有云"]),
            _p("P2", [("长椅", "object")], ["猫睡在长椅上"]), STAGE[2]], peak="P3")
    sky = _ai("那天的天空", ["P1"], ["天空"])
    cat = _ai("长椅上，猫睡着", ["P2"], ["长椅"])
    pick = _pick([sky, cat], a=a)
    assert pick.text == "长椅上，猫睡着"


def test_coloured_sky_not_penalised():
    a = _a([_p("P1", [("天空", "scenery")], ["蓝色的天空"]),
            _p("P2", [("长椅", "object")], ["猫睡在长椅上"]), STAGE[2]], peak="P3")
    sky = _ai("蓝色的天空", ["P1"], ["天空"])
    cat = _ai("长椅上，猫睡着", ["P2"], ["长椅"])
    assert _pick([sky, cat], a=a).text == "蓝色的天空"


def test_program_counts_counts_selected_photos():
    a = _a(STAGE, peak="P2", selected=["P2", "P3"])
    assert body.program_counts(a) == {"舞台": 2, "灯光": 1}

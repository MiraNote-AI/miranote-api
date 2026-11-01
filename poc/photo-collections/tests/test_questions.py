"""Question bank, conditions, menus and the pick (题库-V1.md, AI doc 2.2 + 5)."""

from __future__ import annotations

from typing import List

import questions
from check_a import AResult, PhotoFacts, Subject


def _p(pid, subjects=(), details=(), people="0", face=False, scene="park_nature"):
    return PhotoFacts(
        id=pid,
        subjects=[Subject(n, t) for n, t in subjects],
        details=list(details),
        people_count=people,
        face_prominent=face,
        scene=scene,
    )


def _a(photos: List[PhotoFacts], peak=None) -> AResult:
    ids = [p.id for p in photos]
    return AResult(photos={p.id: p for p in photos}, selected=ids, peak=peak or ids[-1],
                   verdict="ok", order=ids)


META = {"start_time": "14:03", "end_time": "17:40", "recent_question_ids": []}

DINNER = [
    _p("P1", [("拿铁", "drink"), ("桌子", "object")], ["拉花是一片叶子"], scene="restaurant_cafe"),
    _p("P2", [("拉面", "food")], ["面上冒着热气"], scene="restaurant_cafe"),
    _p("P3", [("饺子", "food"), ("橘猫", "animal")], ["猫趴在窗台上"], scene="restaurant_cafe"),
    _p("P4", [("晚霞", "scenery"), ("天空", "scenery")], ["橙红色的云"], scene="street_city"),
]


def _cands(a=None, meta=None, daypart="afternoon"):
    return questions.candidates(a or _a(DINNER, peak="P3"), meta or META, daypart)


def _by_id(cands):
    return {c.id: c for c in cands}


def test_bank_has_6_a_and_19_b():
    bank = questions.bank()
    assert sum(q["kind"] == "A" for q in bank) == 6
    assert sum(q["kind"] == "B" for q in bank) == 19
    assert len({q["id"] for q in bank}) == 25


def test_recent_ids_not_offered():
    meta = dict(META, recent_question_ids=["B4", "A2"])
    ids = _by_id(_cands(meta=meta))
    assert "B4" not in ids and "A2" not in ids
    assert "B5" in ids


def test_empty_menu_hides_a_question():
    a = _a([_p("P1", [("长椅", "object")], ["蓝色的幕布"]),
            _p("P2", [("长椅", "object")], []),
            _p("P3", [("舞台", "object")], [])])
    ids = _by_id(_cands(a=a))
    assert "A1" not in ids  # no detail with an action verb
    assert "A5" not in ids  # no food / drink
    assert "A2" in ids


def test_a5_menu_only_food_drink():
    assert _by_id(_cands())["A5"].menu == ["饺子", "拿铁", "拉面"]  # peak P3 first


def test_a2_menu_drops_common_objects():
    menu = _by_id(_cands())["A2"].menu
    assert "桌子" not in menu and "天空" not in menu
    assert menu == ["橘猫", "晚霞"]
    assert "桌子" in _by_id(_cands())["A3"].menu


def test_a6_menu_drops_person():
    a = _a([_p("P1", [("女孩", "person"), ("气球", "object")], ["气球飘在空中"]),
            _p("P2", [("舞台", "object")]), _p("P3", [("舞台", "object")])], peak="P1")
    assert _by_id(_cands(a=a))["A6"].menu == ["气球", "舞台"]


def test_scene_condition_needs_half():
    # 3 of 4 selected in restaurant_cafe: B19 offered.
    assert "B19" in _by_id(_cands())
    half = [DINNER[0], DINNER[1], _p("P3", [("湖面", "scenery")], scene="waterside"),
            _p("P4", [("树林", "scenery")], scene="mountain")]
    assert "B19" in _by_id(_cands(a=_a(half)))  # exactly half
    under = half[:1] + [_p("P2", [("湖面", "scenery")], scene="waterside")] + half[2:]
    assert "B19" not in _by_id(_cands(a=_a(under)))


def test_b16_needs_group_face():
    assert "B16" not in _by_id(_cands())
    no_face = DINNER[:3] + [_p("P4", [("合影", "object")], people="2-4", face=False)]
    assert "B16" not in _by_id(_cands(a=_a(no_face)))
    group = DINNER[:3] + [_p("P4", [("合影", "object")], people="5+", face=True)]
    assert "B16" in _by_id(_cands(a=_a(group)))


def test_people_conditions():
    assert "B17" not in _by_id(_cands())
    assert "A4" not in _by_id(_cands())
    one = DINNER[:3] + [_p("P4", [("女孩", "person")], ["女孩站在桥上"], people="1")]
    ids = _by_id(_cands(a=_a(one)))
    assert "B17" in ids and "A4" in ids


def test_b12_not_at_night():
    assert "B12" in _by_id(_cands(daypart="afternoon"))
    assert "B12" not in _by_id(_cands(daypart="night"))
    short = dict(META, start_time="14:03", end_time="14:50")
    assert "B12" not in _by_id(_cands(meta=short))
    assert "B12" not in _by_id(_cands(a=_a(DINNER[:3])))


def test_b15_needs_two_food_photos():
    assert "B15" in _by_id(_cands())
    assert "B15" not in _by_id(_cands(a=_a([DINNER[0], DINNER[1], DINNER[3]])))


def test_hua_mian_menu_needs_action_verb():
    menu = _by_id(_cands())["A1"].menu
    assert menu == ["猫趴在窗台上"]  # 趴 is an action verb; the others have none


def test_daypart_filled_in_offered_text():
    assert _by_id(_cands(daypart="evening"))["B4"].text == "如果那个傍晚能按暂停，你会停在哪一秒？"


def test_prompt_block_lines():
    block = questions.prompt_block(_cands())
    lines = block.splitlines()
    assert "A2｜{物}哪一点，让你停下来拍它？｜菜单：橘猫, 晚霞" in lines
    assert "B5｜在这张照片背面，你会写一句什么？" in lines


# ---------- pick (AI doc 5) ----------

def _pick(ranked, cands=None, recent=(), daypart_name="下午"):
    return questions.pick(ranked, cands if cands is not None else _cands(), list(recent), daypart_name)


def _r(qid, fill=""):
    return {"id": qid, "fill": fill, "why": "贴切"}


def test_valid_a_question_is_used():
    q = _pick([_r("A5", "拉面")])
    assert (q.text, q.id, q.source) == ("拉面，下次还会点吗？", "A5", "llm")


def test_unknown_id_rejected():
    q = _pick([_r("Z9"), _r("B5")])
    assert dict(q.rejected)["Z9"] == "not_offered"
    assert q.id == "B5"


def test_recent_not_offered_id_rejected():
    meta = dict(META, recent_question_ids=["B4"])
    q = _pick([_r("B4"), _r("B5")], cands=_cands(meta=meta))
    assert dict(q.rejected)["B4"] == "not_offered"


def test_fill_not_in_menu_rejected():
    q = _pick([_r("A5", "牛排"), _r("B5")])
    assert dict(q.rejected)["A5"] == "fill_not_in_menu"


def test_b_with_fill_rejected():
    q = _pick([_r("B5", "拉面"), _r("B6")])
    assert dict(q.rejected)["B5"] == "b_with_fill"
    assert q.id == "B6"


def test_over_22_chars_rejected():
    cands = _cands()
    a1 = [c for c in cands if c.id == "A1"][0]
    a1.menu.append("一只橘猫趴在洒满阳光的窗台上睡觉")
    q = _pick([_r("A1", "一只橘猫趴在洒满阳光的窗台上睡觉"), _r("B5")], cands=cands)
    assert dict(q.rejected)["A1"] == "too_long"


def test_forbidden_word_rejected():
    cands = _cands()
    a5 = [c for c in cands if c.id == "A5"][0]
    a5.menu.append("又一碗面")
    q = _pick([_r("A5", "又一碗面"), _r("B5")], cands=cands)
    assert dict(q.rejected)["A5"] == "forbidden:又"


def test_recent_id_rejected():
    # Offered by mistake but used recently: still rejected.
    q = _pick([_r("B5"), _r("B6")], recent=["B5"])
    assert dict(q.rejected)["B5"] == "recent"
    assert q.id == "B6"


def test_second_rank_used_when_first_fails():
    q = _pick([_r("A5", "牛排"), _r("A2", "橘猫"), _r("B5")])
    assert q.id == "A2" and q.text == "橘猫哪一点，让你停下来拍它？"


def test_only_top_3_considered():
    q = _pick([_r("Z1"), _r("Z2"), _r("Z3"), _r("B5")])
    assert q.source == "fallback_B1"


def test_all_fail_falls_back_b1_then_b2_skipping_recent():
    q = _pick([_r("Z1")])
    assert (q.id, q.source) == ("B1", "fallback_B1")
    q = _pick([], recent=["B1"])
    assert (q.id, q.source) == ("B2", "fallback_B2")
    q = _pick([], recent=["B1", "B2"])
    assert (q.id, q.source) == ("B3", "fallback_B3")


def test_b1_text_fills_daypart():
    q = _pick([], daypart_name="下午")
    assert q.text == "那个下午的三个关键词：＿＿、＿＿、＿＿"

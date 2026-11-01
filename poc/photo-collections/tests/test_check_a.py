"""Program check A (AI doc 1.4): one deliberately violating fixture per rule."""

from __future__ import annotations

import copy
import json

import pytest

import check_a

IDS = ["P1", "P2", "P3", "P4", "P5", "P6"]


def _photo(pid, subjects=None, flags=None, keep=True, drop_reason="", people="0",
           face=False, scene="park_nature", details=None, text=None):
    return {
        "id": pid,
        "subjects": subjects if subjects is not None else [{"name": "长椅", "type": "object"}],
        "details": details if details is not None else ["猫睡在长椅上"],
        "text": text if text is not None else [],
        "people": {"count": people, "face_prominent": face},
        "scene": scene,
        "flags": flags if flags is not None else [],
        "keep": keep,
        "drop_reason": drop_reason,
    }


def _raw(**over):
    raw = {
        "photos": [_photo(pid) for pid in IDS],
        "selected_ids": ["P1", "P3", "P5", "P6"],
        "peak_id": "P5",
        "skip": False,
        "skip_reason": "",
    }
    raw.update(over)
    return raw


def test_valid_output_passes():
    a = check_a.validate(_raw(), IDS)
    assert a.verdict == "ok"
    assert a.selected == ["P1", "P3", "P5", "P6"]
    assert a.peak == "P5"
    assert a.photos["P1"].subjects[0].name == "长椅"
    assert a.photos["P1"].subjects[0].type == "object"


def test_unparseable_json_fails():
    with pytest.raises(check_a.CheckFailed):
        check_a.parse_json("这不是 JSON")
    with pytest.raises(check_a.CheckFailed):
        check_a.parse_json('{"photos": [')


def test_fenced_json_is_parsed():
    text = "好的，结果如下：\n```json\n" + json.dumps(_raw(), ensure_ascii=False) + "\n```\n以上。"
    assert check_a.parse_json(text)["peak_id"] == "P5"


def test_prose_around_json_is_parsed():
    text = "结果：" + json.dumps(_raw(), ensure_ascii=False) + " 完毕"
    assert check_a.parse_json(text)["selected_ids"] == ["P1", "P3", "P5", "P6"]


def test_chinese_quotes_are_repaired():
    text = '{"skip_reason": “都是截图”, "peak_id": "P5"}'
    assert check_a.parse_json(text)["skip_reason"] == "都是截图"


def test_missing_photo_id_fails():
    raw = _raw()
    raw["photos"] = raw["photos"][:-1]
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(raw, IDS)


def test_extra_photo_id_fails():
    raw = _raw()
    raw["photos"].append(_photo("P7"))
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(raw, IDS)


def test_duplicate_photo_entry_fails():
    raw = _raw()
    raw["photos"][1] = _photo("P1")
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(raw, IDS)


def test_selected_unknown_id_fails():
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(_raw(selected_ids=["P1", "P3", "P9"], peak_id="P1"), IDS)


def test_selected_dropped_photo_fails():
    raw = _raw()
    raw["photos"][2] = _photo("P3", keep=False, drop_reason="duplicate")
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(raw, IDS)


def test_selected_more_than_8_fails():
    ids = ["P{}".format(i) for i in range(1, 11)]
    raw = _raw(photos=[_photo(pid) for pid in ids], selected_ids=ids[:9], peak_id="P2")
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(raw, ids)


def test_selected_8_is_fine():
    ids = ["P{}".format(i) for i in range(1, 11)]
    raw = _raw(photos=[_photo(pid) for pid in ids], selected_ids=ids[:8], peak_id="P2")
    assert check_a.validate(raw, ids).verdict == "ok"


def test_peak_outside_selected_fails():
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(_raw(peak_id="P2"), IDS)


def test_flagged_peak_is_replaced_by_most_subjects_then_latest():
    raw = _raw()
    two = [{"name": "长椅", "type": "object"}, {"name": "橘猫", "type": "animal"}]
    raw["photos"][4] = _photo("P5", flags=["unrecognizable"])
    raw["photos"][0] = _photo("P1", subjects=two)
    raw["photos"][2] = _photo("P3", subjects=two)
    a = check_a.validate(raw, IDS)
    assert a.verdict == "ok"
    assert a.peak == "P3"  # P1 and P3 both have 2 subjects; P3 is later
    assert a.peak_replaced_from == "P5"


def test_all_selected_flagged_is_not_pushed():
    raw = _raw()
    for i, pid in enumerate(IDS):
        raw["photos"][i] = _photo(pid, flags=["screenshot"])
    assert check_a.validate(raw, IDS).verdict == "skip_screenshots"
    raw["photos"][0] = _photo("P1", flags=["screenshot", "injury"])
    # injury on a selected photo: sensitive wins
    assert check_a.validate(raw, IDS).verdict == "sensitive"


@pytest.mark.parametrize("flag", ["hospital", "funeral", "accident", "injury", "nudity"])
def test_sensitive_flag_on_selected_rejects_group(flag):
    raw = _raw()
    raw["photos"][2] = _photo("P3", flags=[flag])
    assert check_a.validate(raw, IDS).verdict == "sensitive"


def test_sensitive_flag_on_unselected_photo_does_not_reject():
    raw = _raw()
    raw["photos"][1] = _photo("P2", flags=["hospital"], keep=False, drop_reason="sensitive")
    assert check_a.validate(raw, IDS).verdict == "ok"


def test_skip_true_rejects():
    a = check_a.validate(_raw(skip=True, skip_reason="大多是讲义"), IDS)
    assert a.verdict == "skip_screenshots"


def test_bad_subject_names_dropped_not_truncated():
    subjects = [
        {"name": "花", "type": "scenery"},          # 1 char
        {"name": "玻璃幕墙大楼", "type": "object"},   # 6 chars
        {"name": "红灯笼串", "type": "object"},       # 4 chars: kept
        {"name": " 橘 猫 ", "type": "animal"},        # whitespace stripped: kept
        {"name": "雕像", "type": "statue"},           # type outside enum
    ]
    raw = _raw()
    raw["photos"][0] = _photo("P1", subjects=subjects)
    a = check_a.validate(raw, IDS)
    assert [(s.name, s.type) for s in a.photos["P1"].subjects] == [
        ("红灯笼串", "object"),
        ("橘猫", "animal"),
    ]


def test_selected_under_3_after_dedupe():
    a = check_a.validate(_raw(selected_ids=["P1", "P5", "P5", "P1"], peak_id="P5"), IDS)
    assert a.verdict == "too_few_selected"
    assert a.selected == ["P1", "P5"]


def test_selected_sorted_by_input_order():
    a = check_a.validate(_raw(selected_ids=["P6", "P1", "P5", "P3"]), IDS)
    assert a.selected == ["P1", "P3", "P5", "P6"]


def test_people_and_scene_are_normalised():
    raw = _raw()
    raw["photos"][0] = _photo("P1", people=3, scene="moon_base")
    a = check_a.validate(raw, IDS)
    assert a.photos["P1"].people_count == "2-4"
    assert a.photos["P1"].scene == "other"


def test_not_a_dict_fails():
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(["P1"], IDS)
    raw = copy.deepcopy(_raw())
    raw["photos"][0] = "P1"
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(raw, IDS)


def test_skip_true_with_empty_selection_rejects():
    # Seen on the real model with the lecture-slides group: skip=true and
    # nothing selected is a valid answer, not a malformed one.
    raw = _raw(skip=True, skip_reason="都是幻灯片", selected_ids=[], peak_id="")
    for i, pid in enumerate(IDS):
        raw["photos"][i] = _photo(pid, flags=["document"], keep=False, drop_reason="memo")
    assert check_a.validate(raw, IDS).verdict == "skip_screenshots"


def test_skip_true_with_sensitive_flag_is_sensitive():
    raw = _raw(skip=True, selected_ids=[], peak_id="")
    raw["photos"][0] = _photo("P1", flags=["funeral"], keep=False, drop_reason="sensitive")
    assert check_a.validate(raw, IDS).verdict == "sensitive"


def test_skip_true_still_needs_every_photo_described():
    raw = _raw(skip=True, selected_ids=[], peak_id="")
    raw["photos"] = raw["photos"][:-1]
    with pytest.raises(check_a.CheckFailed):
        check_a.validate(raw, IDS)


def test_chinese_quotes_repaired_inside_objects_and_arrays():
    text = '{"subjects": [{"name":“拿铁”,"type":“drink”}]}'
    assert check_a.parse_json(text)["subjects"] == [{"name": "拿铁", "type": "drink"}]
    text = '{"text": [“x”, “y”]}'
    assert check_a.parse_json(text)["text"] == ["x", "y"]
    text = '{"details": [“猫睡在长椅上” ], "keep": true}'
    assert check_a.parse_json(text)["details"] == ["猫睡在长椅上"]

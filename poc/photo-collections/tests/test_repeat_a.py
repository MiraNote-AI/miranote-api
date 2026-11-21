"""Repeat-theme compose: meta checks, call A and the days rule."""

from __future__ import annotations

import json
import time

import pytest

import check_a
import compose
import repeat
from tests.test_api import _poll, _submit
from tests.test_compose import _log, _meta, _photo, _texts, jpeg

THEME = {"id": "T1", "name": "咖啡", "noun": "咖啡", "same_part": "桌上的一杯咖啡", "where": None}


def _repeat_meta(n=6, **over):
    photos = [{"id": "P{}".format(i), "taken_at": "2026-10-{:02d}T09:10:00-05:00".format(i),
               "labels": ["drink"], "faces": 0, "favorite": False, "edited": False}
              for i in range(1, n + 1)]
    meta = {"mode": "repeat", "group_key": "theme-T1", "tz": "America/Chicago", "theme": dict(THEME),
            "recent_question_types": [], "photos": photos}
    meta.update(over)
    return meta


def _files(n):
    return [("photos", ("P{}.jpg".format(i), jpeg(color=(30 * i, 90, 60)), "image/jpeg"))
            for i in range(1, n + 1)]


def _images(n):
    return {"P{}".format(i): jpeg(color=(30 * i, 90, 60)) for i in range(1, n + 1)}


def _b_repeat():
    return json.dumps({"scene": "早上的一杯咖啡", "cannot_write": False,
                       "candidates": [{"text": "端上来那一刻，先拍了再喝", "style": "轻松",
                                       "photo_ids": ["P3"]}],
                       "question": {"ranked": [{"text": "下一杯咖啡，想点什么？", "type": "next_time",
                                                "why": "下次"}]}}, ensure_ascii=False)


def _a_repeat(n=6, bad=(), peak="P3", **over):
    photos = []
    for i in range(1, n + 1):
        pid = "P{}".format(i)
        kw = {}
        if pid in bad:
            kw = dict(keep=False, drop="unclear")
        photos.append(_photo(pid, [("咖啡", "drink")], "桌上一杯咖啡，光从窗边照进来。", **kw))
    raw = {"photos": photos, "peak_id": peak, "skip": False, "skip_reason": ""}
    raw.update(over)
    return json.dumps(raw, ensure_ascii=False)


def _run(n=6, meta=None):
    return compose.run(meta or _repeat_meta(n), _images(n), time.monotonic() + 300)


def test_repeat_meta_accepted_without_life_fragment_fields(client, fake_model):
    fake_model.reply("a", _a_repeat()).reply("b", _b_repeat())
    r = _submit(client, _repeat_meta(6), _files(6))
    assert r.status_code == 202
    done = _poll(client, r.json()["job_id"]).json()
    assert done["status"] == "ok" and done["memory"]["mode"] == "repeat"


def test_repeat_meta_accepts_empty_noun_and_same_part(client, fake_model):
    meta = _repeat_meta(6)
    meta["theme"].update(noun="", same_part="")
    fake_model.reply("a", _a_repeat()).reply("b", _b_repeat())
    r = _submit(client, meta, _files(6))
    assert r.status_code == 202
    done = _poll(client, r.json()["job_id"]).json()
    assert done["status"] == "ok"
    assert done["memory"]["ending"]["ctaLabel"] == "留住这些瞬间"
    a_text = "\n".join(_texts(fake_model.calls[0]["messages"][1]["content"]))
    assert "（无）" in a_text


@pytest.mark.parametrize("key", ["id", "name"])
def test_repeat_meta_rejects_empty_theme_id_or_name(client, key):
    meta = _repeat_meta(6)
    meta["theme"][key] = " "
    assert _submit(client, meta, _files(6)).status_code == 400


@pytest.mark.parametrize("n", [4, 10])
def test_repeat_meta_rejects_4_or_10_photos(client, n):
    assert _submit(client, _repeat_meta(n), _files(n)).status_code == 400


def test_repeat_meta_rejects_unknown_mode(client):
    assert _submit(client, _repeat_meta(6, mode="nope"), _files(6)).status_code == 400


def test_repeat_meta_rejects_null_mode(client):
    assert _submit(client, _repeat_meta(6, mode=None), _files(6)).status_code == 400


def test_repeat_a_skip_for_sensitive_reason_gives_sensitive():
    ids = ["P{}".format(i) for i in range(1, 7)]
    days = {pid: "2026-10-{:02d}".format(i) for i, pid in enumerate(ids, 1)}
    raw = json.loads(_a_repeat())
    raw["photos"][0]["flags"] = ["hospital"]
    raw["skip"] = True
    assert check_a.validate_repeat(raw, ids, days).verdict == "sensitive"


def test_repeat_meta_rejects_bad_theme(client):
    meta = _repeat_meta(6)
    del meta["theme"]["same_part"]
    assert _submit(client, meta, _files(6)).status_code == 400


def test_repeat_meta_rejects_bad_where(client):
    meta = _repeat_meta(6)
    meta["theme"]["where"] = {"weekpart": "someday"}
    assert _submit(client, meta, _files(6)).status_code == 400


def test_compose_mode_absent_or_life_fragment_unchanged(fake_model, tmp_log):
    from tests.test_compose import IMAGES, _a_json, _b_json
    outs = []
    for meta in (_meta(), _meta(mode="life_fragment")):
        fake_model.reply("a", _a_json()).reply("b", _b_json())
        out = compose.run(meta, dict(IMAGES), time.monotonic() + 300)
        assert out["status"] == "ok"
        memory = dict(out["memory"])
        memory.pop("generation")  # latency differs run to run
        outs.append(json.dumps(memory, sort_keys=True, ensure_ascii=False))
    assert outs[0] == outs[1]


def test_repeat_a_prompt_has_theme_and_no_selection_task(fake_model, tmp_log):
    fake_model.reply("a", _a_repeat()).reply("b", _b_repeat())
    _run()
    content = fake_model.calls[0]["messages"][1]["content"]
    text = "\n".join(_texts(content))
    assert "咖啡" in text and "桌上的一杯咖啡" in text
    assert "精选" not in text
    assert "selected_ids" not in text
    assert text.count("peak_id") >= 1


def test_repeat_a_prompt_skips_only_when_mostly_screenshots_or_sensitive(fake_model, tmp_log):
    fake_model.reply("a", _a_repeat()).reply("b", _b_repeat())
    _run()
    text = "\n".join(_texts(fake_model.calls[0]["messages"][1]["content"]))
    assert "只有这组大多是截图、资料，或者大多是敏感内容时，才 skip=true" in text
    assert "只有一两张碰到敏感场景，就给它标上 flags、keep=false、drop_reason 写 sensitive" in text
    assert "大多来自不同的日子" in text


def test_repeat_a_ok_goes_on_to_b_and_c(fake_model, tmp_log):
    fake_model.reply("a", _a_repeat()).reply("b", _b_repeat())
    out = _run()
    assert out["status"] == "ok"
    assert fake_model.count("b") == 1 and fake_model.count("c") == 1
    assert "stage" not in _log(tmp_log)


def test_repeat_a_drops_below_five_days_rejects(fake_model, tmp_log):
    fake_model.reply("a", _a_repeat(bad=("P1", "P2")))
    out = _run()
    assert (out["status"], out["reason"]) == ("rejected", "too_few_days")
    assert fake_model.count("b") == 0


def test_repeat_a_drops_sensitive_photo_keeps_rest(fake_model, tmp_log):
    raw = json.loads(_a_repeat(n=7))
    raw["photos"][0]["flags"] = ["hospital"]
    ids = ["P{}".format(i) for i in range(1, 8)]
    days = {pid: "2026-10-{:02d}".format(i) for i, pid in enumerate(ids, 1)}
    a = check_a.validate_repeat(raw, ids, days)
    assert a.verdict == "ok"
    assert "P1" not in a.selected and a.selected == ids[1:]


def test_repeat_a_peak_is_models_choice():
    ids = ["P{}".format(i) for i in range(1, 7)]
    days = {pid: "2026-10-{:02d}".format(i) for i, pid in enumerate(ids, 1)}
    a = check_a.validate_repeat(json.loads(_a_repeat(peak="P5")), ids, days)
    assert a.peak == "P5" and a.peak_replaced_from is None


def test_repeat_a_dropped_peak_is_replaced():
    ids = ["P{}".format(i) for i in range(1, 8)]
    days = {pid: "2026-10-{:02d}".format(i) for i, pid in enumerate(ids, 1)}
    a = check_a.validate_repeat(json.loads(_a_repeat(n=7, bad=("P3",), peak="P3")), ids, days)
    assert a.peak != "P3" and a.peak in a.selected and a.peak_replaced_from == "P3"


def test_repeat_a_same_day_photos_count_once():
    ids = ["P{}".format(i) for i in range(1, 7)]
    days = {pid: "2026-10-01" for pid in ids}
    a = check_a.validate_repeat(json.loads(_a_repeat()), ids, days)
    assert a.verdict == "too_few_days"


def test_repeat_days_use_local_date_as_sent():
    photos = [{"id": "P1", "taken_at": "2026-10-18T23:30:00-05:00"}]
    assert repeat.local_days(photos) == {"P1": "2026-10-18"}

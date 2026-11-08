"""Call C's reply (copy refine handoff 2.4): per sentence and question,
booleans; an unsure answer follows decision 2026-11-04 item 3."""

from __future__ import annotations

import json

import pytest

import check_a
import verify


def _reply(sentences, questions):
    return json.dumps({"sentences": sentences, "questions": questions}, ensure_ascii=False)


def test_parse_reads_every_verdict():
    out = verify.parse(_reply(
        [{"id": "S1", "grounded": True, "mentions_cover": False, "reason": ""},
         {"id": "S2", "grounded": False, "mentions_cover": True, "reason": "没有推车"}],
        [{"id": "Q1", "grounded": True, "safe": False, "reason": "假设有人陪"}]), 2, 1)
    assert (out.sentences["S1"].grounded, out.sentences["S1"].mentions_cover) == (True, False)
    assert (out.sentences["S2"].grounded, out.sentences["S2"].reason) == (False, "没有推车")
    assert (out.questions["Q1"].safe, out.questions["Q1"].reason) == (False, "假设有人陪")


def test_missing_verdict_drops_only_that_item():
    # Full check 2026-11-05: one missing item used to throw the whole reply
    # away (templates only). Now only that item is unchecked, so dropped.
    out = verify.parse(_reply([{"id": "S1", "grounded": True, "mentions_cover": True}], []), 2, 0)
    assert out.sentences["S1"].grounded is True
    assert out.sentences["S2"].grounded is False


def test_nothing_answered_fails_the_check():
    with pytest.raises(check_a.CheckFailed):
        verify.parse(_reply([], []), 0, 1)
    with pytest.raises(check_a.CheckFailed):
        verify.parse(_reply([{"id": "S9", "grounded": True}], []), 1, 0)


def test_unsure_follows_the_user_rule():
    # Decision 2026-11-04 item 3: unsure about made-up content passes,
    # unsure about safety fails. A non-boolean answer is "unsure".
    out = verify.parse(_reply(
        [{"id": "S1", "grounded": "unsure", "mentions_cover": None}],
        [{"id": "Q1", "grounded": None, "safe": "unsure"}]), 1, 1)
    assert out.sentences["S1"].grounded is True
    assert out.sentences["S1"].mentions_cover is False
    assert (out.questions["Q1"].grounded, out.questions["Q1"].safe) == (True, False)


def test_string_booleans_and_loose_ids_are_read():
    out = verify.parse(_reply(
        [{"id": 1, "grounded": "false", "mentions_cover": "true"}],
        [{"id": "q1", "grounded": "True", "safe": "true"}]), 1, 1)
    assert (out.sentences["S1"].grounded, out.sentences["S1"].mentions_cover) == (False, True)
    assert (out.questions["Q1"].grounded, out.questions["Q1"].safe) == (True, True)


def test_nothing_to_check_accepts_empty_lists():
    out = verify.parse('{"sentences": [], "questions": []}', 0, 0)
    assert out.sentences == {} and out.questions == {}


def test_user_text_numbers_sentences_and_questions():
    photo_tpl, task = verify.user_text(["天色暗下去，台上的光亮起来"], ["推车上，卖的是什么？"])
    assert photo_tpl == "{id}{cover}"
    assert "S1｜天色暗下去，台上的光亮起来" in task
    assert "Q1｜推车上，卖的是什么？" in task
    assert "拿不准时：grounded 和 fluent 算通过，safe 算不通过。" in task
    assert '{"sentences": [{"id": "S1"' in task  # literal braces survive format_map


def test_fluent_is_read_and_unsure_passes():
    # Decision 2026-11-05 evening item 7: only plainly broken sentences fail.
    out = verify.parse(_reply(
        [{"id": "S1", "grounded": True, "mentions_cover": True, "fluent": False, "reason": "读不通"},
         {"id": "S2", "grounded": True, "mentions_cover": True, "fluent": "unsure"},
         {"id": "S3", "grounded": True, "mentions_cover": True}], []), 3, 0)
    assert [out.sentences[s].fluent for s in ("S1", "S2", "S3")] == [False, True, True]

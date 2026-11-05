"""Call C's reply: every sentence and question answered, booleans only
(copy refine handoff 2.4)."""

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


def test_missing_verdict_fails_the_check():
    with pytest.raises(check_a.CheckFailed, match="S2"):
        verify.parse(_reply([{"id": "S1", "grounded": True, "mentions_cover": True}], []), 2, 0)
    with pytest.raises(check_a.CheckFailed, match="Q1"):
        verify.parse(_reply([], []), 0, 1)


def test_non_boolean_fails_the_check():
    with pytest.raises(check_a.CheckFailed, match="grounded"):
        verify.parse(_reply([{"id": "S1", "grounded": "yes", "mentions_cover": True}], []), 1, 0)


def test_nothing_to_check_accepts_empty_lists():
    out = verify.parse('{"sentences": [], "questions": []}', 0, 0)
    assert out.sentences == {} and out.questions == {}


def test_user_text_numbers_sentences_and_questions():
    photo_tpl, task = verify.user_text(["天色暗下去，台上的光亮起来"], ["推车上，卖的是什么？"])
    assert photo_tpl == "{id}{cover}"
    assert "S1｜天色暗下去，台上的光亮起来" in task
    assert "Q1｜推车上，卖的是什么？" in task
    assert "拿不准时：grounded 算通过，safe 算不通过。" in task
    assert '{"sentences": [{"id": "S1"' in task  # literal braces survive format_map

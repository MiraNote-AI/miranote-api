from __future__ import annotations
import pytest


def _candidates(n: int = 3):
    return [
        {"id": f"c{i}", "text": f"text {i}", "author": f"a{i}",
         "source": f"s{i}", "lang": "en", "score": 0.8 - 0.1 * i}
        for i in range(n)
    ]


def test_rerank_returns_picks(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[{"id": 2, "why": "second matches mood"}, {"id": 1, "why": "first works too"}]')
    out = rerank(fake_llm, "fake-model", "any user text", _candidates(3), max_picks=3)
    assert out == [
        {"index": 2, "why": "second matches mood"},
        {"index": 1, "why": "first works too"},
    ]


def test_rerank_empty_array_is_valid(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[]')
    out = rerank(fake_llm, "fake-model", "totally off-topic", _candidates(3), max_picks=3)
    assert out == []


def test_rerank_malformed_json_raises_valueerror(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('not even close to JSON')
    with pytest.raises(ValueError, match="invalid JSON"):
        rerank(fake_llm, "fake-model", "x", _candidates(3))


def test_rerank_out_of_range_index_raises_valueerror(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[{"id": 99, "why": "..."}]')
    with pytest.raises(ValueError, match="out of range"):
        rerank(fake_llm, "fake-model", "x", _candidates(3))


def test_rerank_missing_keys_raises_valueerror(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[{"wrong_key": 1}]')
    with pytest.raises(ValueError, match="unexpected schema"):
        rerank(fake_llm, "fake-model", "x", _candidates(3))

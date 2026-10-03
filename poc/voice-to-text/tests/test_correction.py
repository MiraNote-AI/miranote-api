"""Offline coverage for the correction module and the benchmark's scoring.

No network. The point is that the refactor that pulled correction.py out of
main.py did not change the request production sends, and that the bench's
cost arithmetic and flags are right before any money is spent on a run.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

POC = Path(__file__).resolve().parent.parent
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

import bench_correction  # noqa: E402
import correction  # noqa: E402

# Two Han characters, built from code points so this file stays ASCII
# (org Rule 3) while still exercising the CJK detection it tests.
ZH = chr(0x4E2D) + chr(0x6587)


# --------------------------------------------------------------------------- #
# Fake OpenAI client
# --------------------------------------------------------------------------- #
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    def __init__(self, pt, ct):
        self.prompt_tokens = pt
        self.completion_tokens = ct


class _Resp:
    def __init__(self, content, pt=100, ct=50, usage=True):
        self.choices = [_Choice(content)]
        self.usage = _Usage(pt, ct) if usage else None


class _Completions:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeClient:
    def __init__(self, outcome):
        self.chat = type("C", (), {"completions": _Completions(outcome)})()

    @property
    def calls(self):
        return self.chat.completions.calls


# --------------------------------------------------------------------------- #
# correction.py
# --------------------------------------------------------------------------- #
def test_prompt_loaded_from_file():
    assert correction.CORRECTION_PROMPT, "prompts/correction.txt did not load"


def test_build_messages_matches_the_original_request_shape():
    """main.py sent exactly one user turn, prompt and text joined by a blank
    line, with no system role. Changing this changes every correction."""
    msgs = correction.build_messages("hello")
    assert msgs == [{
        "role": "user",
        "content": correction.CORRECTION_PROMPT + "\n\n" + "hello",
    }]


def test_correct_once_sends_model_and_max_tokens():
    client = FakeClient(_Resp("fixed text"))
    result = correction.correct_once("raw", client, "some-model")
    assert result.status == "ok"
    assert result.text == "fixed text"
    assert result.prompt_tokens == 100 and result.completion_tokens == 50
    assert result.latency_ms >= 0
    sent = client.calls[0]
    assert sent["model"] == "some-model"
    assert sent["max_tokens"] == correction.DEFAULT_MAX_TOKENS == 4096
    # No extra_body unless a caller asks for one -- the baseline must keep
    # sending the exact body production has always sent.
    assert "extra_body" not in sent


def test_extra_body_is_passed_through_for_qwen():
    client = FakeClient(_Resp("x"))
    correction.correct_once("raw", client, "qwen-flash",
                            extra_body={"enable_thinking": False})
    assert client.calls[0]["extra_body"] == {"enable_thinking": False}


def test_failure_is_returned_not_raised():
    client = FakeClient(RuntimeError("Error code: 429 - rate limited"))
    result = correction.correct_once("raw", client, "m")
    assert result.status == "failed"
    assert result.text is None
    assert "429" in result.error
    assert correction.is_rate_limited(result.error)


def test_non_retryable_error_is_not_rate_limited():
    client = FakeClient(RuntimeError("Error code: 401 - bad key"))
    result = correction.correct_once("raw", client, "m")
    assert result.status == "failed"
    assert not correction.is_rate_limited(result.error)


def test_empty_choices_fails_instead_of_indexerror():
    resp = _Resp("x")
    resp.choices = []
    result = correction.correct_once("raw", FakeClient(resp), "m")
    assert result.status == "failed"
    assert "no choices" in result.error


def test_missing_usage_does_not_crash():
    """Some OpenAI-compatible providers omit usage; a bench must survive it."""
    result = correction.correct_once("raw", FakeClient(_Resp("x", usage=False)), "m")
    assert result.status == "ok"
    assert result.prompt_tokens == 0 and result.completion_tokens == 0


def test_backoff_schedule_matches_the_original():
    """main.py used to compute `45 * (attempt + 1)` inline."""
    assert correction.RETRY_BACKOFF_SECONDS == (45, 90)


# --------------------------------------------------------------------------- #
# main.py still behaves the same
# --------------------------------------------------------------------------- #
def _load_main(monkeypatch):
    """Load main.py with whisper and transformers stubbed.

    Both are imported at module scope (main.py:12, emotion.py:20) and together
    weigh ~2.5 GB, but neither is on the correction path this file tests.
    Stubbing them lets the refactor stay covered in the lean venv the
    benchmark runs in, rather than being skipped exactly where it matters.
    """
    for name in ("whisper", "transformers"):
        monkeypatch.setitem(sys.modules, name,
                            types.ModuleType(name))
    # main.py calls load_dotenv() at import, which would pull the developer's
    # local .env into these tests -- so a "what is the default" assertion would
    # really be reading someone's secrets file. Neutralise it so the tests see
    # the code defaults and whatever the test itself sets.
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    spec = importlib.util.spec_from_file_location("vtt_main_corr", POC / "main.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vtt_main_corr", module)
    spec.loader.exec_module(module)
    return module


def test_skips_when_no_llm_configured(monkeypatch):
    import asyncio
    main = _load_main(monkeypatch)
    monkeypatch.setattr(main, "llm", None)
    assert asyncio.run(main.correct_with_ai("x")) == (None, "skipped")


def test_returns_ok_from_correct_once(monkeypatch):
    import asyncio
    main = _load_main(monkeypatch)
    monkeypatch.setattr(main, "llm", FakeClient(_Resp("corrected")))
    assert asyncio.run(main.correct_with_ai("x")) == ("corrected", "ok")


# --------------------------------------------------------------------------- #
# bench scoring
# --------------------------------------------------------------------------- #
def test_cost_is_per_million_tokens():
    spec = {"price_in": 0.15, "price_out": 1.50}
    # 1M in + 1M out at Qwen Flash list = 0.15 + 1.50
    assert bench_correction.cost_rmb(spec, 1_000_000, 1_000_000) == pytest.approx(1.65)
    assert bench_correction.cost_rmb(spec, 500, 200) == pytest.approx(
        500 / 1e6 * 0.15 + 200 / 1e6 * 1.50)


def test_has_cjk():
    assert bench_correction.has_cjk(ZH)
    assert not bench_correction.has_cjk("plain ascii")
    assert bench_correction.has_cjk("mixed " + ZH + " text")


def test_flags_detect_dropped_proper_noun():
    case = {"text": "we ship in Q3", "must_keep": ["Q3"]}
    assert bench_correction.flags_for(case, "We ship in Q3.")["must_keep_missed"] == []
    assert bench_correction.flags_for(case, "We ship soon.")["must_keep_missed"] == ["Q3"]


def test_flags_detect_summarising_and_drift():
    case = {"text": ZH * 50, "must_keep": []}
    short = bench_correction.flags_for(case, ZH * 5)
    assert short["len_ratio"] == pytest.approx(0.1)
    translated = bench_correction.flags_for(case, "some english summary")
    assert translated["script_drift"] is True


def test_flags_on_failed_call_are_inert():
    flags = bench_correction.flags_for({"text": "x", "must_keep": ["x"]}, None)
    assert flags["len_ratio"] is None
    assert flags["must_keep_missed"] == []


def test_identical_to_input_ignores_surrounding_whitespace():
    case = {"text": "Already correct.", "must_keep": []}
    assert bench_correction.flags_for(case, "\nAlready correct. ")["identical_to_input"]


def test_diff_marks_insertions_and_deletions():
    marked = bench_correction.diff_html("ab", "aXb")
    assert "<ins>X</ins>" in marked
    marked = bench_correction.diff_html("aXb", "ab")
    assert "<del>X</del>" in marked


def test_diff_escapes_html():
    assert "<script>" not in bench_correction.diff_html("a", "a<script>")


def test_every_registry_model_resolves():
    for key in bench_correction.MODELS:
        spec = bench_correction.resolve(key)
        assert spec["model"] and spec["base_url"]
        assert spec["price_in"] > 0 and spec["price_out"] > 0


def test_template_cases_are_ascii_and_valid():
    """The committed fallback corpus must not trip org Rule 3."""
    for case in bench_correction.TEMPLATE:
        assert case["category"] in bench_correction.CATEGORIES
        assert case["text"].isascii()
        for token in case["must_keep"]:
            assert token in case["text"], f"{token!r} not in its own case text"


# --------------------------------------------------------------------------- #
# Fixes prompted by the first real run
# --------------------------------------------------------------------------- #
def test_must_keep_is_case_insensitive_for_ascii():
    """Capitalising "api" to "API" is a wanted fix, not a dropped term.
    The first Qwen run flagged it as a miss on every single call."""
    case = {"text": "the api returns a 500", "must_keep": ["API", "500"]}
    assert bench_correction.flags_for(case, "The API returns a 500.")[
        "must_keep_missed"] == []
    assert bench_correction.flags_for(case, "The api returns a 500.")[
        "must_keep_missed"] == []
    assert bench_correction.flags_for(case, "It returns a 500.")[
        "must_keep_missed"] == ["API"]


def test_must_keep_stays_exact_for_cjk():
    case = {"text": ZH, "must_keep": [ZH]}
    assert bench_correction.flags_for(case, ZH)["must_keep_missed"] == []
    assert bench_correction.flags_for(case, "other")["must_keep_missed"] == [ZH]


def _rec(case_id, model_key, output, status="ok"):
    return {"case_id": case_id, "model_key": model_key, "output": output,
            "status": status}


def test_variants_collapses_identical_repeats():
    recs = [_rec("c1", "m", "same"), _rec("c1", "m", "same "),
            _rec("c1", "m", "\nsame")]
    assert bench_correction.variants_of(recs, "c1", "m") == ["same"]


def test_variants_preserves_order_of_differing_answers():
    recs = [_rec("c1", "m", "first"), _rec("c1", "m", "second"),
            _rec("c1", "m", "first")]
    assert bench_correction.variants_of(recs, "c1", "m") == ["first", "second"]


def test_variants_ignores_failed_calls():
    recs = [_rec("c1", "m", "ok text"), _rec("c1", "m", None, status="failed")]
    assert bench_correction.variants_of(recs, "c1", "m") == ["ok text"]


def test_instability_is_marked_per_case_and_model():
    """qwen-flash translated an English case to Chinese on one run of three;
    a sheet showing only the first attempt would have called it clean."""
    recs = [_rec("flaky", "m1", "english"), _rec("flaky", "m1", ZH),
            _rec("steady", "m1", "same"), _rec("steady", "m1", "same"),
            _rec("flaky", "m2", "same"), _rec("flaky", "m2", "same")]
    bench_correction.annotate_instability(recs)
    marked = {(r["case_id"], r["model_key"]) for r in recs if r["unstable"]}
    assert marked == {("flaky", "m1")}


def test_registry_holds_only_reachable_models():
    """The Gemini rows were removed with the Vertex shim: AI Studio 404s
    gemini-2.5-flash for new keys and no other route to it remains."""
    assert set(bench_correction.MODELS) == {"qwen-flash", "qwen3.5-flash"}
    for key in bench_correction.MODELS:
        spec = bench_correction.resolve(key)
        assert spec["api_key_env"] == "DASHSCOPE_API_KEY"


# --------------------------------------------------------------------------- #
# Multi-error cases: scoring the planted errors
# --------------------------------------------------------------------------- #
def test_planted_error_counts_as_fixed_only_when_wrong_form_is_gone():
    """A model that emits the right word but leaves the wrong one elsewhere
    has not corrected the sentence."""
    case = {"text": "a wrong b wrong c", "must_keep": [],
            "planted": [{"wrong": "wrong", "right": "right"}]}
    assert bench_correction.score_planted(case, "a right b right c") == (["wrong"], [])
    # right form present, wrong form still there -> not fixed
    assert bench_correction.score_planted(case, "a right b wrong c") == ([], ["wrong"])
    # untouched -> not fixed
    assert bench_correction.score_planted(case, "a wrong b wrong c") == ([], ["wrong"])


def test_planted_scores_each_error_independently():
    case = {"text": "x1 y1", "must_keep": [],
            "planted": [{"wrong": "x1", "right": "x2"},
                        {"wrong": "y1", "right": "y2"}]}
    fixed, missed = bench_correction.score_planted(case, "x2 y1")
    assert fixed == ["x1"] and missed == ["y1"]


def test_flags_expose_planted_results():
    case = {"text": "aa bb", "must_keep": [],
            "planted": [{"wrong": "aa", "right": "cc"}]}
    flags = bench_correction.flags_for(case, "cc bb")
    assert flags["planted_fixed"] == ["aa"] and flags["planted_missed"] == []


def test_flags_on_failed_call_have_empty_planted():
    case = {"text": "aa", "must_keep": [],
            "planted": [{"wrong": "aa", "right": "cc"}]}
    flags = bench_correction.flags_for(case, None)
    assert flags["planted_fixed"] == [] and flags["planted_missed"] == []


def test_case_with_no_planted_field_still_works():
    """The original 21-case corpus declares no planted errors."""
    flags = bench_correction.flags_for({"text": "x", "must_keep": []}, "x")
    assert flags["planted_fixed"] == [] and flags["planted_missed"] == []


def test_planted_match_survives_inserted_punctuation():
    """Adding punctuation is the job. A planted span that gets a comma
    inserted into it is a fix, not a miss -- matching raw strings scored
    those as failures and undercounted every model."""
    # difang-zai-jueding -> difang-zai-jueding, built from code points so this
    # file stays ASCII (org Rule 3). COMMA is the fullwidth comma a model
    # inserts, which is exactly what used to break the match.
    difang = chr(0x5730) + chr(0x65B9)          # "place"
    zai_wrong, zai_right = chr(0x5728), chr(0x518D)   # at / again
    jueding = chr(0x51B3) + chr(0x5B9A)         # "decide"
    comma, period = chr(0xFF0C), chr(0x3002)
    wrong = difang + zai_wrong + jueding
    right = difang + zai_right + jueding
    case = {"text": wrong, "must_keep": [],
            "planted": [{"wrong": wrong, "right": right}]}
    fixed, missed = bench_correction.score_planted(
        case, difang + comma + zai_right + jueding + period)
    assert fixed == [wrong] and missed == []


def test_planted_match_keeps_spaces_significant():
    """The English cases plant compounds the ASR split; stripping spaces
    would collapse the wrong and right forms into the same string."""
    case = {"text": "the end point", "must_keep": [],
            "planted": [{"wrong": "end point", "right": "endpoint"}]}
    assert bench_correction.score_planted(case, "the endpoint.")[0] == ["end point"]
    assert bench_correction.score_planted(case, "the end point.")[1] == ["end point"]


def test_strip_punct_keeps_spaces_and_letters():
    assert bench_correction.strip_punct("a, b. c") == "a b c"
    assert bench_correction.strip_punct("back-off") == "backoff"


# --------------------------------------------------------------------------- #
# The production default moved to qwen3.5-flash (2026-09-05 benchmark)
# --------------------------------------------------------------------------- #
def test_thinking_is_disabled_for_qwen_models():
    """The whole benchmark ran with thinking off. Shipping without this would
    put production on a configuration that was never measured -- slower, and
    billed for reasoning tokens on a task that needs no reasoning."""
    assert correction.default_extra_body("qwen3.5-flash") == {"enable_thinking": False}
    assert correction.default_extra_body("qwen-flash") == {"enable_thinking": False}
    assert correction.default_extra_body("Qwen3.5-Flash") == {"enable_thinking": False}


def test_no_extra_body_for_non_qwen_models():
    """A provider that does not know the flag must not receive it."""
    assert correction.default_extra_body("gemini-2.5-flash") is None
    assert correction.default_extra_body("gpt-4o-mini") is None
    assert correction.default_extra_body("deepseek-chat") is None


def test_defaults_are_qwen35_on_dashscope(monkeypatch):
    """LLM_BASE_URL must carry a default of its own: the OpenAI SDK falls back
    to api.openai.com when base_url is None, so a bare LLM_MODEL override
    would silently point at a provider that does not serve the model."""
    for var in ("LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    main = _load_main(monkeypatch)
    assert main.LLM_MODEL == "qwen3.5-flash"
    assert main.LLM_BASE_URL == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_correction_call_carries_the_thinking_switch(monkeypatch):
    """End of the wire: /transcribe must actually send the flag."""
    import asyncio
    monkeypatch.delenv("LLM_MODEL", raising=False)
    main = _load_main(monkeypatch)
    client = FakeClient(_Resp("fixed"))
    monkeypatch.setattr(main, "llm", client)
    assert asyncio.run(main.correct_with_ai("x")) == ("fixed", "ok")
    sent = client.calls[0]
    assert sent["model"] == "qwen3.5-flash"
    assert sent["extra_body"] == {"enable_thinking": False}

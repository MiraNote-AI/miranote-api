"""An empty image response has to say why it was empty.

/generate used to throw away everything the response carried and answer a bare
502, so a safety refusal and a transient blank were indistinguishable in the
log and to the tester. /stylize and /border never had this gap -- they go
through shared.vertex_client._extract_image_bytes, which surfaces the same
fields this module now exposes for /generate.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from generate import fallback


def _response(parts=(), finish_reason=None, safety=None, prompt_feedback=None):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=list(parts)),
        finish_reason=finish_reason,
        safety_ratings=safety,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=prompt_feedback)


class EmptyReasonTests(unittest.TestCase):
    def test_finish_reason_is_reported(self):
        reason = fallback.empty_reason(_response(finish_reason="SAFETY"))
        self.assertIn("SAFETY", reason)

    def test_text_the_model_returned_instead_of_an_image_is_reported(self):
        parts = [SimpleNamespace(inline_data=None, text="I can't make that.")]
        reason = fallback.empty_reason(_response(parts, finish_reason="STOP"))
        self.assertIn("I can't make that.", reason)

    def test_prompt_feedback_is_reported(self):
        reason = fallback.empty_reason(_response(prompt_feedback="BLOCKED_REASON_UNSPECIFIED"))
        self.assertIn("BLOCKED_REASON_UNSPECIFIED", reason)

    def test_a_response_with_no_candidates_says_so(self):
        reason = fallback.empty_reason(SimpleNamespace(candidates=[]))
        self.assertIn("no candidates", reason)

    def test_the_reason_is_bounded(self):
        """It goes in a log line, so one pathological response cannot flood it."""
        parts = [SimpleNamespace(inline_data=None, text="x" * 10_000)]
        self.assertLessEqual(len(fallback.empty_reason(_response(parts))), 600)

    def test_it_never_raises_on_a_malformed_response(self):
        for bad in (SimpleNamespace(), SimpleNamespace(candidates=None), object()):
            with self.subTest(bad=type(bad).__name__):
                self.assertIsInstance(fallback.empty_reason(bad), str)


class SafetyRefusalTests(unittest.TestCase):
    """Separated from empty_reason because the retry decision turns on it:
    a refusal retried is a second call spent to be refused again, out of a
    bucket that only holds two per minute."""

    def test_a_safety_finish_reason_is_a_refusal(self):
        self.assertTrue(fallback.is_safety_refusal(_response(finish_reason="SAFETY")))

    def test_a_prohibited_content_finish_reason_is_a_refusal(self):
        self.assertTrue(
            fallback.is_safety_refusal(_response(finish_reason="PROHIBITED_CONTENT"))
        )

    def test_a_blocked_prompt_is_a_refusal(self):
        self.assertTrue(
            fallback.is_safety_refusal(_response(prompt_feedback="BLOCKED: SAFETY"))
        )

    def test_an_ordinary_blank_is_not_a_refusal(self):
        """The case a retry is actually for."""
        self.assertFalse(fallback.is_safety_refusal(_response(finish_reason="STOP")))
        self.assertFalse(fallback.is_safety_refusal(_response()))
        self.assertFalse(fallback.is_safety_refusal(SimpleNamespace(candidates=[])))

    def test_it_never_raises_on_a_malformed_response(self):
        for bad in (SimpleNamespace(), SimpleNamespace(candidates=None), object()):
            with self.subTest(bad=type(bad).__name__):
                self.assertIsInstance(fallback.is_safety_refusal(bad), bool)


if __name__ == "__main__":
    unittest.main()


import contextlib
import io as _io
import os
from unittest import mock

from fastapi import HTTPException

import config
import main

os.environ.setdefault("BETA_TOKENS", "test-token")


def _client_returning(response):
    client = mock.Mock()
    client.models.generate_content.return_value = response
    return client


def _call_and_capture(response):
    """Run _call_model against a response carrying no image; return (exc, log)."""
    client = _client_returning(response)
    buffer = _io.StringIO()
    with mock.patch.object(main, "_get_client", return_value=client):
        with contextlib.redirect_stdout(buffer):
            with unittest.TestCase().assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1")
    return caught.exception, buffer.getvalue()


class GenerateEmptyResponseTests(unittest.TestCase):
    def test_a_refusal_and_a_blank_do_not_read_the_same(self):
        refusal, _ = _call_and_capture(_response(finish_reason="SAFETY"))
        blank, _ = _call_and_capture(_response(finish_reason="STOP"))
        self.assertEqual(refusal.status_code, 502)
        self.assertEqual(blank.status_code, 502)
        self.assertNotEqual(
            refusal.detail, blank.detail,
            "a refused prompt and an empty answer tell the tester the same thing",
        )

    def test_the_reason_reaches_the_log(self):
        _, log = _call_and_capture(_response(finish_reason="SAFETY"))
        self.assertIn("SAFETY", log)

    def test_the_tester_is_not_shown_raw_provider_wording(self):
        provider_text = "PROHIBITED_CONTENT"
        exc, log = _call_and_capture(_response(finish_reason=provider_text))
        self.assertIn(provider_text, log, "the log must keep the provider's own words")
        self.assertNotIn(provider_text, exc.detail)
        self.assertNotIn("finish_reason", exc.detail)

    def test_a_refusal_is_not_reported_as_a_quota_problem(self):
        """503 means 'wait and it will work'. A refusal will not."""
        exc, _ = _call_and_capture(_response(finish_reason="SAFETY"))
        self.assertNotEqual(exc.status_code, 503)
        self.assertNotIn("busy", exc.detail)

    def test_an_image_still_comes_back_untouched(self):
        """The diagnostics path must not disturb the normal one."""
        parts = [SimpleNamespace(inline_data=SimpleNamespace(data=b"png-1"))]
        client = _client_returning(_response(parts))
        with mock.patch.object(main, "_get_client", return_value=client):
            self.assertEqual(
                main._call_model("a cat", "1:1"),
                [b"png-1"] * config.NUMBER_OF_IMAGES,
            )


from google.genai.types import FinishReason

# Every FinishReason the SDK defines, split by the only question that matters
# here: would retrying the same prompt have a chance? Pinned against the real
# enum rather than a mock because the SDK renders it as
# "FinishReason.IMAGE_SAFETY", and because a new member must not default to
# "retryable" -- that would spend a second call from a 2/min bucket to be
# refused again.
REFUSING = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
}
RETRYABLE = {
    "FINISH_REASON_UNSPECIFIED", "STOP", "MAX_TOKENS", "LANGUAGE", "OTHER",
    "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "NO_IMAGE", "IMAGE_OTHER",
}


class RealFinishReasonTests(unittest.TestCase):
    def test_every_finish_reason_is_classified_the_way_it_was_meant_to_be(self):
        for name in REFUSING | RETRYABLE:
            if name not in FinishReason.__members__:
                continue
            with self.subTest(name=name):
                response = _response(finish_reason=FinishReason[name])
                self.assertEqual(
                    fallback.is_safety_refusal(response),
                    name in REFUSING,
                    f"{name} is classified the wrong way round",
                )

    def test_an_unclassified_finish_reason_fails_loudly(self):
        """A member added by an SDK upgrade must be triaged, not defaulted.

        Defaulting is not safe in either direction: a new refusal treated as
        retryable burns a scarce call, and a new ordinary ending treated as a
        refusal tells the tester to rephrase a prompt that was fine.
        """
        self.assertEqual(
            set(FinishReason.__members__) - (REFUSING | RETRYABLE),
            set(),
            "new FinishReason member(s); add each to REFUSING or RETRYABLE above",
        )

    def test_the_enum_renders_into_the_log_line_readably(self):
        reason = fallback.empty_reason(_response(finish_reason=FinishReason.IMAGE_SAFETY))
        self.assertIn("IMAGE_SAFETY", reason)

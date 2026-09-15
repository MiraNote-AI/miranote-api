"""An empty answer gets one more chance; a refusal does not.

NUMBER_OF_IMAGES = 1 (#70) removed the redundancy that used to cover a blank
response -- with two images in flight, one empty answer was hidden by the
other. At one image a blank is a user-visible 502, measured once in ten paced
calls through the tunnel.

The retry is deliberately asymmetric. Vertex allows two image calls a minute
per model, so a second call is expensive; spending one on a refusal buys
another refusal, while spending one on FinishReason.NO_IMAGE can actually
produce a picture (#78).
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from google.genai.types import FinishReason

import config
import main
from tests.test_fallback import RATE_LIMITED

os.environ.setdefault("BETA_TOKENS", "test-token")


def _empty(finish_reason):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[]),
        finish_reason=finish_reason,
        safety_ratings=None,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=None)


def _image(data=b"png"):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(data=data))]),
        finish_reason=FinishReason.STOP,
        safety_ratings=None,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=None)


def _client(*responses):
    client = mock.Mock()
    client.models.generate_content.side_effect = list(responses)
    return client


def _call(client):
    with mock.patch.object(main, "_get_client", return_value=client):
        return main._call_model("a cat", "1:1")


class EmptyResponseRetryTests(unittest.TestCase):
    def test_a_blank_is_retried_and_the_second_answer_is_used(self):
        client = _client(_empty(FinishReason.NO_IMAGE), _image(b"second-try"))
        self.assertEqual(_call(client), [b"second-try"] * config.NUMBER_OF_IMAGES)

    def test_the_retry_happens_exactly_once(self):
        client = _client(_empty(FinishReason.NO_IMAGE), _empty(FinishReason.NO_IMAGE))
        with self.assertRaises(HTTPException) as caught:
            _call(client)
        self.assertEqual(caught.exception.status_code, 502)
        self.assertEqual(caught.exception.detail, main.EMPTY_DETAIL)
        self.assertEqual(
            client.models.generate_content.call_count,
            2 * config.NUMBER_OF_IMAGES,
            "a bounded retry means two attempts per image, never three",
        )

    def test_a_refusal_is_not_retried(self):
        """The whole point of telling the two apart: a second call buys nothing."""
        client = _client(_empty(FinishReason.IMAGE_SAFETY), _image())
        with self.assertRaises(HTTPException) as caught:
            _call(client)
        self.assertEqual(caught.exception.detail, main.REFUSED_DETAIL)
        self.assertEqual(
            client.models.generate_content.call_count,
            config.NUMBER_OF_IMAGES,
            "a refusal was retried, spending a call from a 2/min bucket to be refused again",
        )

    def test_a_successful_first_answer_costs_one_call(self):
        client = _client(_image())
        self.assertEqual(_call(client), [b"png"] * config.NUMBER_OF_IMAGES)
        self.assertEqual(
            client.models.generate_content.call_count, config.NUMBER_OF_IMAGES
        )

    def test_quota_is_still_not_retried(self):
        client = mock.Mock()
        client.models.generate_content.side_effect = Exception(RATE_LIMITED)
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1")
        self.assertEqual(caught.exception.status_code, 503)
        self.assertLessEqual(
            client.models.generate_content.call_count, config.NUMBER_OF_IMAGES
        )

    def test_a_quota_rejection_on_the_retry_surfaces_as_quota(self):
        """The retry is what spends the last token, so this is the likely shape."""
        client = mock.Mock()
        client.models.generate_content.side_effect = [
            _empty(FinishReason.NO_IMAGE),
            Exception(RATE_LIMITED),
        ]
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1")
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()

"""What a tester's phone receives when DashScope refuses /generate.

Gemini reports why it produced no image in the response object, which
generate/fallback.py reads. DashScope has no equivalent: everything arrives as
one ProviderError whose text is the only signal. main._classify_provider_error
turns that text into a status, and getting it wrong is silent -- the request
still fails, just with the wrong advice attached.

The status is the whole payload, not a detail. MiraNoteKit's BackendError keys
its message off the HTTP code alone and discards the server's detail, so 502
tells a tester "the AI server is not running, ask the team to start the Mac"
while 503 tells them "the image service is out of quota". Bailian ran out of
credit once mid-benchmark (all three modes answered 400 Arrearage at the same
instant); if that shows as 502, everyone goes and looks at a Mac that is fine.

No network. The provider layer is mocked, which is also why these tests say
nothing about whether z-image-turbo draws a good apple.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import httpx
from fastapi import HTTPException

import config
import main
from generate.http_client import ProviderError

os.environ["BETA_TOKENS"] = "test-token"

# Shaped like the real thing: http_client raises with the method, the tail of
# the URL, the status and the body, so a marker is a substring of a longer line
# rather than the whole message.
ARREARAGE = ProviderError(
    "POST generation -> 400 {'code': 'Arrearage', 'message': "
    "'Access denied, please make sure your account is in good standing.'}"
)
THROTTLED = ProviderError("POST generation -> 429 {'code': 'Throttling'}")
BLOCKED = ProviderError(
    "{'code': 'DataInspectionFailed', 'message': 'Input data may contain "
    "inappropriate content.'}"
)
UNKNOWN = ProviderError("POST generation -> 400 {'code': 'InvalidApiKey'}")

MODEL = config.IMAGE_MODELS_BY_MODE["sticker"]


class ClassificationTests(unittest.TestCase):
    """The text -> outcome mapping on its own."""

    def test_arrearage_is_a_quota_problem(self):
        self.assertEqual(main._classify_provider_error(ARREARAGE), "quota")

    def test_a_surviving_429_is_a_quota_problem(self):
        # http_client already backed off four times. Reaching here means the
        # retrying is done, not that it should start.
        self.assertEqual(main._classify_provider_error(THROTTLED), "quota")

    def test_a_content_block_is_a_refusal(self):
        self.assertEqual(main._classify_provider_error(BLOCKED), "refused")

    def test_anything_else_stays_unknown(self):
        # Deliberately not guessed into a bucket: a wrong status is worse advice
        # than the generic path, which at least carries the provider's words.
        self.assertEqual(main._classify_provider_error(UNKNOWN), "unknown")


class StatusTests(unittest.TestCase):
    """_call_model's status, and how many image calls each failure costs."""

    def _call(self, *side_effect):
        provider = mock.Mock(side_effect=list(side_effect))
        with mock.patch.object(main.image_providers, "generate", provider):
            with self.assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1", MODEL)
        return caught.exception, provider

    def test_out_of_credit_is_503_not_502(self):
        error, _ = self._call(ARREARAGE)
        self.assertEqual(error.status_code, 503)

    def test_out_of_credit_does_not_leak_provider_wording(self):
        error, _ = self._call(ARREARAGE)
        self.assertNotIn("Arrearage", error.detail)
        self.assertIn("busy", error.detail.lower())

    def test_out_of_credit_is_not_retried(self):
        # A second call cannot succeed and the account is already empty.
        _, provider = self._call(ARREARAGE)
        self.assertEqual(provider.call_count, config.NUMBER_OF_IMAGES)

    def test_a_content_block_is_502_and_not_retried(self):
        error, provider = self._call(BLOCKED)
        self.assertEqual(error.status_code, 502)
        self.assertEqual(error.detail, main.REFUSED_DETAIL)
        self.assertEqual(provider.call_count, config.NUMBER_OF_IMAGES,
                         "a refusal bought another refusal")

    def test_an_unclassified_failure_is_502_carrying_the_provider_text(self):
        error, provider = self._call(UNKNOWN)
        self.assertEqual(error.status_code, 502)
        self.assertIn("InvalidApiKey", error.detail,
                      "an untriaged failure must keep the provider's own words")
        self.assertEqual(provider.call_count, config.NUMBER_OF_IMAGES,
                         "an unknown failure is not a reason to spend again")

    def test_a_blank_answer_is_retried_once(self):
        provider = mock.Mock(side_effect=[[], [b"png"]])
        with mock.patch.object(main.image_providers, "generate", provider):
            images = main._call_model("a cat", "1:1", MODEL)
        self.assertEqual(images, [b"png"])
        self.assertEqual(provider.call_count, 2)

    def test_a_blank_answer_twice_is_502_empty(self):
        error, provider = self._call([], [])
        self.assertEqual(error.status_code, 502)
        self.assertEqual(error.detail, main.EMPTY_DETAIL)
        self.assertEqual(provider.call_count, 2, "bounded at one extra attempt")

    def test_the_retry_sends_a_freshly_expanded_prompt(self):
        # The point of re-expanding: the second call must not resend the string
        # that just drew a blank.
        provider = mock.Mock(side_effect=[[], [b"png"]])
        with mock.patch.object(main.image_providers, "generate", provider):
            main._call_model("first", "1:1", MODEL, reprompt=lambda: "second")
        sent = [call.args[1] for call in provider.call_args_list]
        self.assertEqual(sent, ["first", "second"])

    def test_a_working_call_costs_one_request(self):
        provider = mock.Mock(return_value=[b"png"])
        with mock.patch.object(main.image_providers, "generate", provider):
            self.assertEqual(main._call_model("a cat", "1:1", MODEL), [b"png"])
        self.assertEqual(provider.call_count, config.NUMBER_OF_IMAGES)

    def test_the_configured_model_is_the_one_asked_for(self):
        provider = mock.Mock(return_value=[b"png"])
        with mock.patch.object(main.image_providers, "generate", provider):
            main._call_model("a cat", "9:16", MODEL)
        self.assertEqual(provider.call_args.args[0], MODEL)
        self.assertEqual(provider.call_args.args[2], "9:16")


class OverTheWireTests(unittest.IsolatedAsyncioTestCase):
    """The status that leaves the app, and the concurrency slot behind it.

    _call_model runs on a worker thread, so a status chosen there only helps if
    it survives that boundary and reaches FastAPI. Asserting on _call_model
    alone would not show that.
    """

    async def _post(self, side_effect):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://probe",
            headers={"Authorization": "Bearer test-token"},
        ) as client:
            with mock.patch.object(main.image_providers, "generate",
                                   mock.Mock(side_effect=side_effect)):
                return await client.post(
                    "/generate",
                    json={"command": "sticker", "prompt": "a cat", "expand": False},
                )

    async def test_out_of_credit_reaches_the_phone_as_503(self):
        response = await self._post(ARREARAGE)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("Arrearage", response.text)

    async def test_a_content_block_reaches_the_phone_as_502(self):
        response = await self._post(BLOCKED)
        self.assertEqual(response.status_code, 502)

    async def test_a_dashscope_failure_releases_the_concurrency_slot(self):
        """A leaked slot wedges /generate for the life of the process, which is
        a worse failure than the one being reported."""
        main._generate_semaphore = None
        self.addCleanup(setattr, main, "_generate_semaphore", None)

        for _ in range(main.GENERATE_CONCURRENCY + 2):
            response = await self._post(ARREARAGE)
            self.assertEqual(response.status_code, 503)

        self.assertEqual(main._generate_semaphore._value,
                         main.GENERATE_CONCURRENCY,
                         "DashScope failures leaked concurrency slots")


if __name__ == "__main__":
    unittest.main()

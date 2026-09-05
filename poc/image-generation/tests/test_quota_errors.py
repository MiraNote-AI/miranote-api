"""How /generate behaves when the image provider is out of quota.

Ten concurrent generations exhausted the provider's quota and every rejection
reached the client as a bare 500, which a tester cannot tell apart from a
crash. These tests pin the three properties that fix needs: the failure is
distinguishable, it does not latch, and it does not retry.
"""

from __future__ import annotations

import unittest
from unittest import mock

import httpx
from fastapi import HTTPException

import config
import main
from tests.test_fallback import MODEL_GATED, RATE_LIMITED


class QuotaExhaustionTests(unittest.TestCase):
    def setUp(self):
        main._imagen_unavailable = False
        self.addCleanup(setattr, main, "_imagen_unavailable", False)

    def _client_raising(self, text):
        client = mock.Mock()
        client.models.generate_images.side_effect = Exception(text)
        client.models.generate_content.side_effect = Exception(text)
        return client

    def test_primary_model_quota_is_not_a_500(self):
        client = self._client_raising(RATE_LIMITED)
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1")
        self.assertEqual(caught.exception.status_code, 503)

    def test_fallback_model_quota_is_not_a_500(self):
        """The path that actually failed under load: Imagen gated, fallback throttled."""
        client = mock.Mock()
        client.models.generate_images.side_effect = Exception(MODEL_GATED)
        client.models.generate_content.side_effect = Exception(RATE_LIMITED)
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1")
        self.assertEqual(caught.exception.status_code, 503)

    def test_the_message_tells_a_tester_what_to_do(self):
        client = self._client_raising(RATE_LIMITED)
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1")
        detail = caught.exception.detail.lower()
        self.assertNotIn("resource_exhausted", detail, "raw provider text leaked")
        self.assertTrue(
            "again" in detail or "busy" in detail,
            "detail does not tell the tester what to do: " + repr(caught.exception.detail),
        )

    def test_quota_does_not_disable_the_primary_model(self):
        """A gated model latches for the process; a throttled one must not.

        _imagen_unavailable is never reset, so latching on a transient quota
        error would send every later request to the fallback until restart --
        including requests made after quota recovered.
        """
        client = self._client_raising(RATE_LIMITED)
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException):
                main._call_model("a cat", "1:1")
        self.assertFalse(
            main._imagen_unavailable,
            "a quota error latched the primary model off for the whole process",
        )

    def test_quota_is_not_retried(self):
        """Section 7 of the deploy spec rules out automatic retry."""
        client = self._client_raising(RATE_LIMITED)
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException):
                main._call_model("a cat", "1:1")
        self.assertEqual(
            client.models.generate_images.call_count,
            1,
            "the primary model was called more than once",
        )

    def test_a_gated_model_still_latches(self):
        """The existing 404 behaviour must survive this change."""
        client = mock.Mock()
        client.models.generate_images.side_effect = Exception(MODEL_GATED)
        client.models.generate_content.side_effect = Exception("500 INTERNAL")
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(Exception):
                main._call_model("a cat", "1:1")
        self.assertTrue(main._imagen_unavailable)

    def test_an_unrelated_failure_still_propagates(self):
        client = self._client_raising("500 INTERNAL")
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(Exception) as caught:
                main._call_model("a cat", "1:1")
        self.assertNotIsInstance(caught.exception, HTTPException)


class QuotaOverTheWireTests(unittest.IsolatedAsyncioTestCase):
    """The status a tester's phone actually receives.

    _call_model runs on a worker thread via asyncio.to_thread, so raising
    HTTPException there only helps if it survives that boundary and reaches
    FastAPI's handler. Asserting on _call_model alone would not show that.
    """

    async def test_generate_answers_503_rather_than_500(self):
        main._imagen_unavailable = False
        self.addCleanup(setattr, main, "_imagen_unavailable", False)

        stub = mock.Mock()
        stub.models.generate_images.side_effect = Exception(RATE_LIMITED)

        with mock.patch.object(main, "_get_client", return_value=stub):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://probe"
            ) as client:
                response = await client.post(
                    "/generate",
                    json={"command": "sticker", "prompt": "a cat", "expand": False},
                )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("RESOURCE_EXHAUSTED", response.text)


if __name__ == "__main__":
    unittest.main()

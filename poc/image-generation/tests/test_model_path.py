"""One model serves /generate, and it is the one the config names.

The previous arrangement named Imagen as primary and reached
gemini-2.5-flash-image only after a 404. Imagen is enabled per project on
Vertex and was never enabled on any project this has run against, so that 404
was paid on the first request after every restart and the config pointed at a
model that had never produced an image.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from fastapi import HTTPException

import config
import main
from tests.test_fallback import RATE_LIMITED

os.environ["BETA_TOKENS"] = "test-token"


class ModelPathTests(unittest.TestCase):
    def _client(self):
        client = mock.Mock()
        client.models.generate_content.return_value = mock.Mock()
        return client

    def test_the_configured_model_is_the_one_called(self):
        client = self._client()
        with mock.patch.object(main, "_get_client", return_value=client), mock.patch.object(
            main.fallback, "image_parts", return_value=[b"png"]
        ):
            main._call_model("a cat", "1:1")

        called = {c.kwargs.get("model") for c in client.models.generate_content.call_args_list}
        self.assertEqual(called, {config.MODEL_ID})

    def test_no_request_is_wasted_on_an_unreachable_model(self):
        """The first call after a restart must reach the image model directly."""
        client = self._client()
        with mock.patch.object(main, "_get_client", return_value=client), mock.patch.object(
            main.fallback, "image_parts", return_value=[b"png"]
        ):
            main._call_model("a cat", "1:1")

        client.models.generate_images.assert_not_called()

    def test_a_404_surfaces_instead_of_downgrading_silently(self):
        """It used to mean "try the other model". There is no other model."""
        client = self._client()
        client.models.generate_content.side_effect = Exception(
            "404 NOT_FOUND. {'error': {'code': 404}}"
        )
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(Exception) as caught:
                main._call_model("a cat", "1:1")
        self.assertIn("404", str(caught.exception))

    def test_nothing_can_disable_a_model_for_the_process(self):
        """The latch was never reset, so one bad minute cost until a restart."""
        self.assertFalse(
            hasattr(main, "_imagen_unavailable"),
            "a process-lifetime model latch is back",
        )

    def test_quota_still_maps_to_503(self):
        """Unaffected by which model is in front: it is about quota."""
        client = self._client()
        client.models.generate_content.side_effect = Exception(RATE_LIMITED)
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException) as caught:
                main._call_model("a cat", "1:1")
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()

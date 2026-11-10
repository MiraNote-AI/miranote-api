"""How /stylize answers when an edit cannot be made (api #96).

Every one of these used to reach the phone as a bare 500 with a text/plain
body, which the app shows as "status 500" and a tester cannot tell from a
crash. Each now has its own status and a JSON body with a stable `reason`
and a `message` to show. Driven over ASGI with the Vertex client stubbed, so
no model is called.
"""

from __future__ import annotations

import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from PIL import Image

import main
from stylize import stylizer
from tests.test_fallback import RATE_LIMITED

os.environ["BETA_TOKENS"] = "test-token"


def _png(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 160, 210)).save(buf, format="PNG")
    return buf.getvalue()


def _response(parts=(), finish_reason=None):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=list(parts)),
        finish_reason=finish_reason,
        safety_ratings=None,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=None)


def _image_response():
    return _response([SimpleNamespace(inline_data=SimpleNamespace(data=_png()),
                                      text=None)])


# The start of an iPhone photo as it leaves the camera roll. Pillow has no HEIC
# decoder without a plugin this service does not install.
HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64


class StylizeErrorTests(unittest.IsolatedAsyncioTestCase):
    async def _post(self, data=None, prompt="make it warmer", client=None):
        stub = client or mock.Mock()
        if client is None:
            stub.models.generate_content.return_value = _image_response()
        with mock.patch.object(stylizer, "_get_client", return_value=stub):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://probe",
                headers={"Authorization": "Bearer test-token"},
            ) as http:
                response = await http.post(
                    "/stylize",
                    params={"prompt": prompt},
                    files={"file": ("image.png", _png() if data is None else data,
                                    "image/png")},
                )
        return response, stub

    def _model_raising(self, error):
        stub = mock.Mock()
        stub.models.generate_content.side_effect = error
        return stub

    def _model_answering(self, response):
        stub = mock.Mock()
        stub.models.generate_content.return_value = response
        return stub

    def assertEditError(self, response, status, reason):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["reason"], reason)
        self.assertTrue(body["message"])

    async def test_a_good_edit_still_returns_the_image(self):
        response, stub = await self._post()
        self.assertEqual(response.status_code, 200)
        self.assertIn("image", response.json())
        config = stub.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.http_options.timeout, stylizer.TIMEOUT_SECONDS * 1000)

    async def test_an_empty_upload_is_a_400(self):
        response, stub = await self._post(data=b"")
        self.assertEditError(response, 400, "unreadable_image")
        stub.models.generate_content.assert_not_called()

    async def test_bytes_that_are_not_an_image_are_a_400(self):
        response, stub = await self._post(data=b"not an image at all")
        self.assertEditError(response, 400, "unreadable_image")
        stub.models.generate_content.assert_not_called()

    async def test_heic_is_a_400(self):
        response, stub = await self._post(data=HEIC)
        self.assertEditError(response, 400, "unreadable_image")
        stub.models.generate_content.assert_not_called()

    async def test_an_image_past_the_pixel_limit_is_a_413(self):
        # 32x32 is past twice this limit, which is where Pillow refuses to open.
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            response, stub = await self._post()
        self.assertEditError(response, 413, "image_too_large")
        stub.models.generate_content.assert_not_called()

    async def test_an_empty_prompt_is_a_400(self):
        response, stub = await self._post(prompt="")
        self.assertEditError(response, 400, "bad_request")
        stub.models.generate_content.assert_not_called()

    async def test_a_refusal_is_a_422(self):
        refused = self._model_answering(
            _response(finish_reason="FinishReason.IMAGE_SAFETY"))
        response, _ = await self._post(client=refused)
        self.assertEditError(response, 422, "edit_refused")
        self.assertNotIn("IMAGE_SAFETY", response.text)

    async def test_an_answer_in_words_only_is_a_502(self):
        words = self._model_answering(_response(
            [SimpleNamespace(inline_data=None, text="The picture is blue.")],
            finish_reason="FinishReason.STOP"))
        response, _ = await self._post(client=words)
        self.assertEditError(response, 502, "no_image")
        self.assertNotIn("The picture is blue.", response.text)

    async def test_a_timeout_is_a_504(self):
        response, _ = await self._post(
            client=self._model_raising(httpx.ReadTimeout("timed out")))
        self.assertEditError(response, 504, "edit_timeout")

    async def test_an_exhausted_quota_is_a_503(self):
        response, _ = await self._post(
            client=self._model_raising(Exception(RATE_LIMITED)))
        self.assertEditError(response, 503, "busy")
        self.assertNotIn("RESOURCE_EXHAUSTED", response.text)

    async def test_any_other_provider_failure_is_a_502(self):
        response, _ = await self._post(
            client=self._model_raising(Exception("500 INTERNAL")))
        self.assertEditError(response, 502, "provider_error")
        self.assertNotIn("INTERNAL", response.text)


if __name__ == "__main__":
    unittest.main()

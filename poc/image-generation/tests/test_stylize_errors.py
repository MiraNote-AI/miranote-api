"""How /stylize answers when an edit cannot be made (api #96).

Every one of these used to reach the phone as a bare 500 with a text/plain
body, which the app shows as "status 500" and a tester cannot tell from a
crash. Each now has its own status and a JSON body with a stable `reason`
and a `message` to show. Driven over ASGI with the Vertex client stubbed, so
no model is called.
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import httpx
from google import genai
from google.genai import errors, types
from PIL import Image

import main
from stylize import stylizer
from tests.test_fallback import RATE_LIMITED

os.environ["BETA_TOKENS"] = "test-token"


def _png(size=(32, 32)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 160, 210)).save(buf, format="PNG")
    return buf.getvalue()


def _response(parts=(), finish_reason=None, prompt_feedback=None):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=list(parts)),
        finish_reason=finish_reason,
        safety_ratings=None,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=prompt_feedback)


def _words_only(prompt_feedback=None):
    return _response([SimpleNamespace(inline_data=None, text="The picture is blue.")],
                     finish_reason="FinishReason.STOP", prompt_feedback=prompt_feedback)


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def _broken_png() -> bytes:
    """An RGBA PNG whose pixels span two IDAT chunks, the second one's type
    zeroed. Image.open succeeds; decoding raises SyntaxError("broken PNG")."""
    buf = io.BytesIO()
    Image.effect_noise((64, 64), 64).convert("RGBA").save(buf, format="PNG")
    png = buf.getvalue()
    out, pos, idat = [png[:8]], 8, b""
    while pos < len(png):
        size = struct.unpack(">I", png[pos:pos + 4])[0]
        kind, data = png[pos + 4:pos + 8], png[pos + 8:pos + 8 + size]
        pos += 12 + size
        if kind == b"IDAT":
            idat += data
            continue
        if kind == b"IEND":
            half = len(idat) // 2
            second = _chunk(b"IDAT", idat[half:])
            out += [_chunk(b"IDAT", idat[:half]), second[:4] + b"\0\0\0\0" + second[8:]]
        out.append(_chunk(kind, data))
    return b"".join(out)


def _short_header_png() -> bytes:
    """A valid PNG whose IHDR length field says 12, not 13. Image.open raises
    ValueError("Truncated IHDR chunk"), neither OSError nor SyntaxError."""
    png = bytearray(_png())
    png[8:12] = struct.pack(">I", 12)
    return bytes(png)


def _truncated_jpeg() -> bytes:
    """The first third of an 800x600 JPEG: under the 1536 px cap and opaque,
    so neither the shrink nor the transparency check decodes it."""
    buf = io.BytesIO()
    Image.effect_noise((800, 600), 64).convert("RGB").save(buf, format="JPEG")
    jpeg = buf.getvalue()
    return jpeg[:len(jpeg) // 3]


def _image_response():
    return _response([SimpleNamespace(inline_data=SimpleNamespace(data=_png()),
                                      text=None)])


# The start of an iPhone photo as it leaves the camera roll. Pillow has no HEIC
# decoder without a plugin this service does not install.
HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64


class _Stalled(BaseHTTPRequestHandler):
    """A model endpoint that reads the request and never answers."""
    release = threading.Event()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.release.wait(10)

    def log_message(self, *args):
        pass


class StylizeErrorTests(unittest.IsolatedAsyncioTestCase):
    async def _post(self, data=None, prompt="make it warmer", client=None,
                    send_file=True, params=None):
        stub = client or mock.Mock()
        if client is None:
            stub.models.generate_content.return_value = _image_response()
        with mock.patch.object(stylizer, "_get_client", return_value=stub):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://probe",
                headers={"Authorization": "Bearer test-token"},
            ) as http:
                upload = ("image.png", _png() if data is None else data, "image/png")
                response = await http.post(
                    "/stylize",
                    params={"prompt": prompt, **(params or {})},
                    files={"file" if send_file else "other": upload},
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

    async def test_a_png_that_breaks_while_decoding_is_a_400(self):
        response, stub = await self._post(data=_broken_png())
        self.assertEditError(response, 400, "unreadable_image")
        stub.models.generate_content.assert_not_called()

    async def test_a_png_with_a_short_header_is_a_400(self):
        response, stub = await self._post(data=_short_header_png())
        self.assertEditError(response, 400, "unreadable_image")
        stub.models.generate_content.assert_not_called()

    async def test_a_truncated_jpeg_is_a_400_without_a_model_call(self):
        response, stub = await self._post(data=_truncated_jpeg())
        self.assertEditError(response, 400, "unreadable_image")
        stub.models.generate_content.assert_not_called()

    async def test_no_file_part_is_a_422_in_the_same_shape(self):
        response, stub = await self._post(send_file=False)
        self.assertEditError(response, 422, "bad_request")
        self.assertEqual(response.json()["message"], main.MISSING_FILE_DETAIL)
        stub.models.generate_content.assert_not_called()

    async def test_a_bad_parameter_is_a_422_that_names_it(self):
        """Not the missing-file message: the file is there. The cause goes
        to the log as well."""
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            response, stub = await self._post(params={"temperature": "warm"})
        self.assertEditError(response, 422, "bad_request")
        self.assertNotEqual(response.json()["message"], main.MISSING_FILE_DETAIL)
        self.assertIn("temperature", response.json()["message"])
        self.assertIn("temperature", log.getvalue())
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

    async def test_an_answer_in_words_only_is_a_500(self):
        # 500, not 502: shipped apps read 502 as "the server is not running",
        # and Cloudflare replaces an origin's 502 body with its own page.
        response, _ = await self._post(client=self._model_answering(_words_only()))
        self.assertEditError(response, 500, "no_image")
        self.assertNotIn("The picture is blue.", response.text)

    async def test_an_unblocked_prompt_feedback_is_not_a_refusal(self):
        # The SDK object prints as "block_reason=None ...", which used to
        # match the "BLOCK" refusal marker.
        feedback = types.GenerateContentResponsePromptFeedback()
        response, _ = await self._post(
            client=self._model_answering(_words_only(feedback)))
        self.assertEditError(response, 500, "no_image")

    async def test_vertex_deadline_exceeded_is_a_504(self):
        deadline = errors.ServerError(504, {"error": {
            "code": 504, "status": "DEADLINE_EXCEEDED", "message": "Deadline expired"}})
        response, _ = await self._post(client=self._model_raising(deadline))
        self.assertEditError(response, 504, "edit_timeout")
        self.assertNotIn("DEADLINE_EXCEEDED", response.text)

    async def test_a_timeout_is_a_504(self):
        response, _ = await self._post(
            client=self._model_raising(httpx.ReadTimeout("timed out")))
        self.assertEditError(response, 504, "edit_timeout")

    async def test_a_timeout_in_the_real_sdk_is_a_504(self):
        """The SDK's own client against a server that never answers, not a
        stub: google-genai 3.x sends through httpx2, whose ReadTimeout is not
        an httpx.TimeoutException."""
        _Stalled.release.clear()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Stalled)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(_Stalled.release.set)
        # A proxy from the environment must not stand between us and 127.0.0.1.
        with mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
            client = genai.Client(api_key="test-not-a-key", http_options=types.HttpOptions(
                base_url=f"http://127.0.0.1:{server.server_address[1]}"))
        with mock.patch.object(stylizer, "TIMEOUT_SECONDS", 1):
            response, _ = await self._post(client=client)
        self.assertEditError(response, 504, "edit_timeout")

    async def test_an_exhausted_quota_is_a_503(self):
        response, _ = await self._post(
            client=self._model_raising(Exception(RATE_LIMITED)))
        self.assertEditError(response, 503, "busy")
        self.assertNotIn("RESOURCE_EXHAUSTED", response.text)

    async def test_any_other_provider_failure_is_a_500(self):
        response, _ = await self._post(
            client=self._model_raising(Exception("500 INTERNAL")))
        self.assertEditError(response, 500, "provider_error")
        self.assertNotIn("INTERNAL", response.text)


if __name__ == "__main__":
    unittest.main()

"""One model call is one request.

google-genai 2.30 added automatic continuation, on by default: when a reply
ends with finish_reason CONTINUATION and carries a continuation_token,
generate_content quietly sends the request again, with no cap, and each hop is
billed. An image call draws on Vertex's 1/min/{project}/{base_model} bucket
(config.py), and nothing in this service retries a quota on purpose (main.py,
EMPTY_RESPONSE_RETRIES), so every call site goes through
shared.vertex_client.generate_config, which turns it off.

Only where the installed SDK has the field: beta's venv is on 2.12.1, whose
GenerateContentConfig rejects it, so code that always set it would break every
/stylize and /border call there until the venv is upgraded.

The call sites run against a real genai.Client whose HTTP transport is a stub,
so what is counted is requests on the wire, not calls to a mock.
"""

from __future__ import annotations

import base64
import io
import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from PIL import Image

import config
import main
from border import border
from cutout import bbox_detector
from generate import gemini_image, prompt_expander
from shared import vertex_client
from stylize import stylizer

os.environ.setdefault("BETA_TOKENS", "test-token")

# The stub has to be the transport the SDK itself uses: httpx2 from
# google-genai 3.0, httpx before. Not "whichever imports": other packages pull
# httpx2 in too (huggingface_hub 2.x does), next to a 2.x SDK still on httpx.
if int(genai.__version__.split(".")[0]) >= 3:
    import httpx2 as _transport
else:  # google-genai 2.x, as on beta
    import httpx as _transport

_BUF = io.BytesIO()
Image.new("RGB", (4, 4), "red").save(_BUF, format="PNG")
PNG = _BUF.getvalue()

# Enough hops to tell "continued" from "did not", and a cap so a regression
# fails instead of hanging: the SDK's loop has no limit of its own.
_CONTINUED_HOPS = 3


class _StubVertex:
    """Answers every generateContent with an image and a text part. The first
    _CONTINUED_HOPS answers say CONTINUATION and carry a token, the way a reply
    cut at the per-request token limit does; after that, STOP."""

    def __init__(self):
        self.requests = 0

    def __call__(self, request):
        self.requests += 1
        candidate = {
            "content": {"role": "model", "parts": [
                {"text": '{"box_2d": [100, 100, 900, 900]}'},
                {"inlineData": {"mimeType": "image/png",
                                "data": base64.b64encode(PNG).decode()}},
            ]},
            "finishReason": "STOP",
        }
        if self.requests < _CONTINUED_HOPS:
            candidate["finishReason"] = "CONTINUATION"
            candidate["continuationToken"] = base64.b64encode(b"tok").decode()
        return _transport.Response(200, json={"candidates": [candidate]})


def _describe():
    response = TestClient(main.app).post(
        "/describe",
        files={"file": ("page.png", PNG, "image/png")},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200, response.text


# Every generate_content call site in the service, through its real code path.
CALL_SITES = {
    "stylize": lambda: stylizer.stylize(PNG, "watercolor", config.STYLE_MODEL),
    "border": lambda: border._gemini_generate_image(PNG, "a film border", config.BORDER_MODEL),
    "bbox": lambda: bbox_detector.detect_bbox(PNG, "the cat", config.BBOX_DETECTOR_MODEL),
    "gemini_image": lambda: gemini_image.generate(config.MODEL_ID, "a cat", "1:1", 1),
    "prompt_expander": lambda: prompt_expander.expand("a cat", "gemini-2.5-flash"),
    "main._one_image": lambda: main._one_image(config.MODEL_ID, "a cat", "1:1"),
    "main /describe": _describe,
}


class OneRequestPerCallTests(unittest.TestCase):
    def setUp(self):
        self.vertex = _StubVertex()
        client = genai.Client(
            vertexai=True, project="test-project", location="global",
            credentials=Credentials(token="not-a-real-token"),
            http_options=types.HttpOptions(
                client_args={"transport": _transport.MockTransport(self.vertex)}),
        )
        patcher = mock.patch.object(vertex_client, "_client", client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_continuation_reply_is_not_followed_up(self):
        for name, call in CALL_SITES.items():
            with self.subTest(call_site=name):
                self.vertex.requests = 0
                call()
                self.assertEqual(
                    self.vertex.requests, 1,
                    f"{name}: one call sent {self.vertex.requests} requests -- "
                    "the SDK followed the continuation token",
                )


class GenerateConfigTests(unittest.TestCase):
    def test_continuation_is_off_where_the_sdk_has_the_field(self):
        if not vertex_client._HAS_AUTOMATIC_CONTINUATION:
            self.skipTest("this google-genai has no automatic_continuation")
        self.assertIs(vertex_client.generate_config().automatic_continuation, False)
        config_ = vertex_client.generate_config(temperature=0, seed=0)
        self.assertEqual(
            (config_.temperature, config_.seed, config_.automatic_continuation),
            (0, 0, False),
        )

    def test_an_sdk_without_the_field_is_never_given_it(self):
        """google-genai 2.12.1, on beta, raises a ValidationError for it."""
        with mock.patch.object(vertex_client, "_HAS_AUTOMATIC_CONTINUATION", False):
            # A call that passed no config before still passes none.
            self.assertIsNone(vertex_client.generate_config())
            config_ = vertex_client.generate_config(temperature=0, seed=0)
        self.assertNotIn("automatic_continuation", config_.model_fields_set)
        self.assertEqual((config_.temperature, config_.seed), (0, 0))

    def test_the_flag_follows_the_installed_sdk(self):
        self.assertEqual(
            vertex_client._HAS_AUTOMATIC_CONTINUATION,
            "automatic_continuation" in types.GenerateContentConfig.model_fields,
        )


if __name__ == "__main__":
    unittest.main()

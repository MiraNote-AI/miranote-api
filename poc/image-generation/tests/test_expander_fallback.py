"""A failed expansion falls back to the user's words, and /health says so.

On 2026-10-15 qwen-turbo answered 429 Throttling.RateQuota to every call. The
expansion in _generate was unguarded, so sticker, background and art all became
a bare 500 after the HTTP client's ~20s of backoff -- while /health still read
clean, because dashscope_error is only the boot probe.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import httpx

import config
import main
from generate import generate_presets
from generate.http_client import ProviderError

os.environ.setdefault("BETA_TOKENS", "test-token")

_THROTTLED = ProviderError(
    'POST generation -> 429 {"code":"Throttling.RateQuota","message":'
    '"Requests rate limit exceeded, please try again later.","request_id":"r-1"}')
_USER_WORDS = "a little bear wearing a red scarf"


class ExpanderFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.dict(main._generate_status,
                                  {"last_ok_at": None, "last_error": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _request(self, method, path, **kwargs):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://probe",
            headers={"Authorization": "Bearer test-token"},
        ) as client:
            return await client.request(method, path, **kwargs)

    async def _generate(self, *, expand_art, images=lambda *a: [b"png"]):
        with mock.patch.object(main.prompt_expander, "expand_art", expand_art), \
                mock.patch.object(main.image_providers, "generate",
                                  side_effect=images) as generate:
            response = await self._request(
                "POST", "/generate", json={"command": "art", "prompt": _USER_WORDS})
        return response, generate

    async def test_a_failed_expansion_draws_the_users_own_words(self):
        def _throttled(user_input, model):
            raise _THROTTLED

        response, generate = await self._generate(expand_art=_throttled)

        self.assertEqual(response.status_code, 200)
        sent = generate.call_args.args[1]
        self.assertEqual(sent, generate_presets.build_art_prompt(_USER_WORDS))

    async def test_a_failed_expansion_is_not_retried(self):
        """A second expansion would only add another ~20s of backoff."""
        calls = []

        def _throttled(user_input, model):
            calls.append(user_input)
            raise _THROTTLED

        # A blank first image is what triggers the re-expanding retry.
        response, _ = await self._generate(expand_art=_throttled, images=[[], [b"png"]])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(calls), 1)

    async def test_health_reports_the_failure_without_the_prompt(self):
        def _throttled(user_input, model):
            raise _THROTTLED

        await self._generate(expand_art=_throttled)
        health = (await self._request("GET", "/health")).json()

        error = health["generate"]["last_error"]
        self.assertEqual(health["status"], "ok")
        self.assertEqual(error["stage"], "expand")
        self.assertEqual(error["error"],
                         f"{config.PROMPT_EXPANDER_MODEL}: 429 Throttling.RateQuota")
        self.assertNotIn(_USER_WORDS, str(health))
        self.assertNotIn("rate limit exceeded", str(health))
        self.assertIsNone(health["generate"]["last_ok_at"],
                          "a request rescued by the raw words is not a full success")

    async def test_health_reports_an_image_failure(self):
        def _expand(user_input, model):
            return "a small bear"

        def _image_down(*args):
            raise ProviderError("POST generation -> 500 {\"code\":\"InternalError\"}")

        response, _ = await self._generate(expand_art=_expand, images=_image_down)
        health = (await self._request("GET", "/health")).json()

        self.assertEqual(response.status_code, 502)
        self.assertEqual(health["generate"]["last_error"]["stage"], "image")
        self.assertEqual(health["generate"]["last_error"]["error"],
                         f"{config.IMAGE_MODELS_BY_MODE['art']}: 502 InternalError")

    async def test_a_full_success_stamps_last_ok_at(self):
        def _expand(user_input, model):
            return "a small bear"

        response, _ = await self._generate(expand_art=_expand)
        health = (await self._request("GET", "/health")).json()

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(health["generate"]["last_ok_at"])
        self.assertIsNone(health["generate"]["last_error"])


if __name__ == "__main__":
    unittest.main()

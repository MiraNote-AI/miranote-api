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
from generate import generate_presets, http_client
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


_TIMED_OUT = http_client.Timeout("POST completions timed out: ReadTimeout")


class RequiredExpansionTests(unittest.IsolatedAsyncioTestCase):
    """Sticker and background are never drawn from the user's bare words.

    Their expansion is not decoration. For a sticker it picks the saturated
    backdrop the on-device lift cuts against, and strips "sticker"/"decal",
    which otherwise draw a white-bordered paper sticker. For a background it
    turns objects into texture, so "cat" becomes a pattern rather than one big
    cat. Without it the picture is likely wrong, so the request fails with a
    retryable status instead of spending an image call on it. Art keeps the
    fallback above.
    """

    def setUp(self):
        main._generate_semaphore = None
        patcher = mock.patch.dict(main._generate_status,
                                  {"last_ok_at": None, "last_error": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _generate(self, command, expander):
        name = {"sticker": "expand", "background": "expand_background"}[command]
        with mock.patch.object(main.prompt_expander, name, side_effect=expander) as expand, \
                mock.patch.object(main.image_providers, "generate",
                                  return_value=[b"png"]) as generate:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://probe",
                headers={"Authorization": "Bearer test-token"},
            ) as client:
                response = await client.post(
                    "/generate", json={"command": command, "prompt": _USER_WORDS,
                                       "matte": "none"})
        return response, expand, generate

    async def test_a_sticker_whose_expansion_keeps_timing_out_is_not_drawn(self):
        response, expand, generate = await self._generate("sticker", _TIMED_OUT)

        self.assertEqual(response.status_code, 504)
        self.assertEqual(expand.call_count, 2, "a timed-out expansion gets one retry")
        generate.assert_not_called()
        self.assertEqual(main._generate_status["last_error"]["stage"], "expand")

    async def test_a_sticker_expansion_that_times_out_once_is_retried(self):
        response, expand, generate = await self._generate(
            "sticker", [_TIMED_OUT, "A kawaii illustration of a small bear"])

        self.assertEqual(response.status_code, 200)
        sent = generate.call_args.args[1]
        self.assertEqual(sent, generate_presets.build_sticker_prompt(
            "A kawaii illustration of a small bear"))

    async def test_a_throttled_sticker_expansion_answers_busy_and_is_not_drawn(self):
        response, expand, generate = await self._generate("sticker", _THROTTLED)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(expand.call_count, 1,
                         "http_client already backed off; only a timeout is retried")
        generate.assert_not_called()

    async def test_a_sticker_expansion_that_hands_back_the_users_words_is_a_failure(self):
        """prompt_expander returns the user's input when the model sends no
        text (a safety block, an empty answer) -- the same bare words by
        another route."""
        response, _, generate = await self._generate("sticker", lambda *a: _USER_WORDS)

        self.assertEqual(response.status_code, 502)
        generate.assert_not_called()

    async def test_a_background_whose_expansion_keeps_timing_out_is_not_drawn(self):
        response, _, generate = await self._generate("background", _TIMED_OUT)

        self.assertEqual(response.status_code, 504)
        generate.assert_not_called()

    async def test_no_retry_when_the_budget_cannot_cover_it(self):
        with mock.patch.object(config, "GENERATE_DEADLINE", config.EXPANDER_TIMEOUT):
            response, expand, generate = await self._generate("sticker", _TIMED_OUT)

        self.assertEqual(response.status_code, 504)
        self.assertEqual(expand.call_count, 1)
        generate.assert_not_called()

    async def test_art_still_draws_the_users_words_after_a_timeout(self):
        with mock.patch.object(main.prompt_expander, "expand_art",
                               side_effect=_TIMED_OUT), \
                mock.patch.object(main.image_providers, "generate",
                                  return_value=[b"png"]) as generate:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://probe",
                headers={"Authorization": "Bearer test-token"},
            ) as client:
                response = await client.post(
                    "/generate", json={"command": "art", "prompt": _USER_WORDS})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(generate.call_args.args[1],
                         generate_presets.build_art_prompt(_USER_WORDS))


if __name__ == "__main__":
    unittest.main()

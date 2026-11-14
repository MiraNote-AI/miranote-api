"""/generate ends within one deadline, and says which step used it up.

Before this, every DashScope call had its own generous read timeout (60 s for
the expander and the download, 180 s for the image call) and up to four
attempts, with nothing bounding the whole request. One stalled upstream call
kept a phone spinning far past its own 60 s sticker limit, and kept holding one
of the three /generate slots after the phone had given up -- so the retry the
phone offered queued behind the request it had just abandoned (ios #169).

The deadline rides in a context variable rather than a parameter, so the
provider modules and every test that fakes _call_model keep their signatures.

No network: requests is faked at the http_client boundary.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

import httpx
import requests

import config
import main
from generate import dashscope_text, http_client

os.environ.setdefault("BETA_TOKENS", "test-token")
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")


def _response(status: int, body: dict | None = None) -> mock.Mock:
    response = mock.Mock(status_code=status, ok=200 <= status < 300,
                         text=str(body or {}), content=b"png")
    response.json.return_value = body or {}
    return response


class BudgetTests(unittest.TestCase):
    """http_client on its own."""

    def test_no_budget_leaves_the_read_timeout_alone(self):
        with mock.patch.object(http_client.requests, "request",
                               return_value=_response(200)) as request:
            http_client.post_json("https://x/generation", {}, "DASHSCOPE_API_KEY",
                                  timeout=60.0)
        connect, read = request.call_args.kwargs["timeout"]
        self.assertEqual(read, 60.0)
        self.assertLessEqual(connect, http_client.CONNECT_TIMEOUT)

    def test_a_budget_shortens_the_read_timeout(self):
        with mock.patch.object(http_client.requests, "request",
                               return_value=_response(200)) as request, \
                http_client.budget(5.0, {}):
            http_client.post_json("https://x/generation", {}, "DASHSCOPE_API_KEY",
                                  timeout=180.0)
        _, read = request.call_args.kwargs["timeout"]
        self.assertLessEqual(read, 5.0)

    def test_a_read_timeout_becomes_a_timeout_error(self):
        with mock.patch.object(http_client.requests, "request",
                               side_effect=requests.ReadTimeout("slow")):
            with self.assertRaises(http_client.Timeout):
                http_client.post_json("https://x/generation", {},
                                      "DASHSCOPE_API_KEY")

    def test_a_spent_budget_never_sends(self):
        with mock.patch.object(http_client.requests, "request") as request, \
                http_client.budget(0.0, {}):
            with self.assertRaises(http_client.Timeout):
                http_client.post_json("https://x/generation", {},
                                      "DASHSCOPE_API_KEY")
        request.assert_not_called()

    def test_no_backoff_sleep_the_budget_cannot_cover(self):
        """A 429 is retried after 2-3 s; with 1 s left that sleep only delays
        the same failure, so the error goes back at once."""
        with mock.patch.object(http_client.requests, "request",
                               return_value=_response(429, {"code": "Throttling"})) as request, \
                mock.patch.object(http_client.time, "sleep") as sleep, \
                http_client.budget(1.0, {}):
            with self.assertRaises(http_client.ProviderError):
                http_client.post_json("https://x/generation", {},
                                      "DASHSCOPE_API_KEY")
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_a_download_timeout_becomes_a_timeout_error(self):
        with mock.patch.object(http_client.requests, "get",
                               side_effect=requests.ReadTimeout("slow")):
            with self.assertRaises(http_client.Timeout):
                http_client.fetch_image("https://oss/img.png")

    def test_timed_adds_up_repeated_steps(self):
        timings = {}
        with http_client.budget(10.0, timings):
            with http_client.timed("model_request"):
                pass
            with http_client.timed("model_request"):
                pass
        self.assertIn("model_request", timings)

    def test_timed_without_a_budget_is_a_no_op(self):
        with http_client.timed("model_request"):
            pass  # must not raise

    def test_expander_gets_its_own_short_cap(self):
        with mock.patch.object(dashscope_text.http_client, "post_json",
                               return_value={"choices": [{"message": {"content": "x"}}]}) as post:
            dashscope_text.complete("a cat", "qwen3.8-flash")
        self.assertEqual(post.call_args.kwargs["timeout"], config.EXPANDER_TIMEOUT)
        self.assertLessEqual(post.call_args.kwargs["attempts"], 2)


class GenerateDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        main._generate_semaphore = None
        patcher = mock.patch.dict(main._generate_status,
                                  {"last_ok_at": None, "last_error": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _post(self, payload=None):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://probe",
            headers={"Authorization": "Bearer test-token"},
        ) as client:
            return await client.post(
                "/generate", json=payload or {"command": "sticker", "prompt": "a cat",
                                              "expand": False, "matte": "none"})

    async def test_an_image_call_out_of_time_answers_504(self):
        def _stalled(*args):
            raise http_client.Timeout("POST generation timed out")

        with mock.patch.object(main.image_providers, "generate", side_effect=_stalled):
            response = await self._post()

        self.assertEqual(response.status_code, 504)
        self.assertEqual(main._generate_status["last_error"]["stage"], "image")

    async def test_the_deadline_reaches_the_image_call(self):
        """The image call runs two threads away from the handler (to_thread,
        then _call_model's pool); the budget has to arrive there."""
        seen = []

        def _record(*args):
            seen.append(http_client.remaining())
            return [b"png"]

        with mock.patch.object(main.image_providers, "generate", side_effect=_record):
            response = await self._post()

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(seen[0])
        self.assertLessEqual(seen[0], config.GENERATE_DEADLINE)

    async def test_waiting_for_a_slot_counts_against_the_deadline(self):
        gate = main._generate_gate()
        for _ in range(main.GENERATE_CONCURRENCY):
            await gate.acquire()
        try:
            with mock.patch.object(config, "GENERATE_DEADLINE", 0.2), \
                    mock.patch.object(main.image_providers, "generate",
                                      return_value=[b"png"]) as generate:
                started = time.perf_counter()
                response = await self._post()
                elapsed = time.perf_counter() - started
        finally:
            for _ in range(main.GENERATE_CONCURRENCY):
                gate.release()

        self.assertEqual(response.status_code, 504)
        self.assertLess(elapsed, 2.0)
        generate.assert_not_called()
        self.assertEqual(main._generate_status["last_error"]["stage"], "queue")

    async def test_a_timed_out_request_gives_its_slot_back(self):
        def _stalled(*args):
            raise http_client.Timeout("POST generation timed out")

        with mock.patch.object(main.image_providers, "generate", side_effect=_stalled):
            for _ in range(main.GENERATE_CONCURRENCY + 1):
                await self._post()
        with mock.patch.object(main.image_providers, "generate", return_value=[b"png"]):
            response = await asyncio.wait_for(self._post(), timeout=2.0)
        self.assertEqual(response.status_code, 200)


class GenerateTimingTests(unittest.IsolatedAsyncioTestCase):
    """One log line per request, and the same numbers on the response, so a
    slow request can be pinned to a step from the beta Mac's log alone."""

    def setUp(self):
        main._generate_semaphore = None

    async def _post(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://probe",
            headers={"Authorization": "Bearer test-token"},
        ) as client:
            return await client.post(
                "/generate", json={"command": "art", "prompt": "a lighthouse",
                                   "expand": False})

    async def test_the_response_carries_per_step_timings(self):
        with mock.patch.object(main.image_providers, "generate", return_value=[b"png"]):
            response = await self._post()

        self.assertEqual(response.status_code, 200)
        timings = response.json()["timings"]
        for step in ("queue", "expand", "model", "total"):
            self.assertIn(step, timings)

    async def test_every_request_logs_one_line_even_when_it_fails(self):
        outcomes = ((mock.Mock(return_value=[b"png"]), 200),
                    (mock.Mock(side_effect=http_client.Timeout("timed out")), 504))
        for fake, status in outcomes:
            out = io.StringIO()
            with mock.patch.object(main.image_providers, "generate", fake), \
                    redirect_stdout(out):
                await self._post()
            lines = [line for line in out.getvalue().splitlines()
                     if line.startswith("[generate] ") and " total=" in line]
            self.assertEqual(len(lines), 1, out.getvalue())
            self.assertIn(f" art {status} ", lines[0])


if __name__ == "__main__":
    unittest.main()

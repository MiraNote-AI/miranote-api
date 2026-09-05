"""Concurrency regression tests for /generate.

Background removal is CPU-bound. When it runs unwrapped inside the async
handler it blocks the event loop, freezing every other in-flight request
including /health. These tests pin that it does not, and that concurrent
generations are capped rather than allowed to saturate the machine.

The rembg call is replaced with a sleep of the same shape: a synchronous,
GIL-releasing wait. That is what makes the difference between "blocks the
loop" and "runs on a worker thread" observable without generating images.
"""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import List
from unittest import mock

import httpx

import main


BLOCKING_SECONDS = 0.4
IMAGES_PER_REQUEST = 2
SAMPLE_INTERVAL = 0.005
SETTLE = 0.05


def _fake_call_model(prompt, aspect_ratio):
    return [b"first-image", b"second-image"]


class _ConcurrencyProbe:
    """A stand-in for rembg.remove that records how many run at once."""

    def __init__(self, seconds=BLOCKING_SECONDS):
        self.seconds = seconds
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def __call__(self, raw, session=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.seconds)
            return raw
        finally:
            with self._lock:
                self.active -= 1


def _sticker_payload(prompt="a cat"):
    return {"command": "sticker", "prompt": prompt, "expand": False}


class GenerateConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Each test runs on its own event loop, and an asyncio.Semaphore binds
        # to the first loop that blocks on it. Drop the module-level instance
        # so every test builds its own rather than inheriting a stale binding.
        main._generate_semaphore = None

    def _client(self):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://probe"
        )

    async def test_generate_does_not_stall_concurrent_health_checks(self):
        """/health must keep answering while a sticker generation runs.

        Two details here are load-bearing and must not be tidied away.

        The heartbeat sleeps explicitly. Requests through ASGITransport are
        in-process and never suspend, so a loop that only awaits them never
        yields to the scheduler and starves every other task -- including the
        generation it is supposed to be observing.

        The heartbeat is also given time to take a sample after the generation
        returns, before it is stopped. A stall is measured as the gap between
        two samples, so it is only visible if a sample lands on each side of
        it. Stopping the heartbeat the instant the generation finishes hides
        the very stall the test exists to catch, and the test passes against
        the unfixed code.
        """
        probe = _ConcurrencyProbe()
        stamps: List[float] = []
        stop = asyncio.Event()

        with mock.patch.object(main, "remove", probe), mock.patch.object(
            main, "_call_model", _fake_call_model
        ):
            async with self._client() as client:

                async def heartbeat():
                    while not stop.is_set():
                        response = await client.get("/health")
                        self.assertEqual(response.status_code, 200)
                        stamps.append(time.perf_counter())
                        await asyncio.sleep(SAMPLE_INTERVAL)

                ticker = asyncio.create_task(heartbeat())
                await asyncio.sleep(SETTLE)  # establish a baseline rhythm
                generation = await client.post("/generate", json=_sticker_payload())
                await asyncio.sleep(SETTLE)  # let a sample land after the work
                stop.set()
                await ticker

        self.assertEqual(generation.status_code, 200)
        self.assertGreater(len(stamps), 2, "heartbeat never established a baseline")

        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        worst = max(gaps)
        self.assertLess(
            worst,
            BLOCKING_SECONDS / 2,
            "/health stalled for {:.2f}s while /generate ran; background removal "
            "is blocking the event loop".format(worst),
        )

    async def test_concurrent_generations_do_not_serialise(self):
        """Three generations must overlap, not queue behind one another."""
        probe = _ConcurrencyProbe()

        with mock.patch.object(main, "remove", probe), mock.patch.object(
            main, "_call_model", _fake_call_model
        ):
            async with self._client() as client:
                started = time.perf_counter()
                responses = await asyncio.gather(
                    *[
                        client.post("/generate", json=_sticker_payload())
                        for _ in range(3)
                    ]
                )
                elapsed = time.perf_counter() - started

        for response in responses:
            self.assertEqual(response.status_code, 200)

        serial = 3 * IMAGES_PER_REQUEST * BLOCKING_SECONDS
        self.assertLess(
            elapsed,
            serial / 2,
            "three generations took {:.2f}s against a serial cost of {:.2f}s; "
            "they are not running concurrently".format(elapsed, serial),
        )

    async def test_generation_concurrency_is_capped(self):
        """More requests than the cap must wait rather than pile on."""
        probe = _ConcurrencyProbe(seconds=0.2)
        cap = main.GENERATE_CONCURRENCY

        with mock.patch.object(main, "remove", probe), mock.patch.object(
            main, "_call_model", _fake_call_model
        ):
            async with self._client() as client:
                responses = await asyncio.gather(
                    *[
                        client.post("/generate", json=_sticker_payload())
                        for _ in range(cap + 3)
                    ]
                )

        for response in responses:
            self.assertEqual(response.status_code, 200)
        self.assertLessEqual(
            probe.max_active,
            cap,
            "{} removals ran at once against a cap of {}".format(
                probe.max_active, cap
            ),
        )

    async def test_cap_is_released_when_the_handler_fails(self):
        """A failed generation must not leak its slot."""
        cap = main.GENERATE_CONCURRENCY

        def explode(prompt, aspect_ratio):
            raise RuntimeError("vertex is down")

        with mock.patch.object(main, "_call_model", explode):
            async with self._client() as client:
                for _ in range(cap + 1):
                    with self.assertRaises(RuntimeError):
                        await client.post("/generate", json=_sticker_payload())

        self.assertEqual(
            main._generate_semaphore._value,
            cap,
            "the semaphore did not return to full capacity after failures",
        )


if __name__ == "__main__":
    unittest.main()

"""Tests for the shared beta bearer-token auth layer.

The token ships inside a TestFlight build and can be extracted from the IPA,
so it is treated as public: the rate limit, not the secrecy of the token, is
what bounds the damage. These tests pin both halves.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import beta_auth


class _Clock:
    """A hand-cranked replacement for time.monotonic."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _build_app():
    app = FastAPI(dependencies=[Depends(beta_auth.require_beta_token)])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/work")
    async def work():
        return {"done": True}

    return app


class BetaAuthTests(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self._limiter_patch = mock.patch.object(
            beta_auth,
            "_limiter",
            beta_auth.TokenRateLimiter(limit=3, window=60, clock=self.clock),
        )
        self._limiter_patch.start()
        self.addCleanup(self._limiter_patch.stop)

        self._env_patch = mock.patch.dict(
            os.environ, {"BETA_TOKENS": "alpha-token,beta-token"}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        self.client = TestClient(_build_app())

    def _work(self, token=None):
        headers = {"Authorization": "Bearer " + token} if token else {}
        return self.client.post("/work", headers=headers)

    def test_request_without_a_token_is_rejected(self):
        self.assertEqual(self._work().status_code, 401)

    def test_request_with_an_unknown_token_is_rejected(self):
        self.assertEqual(self._work("not-a-real-token").status_code, 401)

    def test_every_configured_token_is_accepted(self):
        for token in ("alpha-token", "beta-token"):
            with self.subTest(token=token):
                self.assertEqual(self._work(token).status_code, 200)

    def test_surrounding_whitespace_in_the_token_list_is_ignored(self):
        with mock.patch.dict(os.environ, {"BETA_TOKENS": " alpha-token , beta-token "}):
            self.assertEqual(self._work("beta-token").status_code, 200)

    def test_an_empty_token_list_accepts_nothing(self):
        with mock.patch.dict(os.environ, {"BETA_TOKENS": ""}):
            self.assertEqual(self._work("").status_code, 401)
            self.assertEqual(self._work("alpha-token").status_code, 401)

    def test_a_malformed_authorization_header_is_rejected(self):
        for header in ("alpha-token", "Basic alpha-token", "Bearer", "Bearer  "):
            with self.subTest(header=header):
                response = self.client.post("/work", headers={"Authorization": header})
                self.assertEqual(response.status_code, 401)

    def test_health_needs_no_token(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_health_is_never_rate_limited(self):
        for _ in range(10):
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_requests_beyond_the_limit_are_rejected(self):
        for _ in range(3):
            self.assertEqual(self._work("alpha-token").status_code, 200)
        self.assertEqual(self._work("alpha-token").status_code, 429)

    def test_the_limit_is_counted_per_token_not_globally(self):
        for _ in range(3):
            self.assertEqual(self._work("alpha-token").status_code, 200)
        self.assertEqual(self._work("alpha-token").status_code, 429)
        self.assertEqual(
            self._work("beta-token").status_code,
            200,
            "one token exhausting its budget must not lock out the others",
        )

    def test_the_window_recovers(self):
        for _ in range(3):
            self._work("alpha-token")
        self.assertEqual(self._work("alpha-token").status_code, 429)

        self.clock.advance(61)
        self.assertEqual(
            self._work("alpha-token").status_code,
            200,
            "the window did not roll forward",
        )

    def test_the_window_slides_rather_than_resetting_in_blocks(self):
        """A fixed-bucket counter would allow a double burst across a boundary."""
        for _ in range(3):
            self._work("alpha-token")
        self.clock.advance(30)
        self.assertEqual(
            self._work("alpha-token").status_code,
            429,
            "requests from 30s ago are still inside a 60s window",
        )

    def test_only_expired_requests_leave_the_window(self):
        """Expiry drops the requests that aged out, not the whole history.

        A limiter that clears its record once the oldest entry expires passes
        every other test here while allowing a full fresh burst too early.
        """
        self._work("alpha-token")
        self.clock.advance(50)
        self._work("alpha-token")
        self._work("alpha-token")
        self.assertEqual(self._work("alpha-token").status_code, 429)

        self.clock.advance(11)  # only the first request has aged out
        self.assertEqual(
            self._work("alpha-token").status_code,
            200,
            "the expired request did not free a slot",
        )
        self.assertEqual(
            self._work("alpha-token").status_code,
            429,
            "the two requests from 11s ago must still occupy the window",
        )

    def test_rejection_names_the_scheme(self):
        response = self._work()
        self.assertEqual(response.headers.get("WWW-Authenticate"), "Bearer")


if __name__ == "__main__":
    unittest.main()

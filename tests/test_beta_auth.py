"""Tests for the shared beta bearer-token auth layer.

The token ships inside a TestFlight build and can be extracted from the IPA,
so it is treated as public: the rate limit, not the secrecy of the token, is
what bounds the damage. These tests pin both halves.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import pathlib
import tempfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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


def _build_app(static_dir=None):
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/work")
    async def work():
        return {"done": True}

    beta_auth.install(app)

    # Mounted after the gate on purpose: two of the four services mount a UI,
    # and voice-to-text mounts at "/" where it catches every unrouted path.
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")

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

    def test_mounted_files_are_gated_too(self):
        """The reason this is middleware and not a route dependency.

        A mounted sub-application does not inherit FastAPI(dependencies=[...]):
        measured, the route answers 401 while the mounted file answers 200 and
        serves its contents. voice-to-text mounts at "/" with html=True, so a
        dependency would leave every unrouted path public.
        """
        directory = tempfile.mkdtemp()
        pathlib.Path(directory, "index.html").write_text("<h1>ui</h1>")
        client = TestClient(_build_app(static_dir=directory))

        for path in ("/index.html", "/"):
            with self.subTest(path=path):
                anonymous = client.get(path)
                self.assertEqual(anonymous.status_code, 401)
                self.assertNotIn("<h1>ui</h1>", anonymous.text)

        allowed = client.get(
            "/index.html", headers={"Authorization": "Bearer alpha-token"}
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertIn("<h1>ui</h1>", allowed.text)

    def test_a_rejection_is_never_a_server_error(self):
        """Raising HTTPException inside middleware yields 500, not the status.

        Measured: the middleware runs outside the exception handlers, so a
        raise turns every rejection into a server error. The gate must return
        a response rather than raise one.
        """
        self.assertEqual(self._work().status_code, 401)
        self.assertEqual(self._work("not-a-real-token").status_code, 401)
        for _ in range(3):
            self._work("alpha-token")
        self.assertEqual(self._work("alpha-token").status_code, 429)

    def test_installing_reports_how_many_tokens_are_live(self):
        with mock.patch("builtins.print") as printed:
            beta_auth.install(FastAPI())
        said = " ".join(str(c) for c in printed.call_args_list)
        self.assertIn("2", said, "did not report the token count")
        self.assertNotIn("alpha-token", said, "printed a token value")

    def test_installing_without_tokens_warns_loudly(self):
        with mock.patch.dict(os.environ, {"BETA_TOKENS": ""}):
            with mock.patch("builtins.print") as printed:
                beta_auth.install(FastAPI())
        said = " ".join(str(c) for c in printed.call_args_list).upper()
        self.assertIn("WARNING", said, "an empty token list must be announced")

    def test_cors_preflight_survives_the_gate(self):
        """install() must run before CORSMiddleware is added.

        Starlette makes the most recently added middleware outermost. With the
        gate outside CORS it sees the browser preflight first, and a preflight
        carries no Authorization header, so it is rejected with 401 and every
        cross-origin caller breaks. Measured both ways: gate-outside gives 401
        on OPTIONS, CORS-outside gives 200.
        """
        preflight = {
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }

        correct = FastAPI()

        @correct.post("/work")
        async def work_ok():
            return {"done": True}

        beta_auth.install(correct)
        correct.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
        )

        client = TestClient(correct)
        self.assertEqual(client.options("/work", headers=preflight).status_code, 200)
        self.assertEqual(client.post("/work").status_code, 401)
        self.assertEqual(
            client.post("/work", headers={"Authorization": "Bearer alpha-token"}).status_code,
            200,
        )

    def test_the_default_limiter_honours_the_configured_limit(self):
        """Changing RATE_LIMIT_REQUESTS must actually change behaviour.

        Every other limiter test injects its own limit, so nothing here
        exercised the module-level instance. A value hardcoded inside
        TokenRateLimiter would leave the constant decorative and make raising
        it silently do nothing.
        """
        limiter = beta_auth.TokenRateLimiter(clock=self.clock)
        allowed = 0
        while limiter.allow("alpha-token"):
            allowed += 1
            if allowed > beta_auth.RATE_LIMIT_REQUESTS * 2:
                self.fail("the default limiter never refused a request")
        self.assertEqual(allowed, beta_auth.RATE_LIMIT_REQUESTS)

    def test_the_limit_leaves_room_for_a_shared_token(self):
        """TestFlight ships one binary, so every tester shares this budget.

        Ten testers holding a conversation is the load this has to absorb;
        /chat is one request per message. The expensive endpoint does not rely
        on this limit -- /generate is bounded by its own semaphore -- so the
        headroom costs little.
        """
        testers = 10
        messages_per_minute_each = 6
        self.assertGreaterEqual(
            beta_auth.RATE_LIMIT_REQUESTS,
            testers * messages_per_minute_each,
            "ten testers on one token would be throttled while behaving normally",
        )

    def test_rejection_names_the_scheme(self):
        response = self._work()
        self.assertEqual(response.headers.get("WWW-Authenticate"), "Bearer")


if __name__ == "__main__":
    unittest.main()

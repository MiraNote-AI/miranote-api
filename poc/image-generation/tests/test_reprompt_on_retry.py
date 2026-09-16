"""The retry sends a different prompt, not the same one again.

#79's retry reused the expanded prompt, because expansion happens once in
_generate before _call_model is reached. Live on 2026-09-15 it fired twice and
recovered nothing, both times with attempt 1 and attempt 2 empty at
finish_reason STOP. The same user input succeeded on other requests, so the
blank is not deterministic per input -- but a retry that cannot vary the string
it sends has no way to find that out (#80).

Expansion runs on a text model, metered separately from the 2/min image quota,
so varying the prompt costs nothing that was scarce.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import HTTPException
from google.genai.types import FinishReason

import config
import main

os.environ.setdefault("BETA_TOKENS", "test-token")


def _empty(finish_reason=FinishReason.STOP):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[]),
        finish_reason=finish_reason,
        safety_ratings=None,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=None)


def _image(data=b"png"):
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(data=data))]),
        finish_reason=FinishReason.STOP,
        safety_ratings=None,
    )
    return SimpleNamespace(candidates=[candidate], prompt_feedback=None)


def _client(*responses):
    client = mock.Mock()
    client.models.generate_content.side_effect = list(responses)
    return client


def _prompts_sent(client) -> list[str]:
    return [c.kwargs["contents"] for c in client.models.generate_content.call_args_list]


class CallModelRepromptTests(unittest.TestCase):
    def test_the_retry_sends_what_the_reprompt_produced(self):
        client = _client(_empty(), _image())
        with mock.patch.object(main, "_get_client", return_value=client):
            main._call_model("first prompt", "1:1", reprompt=lambda: "second prompt")
        sent = _prompts_sent(client)
        self.assertIn("first prompt", sent[0])
        self.assertIn("second prompt", sent[1])
        self.assertNotIn("first prompt", sent[1], "the retry repeated the first prompt")

    def test_without_a_reprompt_the_retry_repeats_the_prompt(self):
        """expand=false has nothing to vary; behaviour there is unchanged."""
        client = _client(_empty(), _image())
        with mock.patch.object(main, "_get_client", return_value=client):
            main._call_model("only prompt", "1:1", reprompt=None)
        sent = _prompts_sent(client)
        self.assertIn("only prompt", sent[0])
        self.assertIn("only prompt", sent[1])

    def test_a_refusal_never_reaches_the_reprompt(self):
        """No second image call means no reason to spend a text call either."""
        calls = []
        client = _client(_empty(FinishReason.IMAGE_SAFETY), _image())
        with mock.patch.object(main, "_get_client", return_value=client):
            with self.assertRaises(HTTPException):
                main._call_model("p", "1:1", reprompt=lambda: calls.append(1) or "unused")
        self.assertEqual(calls, [], "a refusal triggered a re-expansion")

    def test_a_first_attempt_that_works_never_reaches_the_reprompt(self):
        calls = []
        client = _client(_image())
        with mock.patch.object(main, "_get_client", return_value=client):
            main._call_model("p", "1:1", reprompt=lambda: calls.append(1) or "unused")
        self.assertEqual(calls, [])

    def test_a_failing_reprompt_does_not_cost_the_retry(self):
        """The expander is a network call. Losing it must not lose the attempt."""
        def _boom():
            raise RuntimeError("expander unavailable")

        client = _client(_empty(), _image(b"recovered"))
        with mock.patch.object(main, "_get_client", return_value=client):
            result = main._call_model("original", "1:1", reprompt=_boom)
        self.assertEqual(result, [b"recovered"] * config.NUMBER_OF_IMAGES)
        self.assertIn("original", _prompts_sent(client)[1])


class GenerateWiresTheRepromptTests(unittest.IsolatedAsyncioTestCase):
    """_call_model accepting a reprompt is worth nothing if /generate omits it."""

    async def _post(self, client_stub, *, expand):
        with mock.patch.object(main, "_get_client", return_value=client_stub):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://probe",
                headers={"Authorization": "Bearer test-token"},
            ) as client:
                return await client.post(
                    "/generate",
                    json={"command": "art", "prompt": "a fox", "expand": expand},
                )

    async def test_expansion_runs_again_for_the_retry(self):
        expansions = []

        def _expand(user_input, model):
            expansions.append(user_input)
            return f"expanded {len(expansions)}"

        stub = _client(_empty(), _image())
        with mock.patch.object(main.prompt_expander, "expand_art", _expand):
            response = await self._post(stub, expand=True)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(expansions), 2, "the retry reused the first expansion")
        sent = _prompts_sent(stub)
        self.assertIn("expanded 1", sent[0])
        self.assertIn("expanded 2", sent[1])

    async def test_no_expansion_is_requested_when_the_caller_did_not_ask(self):
        expansions = []

        def _expand(user_input, model):
            expansions.append(user_input)
            return "unused"

        stub = _client(_empty(), _image())
        with mock.patch.object(main.prompt_expander, "expand_art", _expand):
            response = await self._post(stub, expand=False)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(expansions, [])


if __name__ == "__main__":
    unittest.main()

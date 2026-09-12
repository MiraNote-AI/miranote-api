"""The beta gate is actually attached to this service.

This is the service that spends Vertex credits per request, so an unwired gate
here is the most expensive of the four to get wrong.
"""

from __future__ import annotations

import os
import unittest

import httpx

import main

os.environ["BETA_TOKENS"] = "test-token"


class BetaGateTests(unittest.IsolatedAsyncioTestCase):
    def _anonymous(self):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://probe"
        )

    async def test_an_endpoint_needs_a_token(self):
        async with self._anonymous() as client:
            response = await client.post(
                "/generate", json={"command": "sticker", "prompt": "a cat"}
            )
        self.assertEqual(response.status_code, 401)

    async def test_health_needs_no_token(self):
        async with self._anonymous() as client:
            response = await client.get("/health")
        self.assertEqual(response.status_code, 200)

    async def test_the_rejection_names_the_scheme(self):
        async with self._anonymous() as client:
            response = await client.post("/generate", json={"command": "sticker"})
        self.assertEqual(response.headers.get("WWW-Authenticate"), "Bearer")


if __name__ == "__main__":
    unittest.main()

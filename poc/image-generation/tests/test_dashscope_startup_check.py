"""DashScope is probed at startup, and a failure is loud but not fatal.

http_client.api_key() reads the environment only when a request is already in
flight. Without a startup probe a host that forgot DASHSCOPE_API_KEY boots
clean, answers /health 200, reports its image models, and then 502s every
/generate -- while the app tells the user the AI server is not running, which
is false. That is the failure the voice service lived with for weeks (#73)
before #77 gave it a startup check; this is the same fix for the same shape.

Not fatal on purpose: /cutout, /stylize, /describe and /border never touch
DashScope, and taking four endpoints down over one of them is the mistake the
Vision-helper block already avoids.
"""

from __future__ import annotations

import inspect
import os
import unittest
from unittest import mock

import config
import main

os.environ["BETA_TOKENS"] = "test-token"


class ProbeShapeTests(unittest.TestCase):
    def test_lifespan_actually_calls_dashscope(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("dashscope_text.complete", source,
                      "a presence check cannot tell a missing key from a "
                      "revoked one or an account in arrears; only a round "
                      "trip can, which is why voice-to-text does the same")

    def test_the_probe_cannot_stop_the_server(self):
        source = inspect.getsource(main.lifespan)
        probe = source[source.index("global _dashscope_error"):]
        before_yield = probe.split("yield")[0]
        self.assertIn("except Exception", before_yield)
        self.assertNotIn("raise", before_yield,
                         "/cutout, /stylize, /describe and /border do not use "
                         "DashScope and must survive its absence")

    def test_a_failure_is_loud_and_says_what_breaks(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("DASHSCOPE NOT WORKING", source)
        self.assertIn("502", source,
                      "naming the symptom is what lets whoever reads the log "
                      "connect it to what the app will show")

    def test_health_reports_the_probe_result(self):
        self.assertIn("dashscope_error", inspect.getsource(main.health),
                      "the log line scrolls away; /health is what a deploy "
                      "script can check")


class ConfiguredDetectionTests(unittest.TestCase):
    """The rollback must not warn about a credential it will not read."""

    def test_dashscope_image_models_need_the_key(self):
        with mock.patch.object(config, "IMAGE_MODELS_BY_MODE",
                               {"sticker": "z-image-turbo"}):
            self.assertTrue(main._dashscope_is_configured())

    def test_a_qwen_expander_alone_needs_the_key(self):
        """The Gemini rollback moves the image models and leaves the expander,
        so asking only about IMAGE_MODEL would miss this."""
        with mock.patch.object(config, "IMAGE_MODELS_BY_MODE",
                               {"sticker": config.MODEL_ID}), \
             mock.patch.object(config, "PROMPT_EXPANDER_MODEL", "qwen-turbo"):
            self.assertTrue(main._dashscope_is_configured())

    def test_a_full_rollback_to_gemini_does_not(self):
        with mock.patch.object(config, "IMAGE_MODELS_BY_MODE",
                               {"sticker": config.MODEL_ID}), \
             mock.patch.object(config, "PROMPT_EXPANDER_MODEL",
                               "gemini-2.5-flash"):
            self.assertFalse(main._dashscope_is_configured())

    def test_the_shipped_default_needs_the_key(self):
        """Guards the real config, not a mock of it."""
        self.assertTrue(main._dashscope_is_configured())


if __name__ == "__main__":
    unittest.main()

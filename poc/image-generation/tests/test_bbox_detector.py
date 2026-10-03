"""The bbox detector reads the model it is given, and reads its answer honestly.

This path had no test at all until the detector model moved off
gemini-2.5-flash, which is a poor combination: main.py degrades to dino-only
whenever the detector returns nothing, so a detector that silently stopped
working would keep serving cutouts and no test would notice.

Two things are pinned here. That the endpoint calls the CONFIGURED model rather
than a hardcoded one -- the reason BBOX_DETECTOR_MODEL exists is to make an A/B
a restart instead of an edit, and that is worthless if the value is not the one
used. And that _parse treats a malformed answer as "no box" rather than
crashing or inventing one: gemini-2.5-flash was observed answering
'{"box": [199, 13, [969, 572]}' on a real image, and the three response schemas
below were all seen from the same model on the same bytes.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import config
import inspect
import main
from cutout import bbox_detector

os.environ["BETA_TOKENS"] = "test-token"


def _response(text: str):
    return mock.Mock(text=text)


class ConfiguredModelTests(unittest.TestCase):
    def test_the_model_argument_is_the_one_called(self):
        client = mock.Mock()
        client.models.generate_content.return_value = _response('{"box": [1, 2, 3, 4]}')
        with mock.patch.object(bbox_detector, "_get_client", return_value=client):
            bbox_detector.detect_bbox(b"\x89PNG_bytes", "a cat", "some-model-id")
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["model"], "some-model-id"
        )

    def test_the_default_is_a_vision_model_not_an_image_one(self):
        """A detector reads an image and writes JSON. Pointing this at an
        "-image" id would ask a model that draws to answer in text, which is
        how PROMPT_EXPANDER_MODEL nearly broke /describe (see config.py)."""
        self.assertFalse(config.BBOX_DETECTOR_MODEL.endswith("-image"))

    def test_the_env_var_overrides_the_default(self):
        """The ladder switch: an A/B arm is a restart, not an edit to config.py."""
        import importlib
        with mock.patch.dict(os.environ, {"BBOX_DETECTOR_MODEL": "gemini-2.5-flash"}):
            reloaded = importlib.reload(config)
            self.assertEqual(reloaded.BBOX_DETECTOR_MODEL, "gemini-2.5-flash")
        importlib.reload(config)   # leave the module as the rest of the suite expects


class SystemPromptTests(unittest.TestCase):
    """The detector's instructions, pinned where getting them wrong is silent.

    A prompt defect does not raise: it returns a plausible box for the wrong
    extent, the cutout comes back truncated, and the request still answers 200.
    That is how "the strawberry parfait glass" lost its strawberries -- the
    model boxed the glass, correctly reading an instruction that only said
    "the requested object".

    These pin the clauses, not the model's behaviour. No unit test can promise
    a model obeys; what they buy is that nobody removes a clause while tidying
    without the removal showing up here. Measured effect of the whole-object
    clause over the 17-case set: 06_parfait +73% of its subject back, the other
    16 cutouts pixel-identical.
    """

    def test_the_whole_object_clause_is_present(self):
        prompt = bbox_detector._SYSTEM_PROMPT.lower()
        self.assertIn("whole", prompt)
        self.assertIn("contents", prompt)

    def test_the_largest_instance_rule_survives_alongside_it(self):
        """The two rules are about different things -- which instance to pick
        vs how much of it to cover -- and the second was added later. A reader
        could mistake them for duplicates and drop one."""
        self.assertIn("LARGEST", bbox_detector._SYSTEM_PROMPT)

    def test_the_not_visible_escape_hatch_survives(self):
        """Without it the model invents a box for an absent target, and
        main.py has no way to answer 404."""
        self.assertIn("{}", bbox_detector._SYSTEM_PROMPT)

    def test_the_target_placeholder_is_still_substituted(self):
        """An unsubstituted {{TARGET}} would ask every image the same
        question and still parse cleanly, which is the silent kind of wrong."""
        self.assertIn("{{TARGET}}", bbox_detector._SYSTEM_PROMPT)
        built = bbox_detector._SYSTEM_PROMPT.replace("{{TARGET}}", "a cat")
        self.assertNotIn("{{TARGET}}", built)
        self.assertIn("a cat", built)


class ParseTests(unittest.TestCase):
    def test_every_schema_the_models_actually_emit_is_read(self):
        """All three were observed from gemini-2.5-flash on identical bytes."""
        for raw in ('{"box": [100, 200, 300, 400]}',
                    '{"box_2d": [100, 200, 300, 400]}',
                    '{"y_min": 100, "x_min": 200, "y_max": 300, "x_max": 400}'):
            with self.subTest(raw=raw):
                self.assertEqual(bbox_detector._parse(raw), (100.0, 200.0, 300.0, 400.0))

    def test_a_fenced_answer_is_read(self):
        self.assertEqual(
            bbox_detector._parse('```json\n{"box": [1, 2, 3, 4]}\n```'),
            (1.0, 2.0, 3.0, 4.0),
        )

    def test_malformed_json_is_no_box_rather_than_a_crash(self):
        """Observed verbatim from gemini-2.5-flash: a stray nested bracket."""
        self.assertIsNone(bbox_detector._parse('{"box": [199, 13, [969, 572]}'))

    def test_the_documented_not_visible_answer_is_no_box(self):
        self.assertIsNone(bbox_detector._parse("{}"))

    def test_an_inverted_or_out_of_range_box_is_rejected(self):
        """A box SAM cannot use must not reach it: main._normalized_to_pixels
        would happily turn either of these into a negative-width crop."""
        self.assertIsNone(bbox_detector._parse('{"box": [300, 200, 100, 400]}'))
        self.assertIsNone(bbox_detector._parse('{"box": [0, 0, 1200, 400]}'))

    def test_prose_around_the_json_is_no_box(self):
        """A reasoning model that narrates before answering degrades the
        pipeline to dino-only. Pinned so a future model swap that starts
        narrating is caught here rather than in the cutouts."""
        self.assertIsNone(
            bbox_detector._parse("Sure! Here is the box you asked for: [1,2,3,4]")
        )



class StartupProbeTests(unittest.TestCase):
    """The detector is probed at boot, and a failure is loud but not fatal.

    Without the probe a mistyped or retired model id costs nothing visible: the
    gather in _cutout_via_hybrid_sam takes return_exceptions=True, so the
    request still answers 200 with chosen_path="dino-only" and /health stays
    green. The pipeline really is designed to degrade rather than fail -- what
    was missing is any way to notice it had. Same silent shape the voice
    service was broken in for weeks before it got a startup check.

    Not fatal on purpose: /stylize, /describe and /border never touch the
    detector, and taking four endpoints down over one of them is the mistake
    the Vision helper block beside it already avoids.
    """

    def test_lifespan_probes_the_detector(self):
        source = inspect.getsource(main.lifespan)
        self.assertIn("detect_bbox", source,
                      "startup must actually call the detector, not just "
                      "report a flag nobody set")
        self.assertIn("config.BBOX_DETECTOR_MODEL", source,
                      "the probe must use the configured model, or it proves "
                      "nothing about what requests will use")

    def test_a_failed_probe_does_not_stop_the_server(self):
        source = inspect.getsource(main.lifespan)
        probe = source[source.index("global _bbox_detector_error"):]
        self.assertIn("except Exception", probe)
        self.assertNotIn("raise", probe.split("yield")[0],
                         "a detector failure must not prevent startup -- three "
                         "endpoints do not use the detector at all")

    def test_a_failed_probe_is_loud(self):
        self.assertIn("BBOX DETECTOR DOWN", inspect.getsource(main.lifespan),
                      "the whole point is that somebody can see it happened")

    def test_health_reports_the_probe_result(self):
        self.assertIn("bbox_detector_error", inspect.getsource(main.health),
                      "the log line scrolls away; /health is how you check "
                      "later, and how a deploy script could check at all")

    def test_the_probe_image_is_tracked_not_gitignored(self):
        """test_input/ is gitignored, so probing from it would crash a fresh
        clone at boot. demo_data/ travels with the repo."""
        self.assertTrue(main._PROBE_IMAGE.exists(), main._PROBE_IMAGE)
        self.assertIn("demo_data", str(main._PROBE_IMAGE))


if __name__ == "__main__":
    unittest.main()

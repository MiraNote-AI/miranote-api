import unittest

import config
from generate import generate_presets


class GeneratePresetTests(unittest.TestCase):
    def test_art_prompt_keeps_subject_and_adds_the_art_suffix(self):
        prompt = generate_presets.build_art_prompt("a paper crane")
        self.assertTrue(prompt.startswith("a paper crane, "))
        self.assertIn("standalone journal illustration", prompt)
        self.assertNotIn("seamless journaling background texture", prompt,
                         "art must never carry the background rule")
        self.assertNotIn("background removal", prompt,
                         "art must never carry the sticker suffix")

    def test_art_aspect_is_square(self):
        self.assertEqual(config.ASPECT_RATIOS["art"], "1:1")

    def test_existing_commands_untouched(self):
        self.assertIn("seamless journaling background texture",
                      generate_presets.build_background_prompt("dusk"))
        self.assertIn("isolated object", generate_presets.build_sticker_prompt("a cat"))
        self.assertEqual(config.ASPECT_RATIOS["sticker"], "1:1")
        self.assertEqual(config.ASPECT_RATIOS["background"], "9:16")


class BackgroundRuleTests(unittest.TestCase):
    """The background rule must not forbid a border, a margin or rounded corners.

    It used to, and that was the bug. A diffusion text encoder does not negate,
    so "no border, no white margin, no rounded corners" put those three things
    into the conditioning and z-image-turbo drew them: every background came
    back as a card on a white page, and one of them had literal rounded
    corners -- a phrase that appeared in no other suffix.

    Measured over the six prompts in test_input/generate_model_cases.json, one
    variable at a time: with the negations, 0 of 6 were clean; with them
    removed and the page wording replaced, 6 of 6. Evidence, including the
    side-by-side sheet:
    test_output/bg_white_margin/20260928_064330/

    This is pinned in a test because background_rule.txt is raw prompt text
    with nowhere to hold a comment, and "we should forbid borders" is an
    obvious-looking edit that would silently restore the defect.

    Moving them to a real negative_prompt field is not the answer either:
    z-image-turbo accepts that parameter on the mm_sync endpoint and ignores
    it -- a prompt for a green forest still came back green with
    "green, trees, forest" in the negative field.
    """

    def test_the_rule_does_not_forbid_what_it_wants_the_model_to_avoid(self):
        prompt = generate_presets.build_background_prompt("dusk")
        for banned in ("no border", "no white margin", "no rounded corners"):
            with self.subTest(phrase=banned):
                self.assertNotIn(banned, prompt)

    def test_the_innocent_negations_are_kept(self):
        """art_suffix carries these three and has no margin defect, so they are
        not implicated; dropping them risks text coming back."""
        prompt = generate_presets.build_background_prompt("dusk")
        for kept in ("no text", "no watermark", "no shadow"):
            with self.subTest(phrase=kept):
                self.assertIn(kept, prompt)

    def test_the_expander_is_told_the_rule_that_is_actually_appended(self):
        """background_system.txt quotes the rule and tells the model not to
        repeat those phrases. If the two drift apart the expander is warned off
        wording that no longer exists and may emit the new wording itself."""
        from pathlib import Path
        system = Path(generate_presets._PROMPT_DIR / "background_system.txt").read_text()
        rule = generate_presets._BACKGROUND_RULE.lstrip(", ").rstrip()
        self.assertIn(rule, system)

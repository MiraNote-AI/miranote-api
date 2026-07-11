import unittest

import config
from generate import generate_presets


class GeneratePresetTests(unittest.TestCase):
    def test_art_prompt_keeps_subject_and_adds_the_art_suffix(self):
        prompt = generate_presets.build_art_prompt("a paper crane")
        self.assertTrue(prompt.startswith("a paper crane, "))
        self.assertIn("standalone journal illustration", prompt)
        self.assertNotIn("full-page journaling background", prompt,
                         "art must never carry the background rule")
        self.assertNotIn("background removal", prompt,
                         "art must never carry the sticker suffix")

    def test_art_aspect_is_square(self):
        self.assertEqual(config.ASPECT_RATIOS["art"], "1:1")

    def test_existing_commands_untouched(self):
        self.assertIn("clean writing space", generate_presets.build_background_prompt("dusk"))
        self.assertIn("isolated object", generate_presets.build_sticker_prompt("a cat"))
        self.assertEqual(config.ASPECT_RATIOS["sticker"], "1:1")
        self.assertEqual(config.ASPECT_RATIOS["background"], "9:16")

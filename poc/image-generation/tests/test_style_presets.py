"""Offline tests for what /stylize actually sends the model.

This module had no tests while it carried the single most damaging line in the
pipeline -- a wrapper that read every user instruction as an art style, and so
told the model to repaint artwork it had been asked to edit. These guard the
shape of both wrappers, not the wording: an edit to the prose is fine, an edit
that puts the user's instruction back into a "style" slot is not.
"""

import unittest

from stylize import style_presets

# The instruction that exposed the bug, in English because rule 3 keeps
# committed text ASCII. What the app actually sent was the Chinese original --
# the wrapper reframes either one the same way, since it only ever wraps.
SCARF = "give it a red scarf"
REMOVE = "remove the bin in the background"


class FreeTextTests(unittest.TestCase):
    """`prompt` -- the only path the app uses."""

    def test_the_users_words_survive(self):
        self.assertIn(SCARF, style_presets.build_instruction(prompt=SCARF))

    def test_an_instruction_is_not_framed_as_an_art_style(self):
        # The regression this module exists for. "Restyle this photo in the
        # following art style: give it a red scarf" is a request to repaint, and
        # the models obliged -- a flat cartoon sticker came back as felt.
        sent = style_presets.build_instruction(prompt=SCARF)
        self.assertNotIn("art style", sent)
        self.assertNotIn("Restyle", sent)

    def test_nothing_forbids_what_the_user_just_asked_for(self):
        # "Do not add, remove, or rearrange objects" contradicts most of what
        # the app sends: the photo panel's own examples are "remove the bin",
        # "make it autumn", and the sticker panel is used to add accessories.
        for words in (SCARF, REMOVE):
            sent = style_presets.build_instruction(prompt=words)
            self.assertNotIn("Do not add, remove, or rearrange", sent)
            self.assertNotIn("only change the rendering style", sent)

    def test_the_unmentioned_parts_are_still_protected(self):
        # The half of the old wrapper that was worth keeping: without it a
        # restyle is free to invent a new scene.
        sent = style_presets.build_instruction(prompt=SCARF)
        self.assertIn("does not mention", sent)

    def test_borders_are_still_forbidden(self):
        # Not decoration. Dropping the wrapper altogether also fixed the edit
        # cases, but "make it look like a Monet oil painting" then came back as
        # a photograph of a framed painting, gilt frame included.
        self.assertIn("border", style_presets.build_instruction(prompt=SCARF))


class PresetTests(unittest.TestCase):
    """`style` -- unused by the app, still a public parameter."""

    def test_a_preset_is_still_framed_as_an_art_style(self):
        # Correct here: the preset value is a style written as a noun phrase,
        # so it needs a sentence built around it.
        sent = style_presets.build_instruction(style="impressionist")
        self.assertIn("Restyle this photo in the following art style:", sent)
        self.assertIn("impressionist", sent)

    def test_every_preset_builds(self):
        for key in style_presets.STYLE_PRESETS:
            self.assertIn(style_presets.STYLE_PRESETS[key],
                          style_presets.build_instruction(style=key))

    def test_the_preset_wrapper_was_left_alone(self):
        # Splitting the two paths must not quietly reword this one: it is the
        # only wrapper whose output was never in question.
        self.assertEqual(
            style_presets.build_instruction(style="sketch"),
            "Restyle this photo in the following art style: "
            + style_presets.STYLE_PRESETS["sketch"]
            + ". Keep the original composition, subjects, poses, and layout "
              "exactly the same; only change the rendering style. Do not add, "
              "remove, or rearrange objects. Do not add any text, watermark, "
              "signature, or border.")


class RejectionTests(unittest.TestCase):
    def test_an_unknown_preset_names_the_valid_ones(self):
        # /stylize turns this into a 400.
        with self.assertRaises(ValueError) as caught:
            style_presets.build_instruction(style="vaporwave")
        self.assertIn("impressionist", str(caught.exception))

    def test_neither_argument_is_an_error_not_an_empty_prompt(self):
        # An empty instruction would otherwise reach the model as a bare
        # wrapper, and the model would answer with something.
        with self.assertRaises(ValueError):
            style_presets.build_instruction()

    def test_a_preset_key_wins_over_a_prompt(self):
        sent = style_presets.build_instruction(style="cartoon", prompt=SCARF)
        self.assertNotIn(SCARF, sent)


if __name__ == "__main__":
    unittest.main()

"""Offline tests for the backdrop /stylize asks for when it is editing a sticker.

A generated sticker cuts out cleanly and an edited one did not, and the whole
difference was in the prompt: generate/sticker_suffix.txt asks for a contrasting
flat backdrop so the matte has an edge to find, and /stylize never did. Left to
itself the model drew the sticker on white -- the colour of the die-cut edge the
matte then had to locate.

These pin the two halves that make it work: only a sticker gets asked, and the
wording does not contradict the sentence already in the wrapper.
"""

import io
import unittest

from PIL import Image

import main
from stylize import style_presets

SCARF = "give it a red scarf"


def _png(mode: str, colour) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (8, 8), colour).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buf, format="JPEG")
    return buf.getvalue()


class TransparencyTests(unittest.TestCase):
    """How /stylize tells a sticker from a photo, with no help from the caller."""

    def test_a_cutout_is_transparent(self):
        self.assertTrue(main._is_transparent(_png("RGBA", (255, 0, 0, 0))))

    def test_a_jpeg_photo_is_not(self):
        self.assertFalse(main._is_transparent(_jpeg()))

    def test_an_opaque_rgba_png_is_not(self):
        # The trap this replaced a band check for. The app stores every image
        # under a .png name; a photo re-encoded as opaque RGBA would otherwise
        # be read as a sticker and asked for a flat backdrop -- which would
        # replace the scene the user wanted edited.
        self.assertFalse(main._is_transparent(_png("RGBA", (255, 0, 0, 255))))

    def test_a_partly_transparent_image_counts(self):
        img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        img.putpixel((0, 0), (255, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.assertTrue(main._is_transparent(buf.getvalue()))


class BackdropClauseTests(unittest.TestCase):
    def test_a_sticker_is_asked_for_a_contrasting_backdrop(self):
        sent = style_presets.build_instruction(prompt=SCARF,
                                               cut_out_afterwards=True)
        self.assertIn("solid flat single-colour background", sent)
        self.assertIn("contrast strongly", sent)

    def test_a_photo_is_not(self):
        # The damage this prevents: a photo told to sit on a flat colour comes
        # back with its scene replaced.
        sent = style_presets.build_instruction(prompt=SCARF)
        self.assertNotIn("background", sent)

    def test_the_default_is_off(self):
        self.assertEqual(style_presets.build_instruction(prompt=SCARF),
                         style_presets.build_instruction(
                             prompt=SCARF, cut_out_afterwards=False))

    def test_an_existing_die_cut_edge_is_kept(self):
        # Without this sentence the painterly sticker of the five came back as
        # bare artwork with its edge gone.
        sent = style_presets.build_instruction(prompt=SCARF,
                                               cut_out_afterwards=True)
        self.assertIn("keep that edge", sent)

    def test_the_backdrop_request_does_not_contradict_the_wrapper(self):
        # _INSTRUCTION already says "Do not add ... border". A second sentence
        # using that word would argue with the first, which is the defect this
        # module was rewritten to remove -- so the new one says "edge".
        sent = style_presets.build_instruction(prompt=SCARF,
                                               cut_out_afterwards=True)
        self.assertEqual(sent.lower().count("border"), 1)

    def test_the_users_words_still_come_first(self):
        sent = style_presets.build_instruction(prompt=SCARF,
                                               cut_out_afterwards=True)
        self.assertTrue(sent.startswith(SCARF))

    def test_a_preset_is_unaffected(self):
        # The preset path is a restyle, not a cutout, and was never in question.
        self.assertEqual(
            style_presets.build_instruction(style="sketch"),
            style_presets.build_instruction(style="sketch",
                                            cut_out_afterwards=True))


class WiringTests(unittest.TestCase):
    def test_the_endpoint_decides_from_the_pixels(self):
        import inspect
        source = inspect.getsource(main.stylize_image)
        self.assertIn("cut_out_afterwards=_is_transparent(raw)", source)


if __name__ == "__main__":
    unittest.main()

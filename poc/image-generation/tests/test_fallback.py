import unittest
from types import SimpleNamespace

from generate import fallback


def _response(parts):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))]
    )


class FallbackTests(unittest.TestCase):
    def test_model_gate_detected_from_vertex_404(self):
        error = Exception("404 NOT_FOUND. {'error': {'code': 404, 'message': 'Publisher model ...'}}")
        self.assertTrue(fallback.is_model_unavailable(error))
        self.assertFalse(fallback.is_model_unavailable(Exception("500 INTERNAL")))

    def test_image_parts_skips_text_and_keeps_order(self):
        parts = [
            SimpleNamespace(inline_data=None, text="Here you go"),
            SimpleNamespace(inline_data=SimpleNamespace(data=b"png-1")),
            SimpleNamespace(inline_data=SimpleNamespace(data=b"png-2")),
        ]
        self.assertEqual(fallback.image_parts(_response(parts)), [b"png-1", b"png-2"])

    def test_image_parts_tolerates_empty_response(self):
        self.assertEqual(fallback.image_parts(SimpleNamespace(candidates=[])), [])
        self.assertEqual(fallback.image_parts(_response([])), [])

    def test_prompt_carries_aspect_ratio(self):
        prompt = fallback.build_prompt("a paper crane", "1:1")
        self.assertIn("a paper crane", prompt)
        self.assertIn("1:1", prompt)

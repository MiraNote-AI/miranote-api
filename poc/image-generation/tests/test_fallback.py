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


# Verbatim payloads captured from Vertex on 2026-09-05 during the ten-concurrent
# load test. Kept exact so the matchers are pinned against what the API really
# sends rather than a paraphrase of it.
RATE_LIMITED = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource "
    "exhausted. Please try again later. Please refer to "
    "https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 "
    "for more details.', 'status': 'RESOURCE_EXHAUSTED'}}"
)
MODEL_GATED = (
    "404 NOT_FOUND. {'error': {'code': 404, 'message': 'Publisher model "
    "`projects/example-project/locations/us-central1/publishers/google/models/"
    "imagen-4.0-generate-001` not found.', 'status': 'NOT_FOUND'}}"
)


class RateLimitDetectionTests(unittest.TestCase):
    def test_quota_exhaustion_is_detected(self):
        self.assertTrue(fallback.is_rate_limited(Exception(RATE_LIMITED)))

    def test_other_failures_are_not_mistaken_for_quota(self):
        for text in (MODEL_GATED, "500 INTERNAL", "503 UNAVAILABLE", ""):
            with self.subTest(text=text[:20]):
                self.assertFalse(fallback.is_rate_limited(Exception(text)))

    def test_the_two_matchers_do_not_overlap(self):
        """A quota error must not be read as a gated model.

        is_model_unavailable latches a module-level flag that disables the
        primary model for the life of the process. Letting a transient quota
        error through that path would take Imagen down until the next restart.
        """
        self.assertFalse(fallback.is_model_unavailable(Exception(RATE_LIMITED)))
        self.assertFalse(fallback.is_rate_limited(Exception(MODEL_GATED)))

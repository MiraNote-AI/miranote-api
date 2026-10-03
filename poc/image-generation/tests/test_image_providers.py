"""Offline tests for the non-Google image providers.

Everything here is request building and response reading -- the two places where
the DashScope request shapes actually differ and where a copy-paste between them would
produce a plausible-looking wrong call. No network, no keys.
"""

import inspect
import unittest
from unittest import mock

import config
from generate import dashscope_image, dashscope_text, image_providers
from stylize import stylizer
from generate.http_client import ProviderError


class SizeTests(unittest.TestCase):
    def test_size_uses_the_dashscope_separator(self):
        # DashScope wants "1024*1024", not the "1024x1024" every other platform
        # uses. It is the one-character difference a copied adapter gets wrong.
        self.assertEqual(dashscope_image.size_of("qwen-image-3.0", "1:1"), "1024*1024")

    def test_portrait_matches_the_shared_target(self):
        self.assertEqual(dashscope_image.size_of("qwen-image-3.0", "9:16"), "720*1280")

    def test_every_aspect_ratio_the_service_uses_has_a_target(self):
        for ratio in set(config.ASPECT_RATIOS.values()):
            self.assertIn(ratio, config.TARGET_SIZES)


class PriceTests(unittest.TestCase):
    def test_every_dashscope_model_has_a_price(self):
        # A missing price silently books a model as free, which would make it
        # win the cost comparison outright.
        for model in config.IMAGE_MODELS:
            self.assertIn(model, config.IMAGE_PRICE_CNY)
            self.assertGreater(config.IMAGE_PRICE_CNY[model], 0)

    def test_the_rollback_model_is_left_unpriced_on_purpose(self):
        # Not an oversight: gemini-3.1-flash-lite-image has not been looked up,
        # and a guessed figure in this table would read as checked. Delete this
        # test at the same moment a real price goes in.
        self.assertNotIn(config.MODEL_ID, config.IMAGE_PRICE_CNY)

    def test_both_expansion_models_have_a_price(self):
        for model in (config.PROMPT_EXPANDER_MODEL, config.DESCRIBE_MODEL):
            self.assertIn(model, config.TEXT_PRICE_CNY_PER_MTOK)


class ProductionWiringTests(unittest.TestCase):
    """What /generate will actually call, guarded so a config edit cannot
    quietly ship a broken or unpriced combination."""

    def test_every_mode_names_a_known_model(self):
        known = set(config.IMAGE_MODELS) | {config.MODEL_ID}
        for mode, model in config.IMAGE_MODELS_BY_MODE.items():
            self.assertIn(model, known, mode)

    def test_modes_match_aspect_ratios(self):
        # A mode in one dict and not the other either 500s on that command or
        # generates at the wrong shape; config raises at import, this pins it.
        self.assertEqual(set(config.IMAGE_MODELS_BY_MODE),
                         set(config.ASPECT_RATIOS))

    def test_every_configured_model_is_priced(self):
        for mode, model in config.IMAGE_MODELS_BY_MODE.items():
            self.assertIn(model, config.IMAGE_PRICE_CNY, mode)

    def test_describe_does_not_use_the_text_only_expander(self):
        # /describe sends image bytes. qwen-turbo cannot read them, so the two
        # constants must stay separate -- this is the regression that would
        # break poc/chatbot without touching any of its own code.
        self.assertNotEqual(config.DESCRIBE_MODEL, config.PROMPT_EXPANDER_MODEL)
        self.assertFalse(config.DESCRIBE_MODEL.startswith("qwen"))

    def test_prompt_extend_stays_off(self):
        # On z-image-turbo this is the difference between 0.10 and 0.20 CNY an
        # image, and it would also discard the expansion model's output.
        self.assertFalse(config.PROMPT_EXTEND)

    def test_stylize_and_border_do_not_share_a_quota_bucket(self):
        # Vertex meters image generation at 1/min/{project}/{base_model}, so two
        # pipelines on one id share one bucket and starve each other. They were
        # both on gemini-2.5-flash-image until /generate left Vertex and freed
        # this id. Making them equal again would halve the rate either can
        # sustain, and nothing would fail loudly enough to notice.
        self.assertNotEqual(config.STYLE_MODEL, config.BORDER_MODEL)

    def test_stylize_sends_text_and_image_modalities(self):
        # Gemini 3.x image models reject an IMAGE-only request. STYLE_MODEL is a
        # 3.x id, so dropping TEXT here -- it looks redundant, /stylize wants an
        # image -- breaks every restyle on the first call. border.py has carried
        # the same pair since it first called a 3.x model.
        source = inspect.getsource(stylizer.stylize)
        self.assertIn('response_modalities=["TEXT", "IMAGE"]', source)


class DashScopeRequestTests(unittest.TestCase):
    def test_t2i_async_uses_flat_prompt_and_carries_n(self):
        body = dashscope_image.build_body("t2i_async", "wan2.2-t2i-flash",
                                          "a paper crane", "1:1", 2)
        self.assertEqual(body["input"], {"prompt": "a paper crane"})
        self.assertEqual(body["parameters"]["n"], 2)

    def test_mm_sync_uses_messages_and_omits_n(self):
        body = dashscope_image.build_body("mm_sync", "qwen-image-3.0",
                                          "a paper crane", "9:16", 1)
        text = body["input"]["messages"][0]["content"][0]["text"]
        self.assertEqual(text, "a paper crane")
        self.assertNotIn("n", body["parameters"])
        self.assertEqual(body["parameters"]["size"], "720*1280")

    def test_prompt_extend_is_off(self):
        # A rewrite by the platform would replace the expansion model's output,
        # which is the whole thing the benchmark measures.
        for shape in ("mm_sync", "t2i_async", "img_async"):
            body = dashscope_image.build_body(shape, "m", "p", "1:1", 1)
            self.assertFalse(body["parameters"]["prompt_extend"], shape)
            self.assertFalse(body["parameters"]["watermark"], shape)


class DashScopeResponseTests(unittest.TestCase):
    def test_t2i_async_reads_results_list(self):
        data = {"output": {"results": [{"url": "http://a/1.png"},
                                       {"url": "http://a/2.png"}]}}
        self.assertEqual(dashscope_image.urls_of("t2i_async", data),
                         ["http://a/1.png", "http://a/2.png"])

    def test_t2i_async_surfaces_a_per_image_failure(self):
        data = {"output": {"results": [{"code": "DataInspectionFailed",
                                        "message": "blocked"}]}}
        with self.assertRaises(ProviderError):
            dashscope_image.urls_of("t2i_async", data)

    def test_mm_sync_reads_choices(self):
        data = {"output": {"choices": [{"message": {"content": [
            {"image": "http://a/1.png"}]}}]}}
        self.assertEqual(dashscope_image.urls_of("mm_sync", data),
                         ["http://a/1.png"])

    def test_mm_sync_skips_text_parts(self):
        data = {"output": {"choices": [{"message": {"content": [
            {"text": "here you go"}, {"image": "http://a/1.png"}]}}]}}
        self.assertEqual(dashscope_image.urls_of("mm_sync", data),
                         ["http://a/1.png"])

    def test_empty_output_is_empty_not_a_crash(self):
        for shape in ("mm_sync", "t2i_async", "img_async"):
            self.assertEqual(dashscope_image.urls_of(shape, {}), [])


class QwenTextTests(unittest.TestCase):
    def test_reads_message_format(self):
        data = {"output": {"choices": [{"message": {"content": "expanded"}}]}}
        self.assertEqual(dashscope_text.text_of(data), "expanded")

    def test_reads_content_given_as_parts(self):
        data = {"output": {"choices": [{"message": {"content": [
            {"text": "ex"}, {"text": "panded"}]}}]}}
        self.assertEqual(dashscope_text.text_of(data), "expanded")

    def test_no_text_is_none_so_the_caller_can_fall_back(self):
        self.assertIsNone(dashscope_text.text_of({}))
        self.assertIsNone(dashscope_text.text_of({"output": {"choices": []}}))


class RoutingTests(unittest.TestCase):
    def test_unknown_model_names_the_valid_ones(self):
        with self.assertRaises(ProviderError) as caught:
            image_providers.generate("gpt-image-1", "p", "1:1", 1)
        self.assertIn("qwen-image-3.0", str(caught.exception))

    def test_every_configured_model_has_a_known_shape(self):
        for model, spec in config.IMAGE_MODELS.items():
            self.assertEqual(spec["provider"], "dashscope", model)
            self.assertIn(spec["shape"], dashscope_image._PATHS, model)

    def test_the_rollback_id_routes_to_gemini_not_an_error(self):
        # IMAGE_MODEL=<MODEL_ID> is the documented way off DashScope, so the
        # router must recognise it even though it is absent from IMAGE_MODELS.
        with mock.patch("generate.gemini_image.generate",
                        return_value=[b"png"]) as gemini:
            self.assertEqual(
                image_providers.generate(config.MODEL_ID, "p", "1:1", 1), [b"png"])
        gemini.assert_called_once()


if __name__ == "__main__":
    unittest.main()

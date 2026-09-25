"""One entry point for every /generate image model, dispatching on the id.

Adding a model is a config.IMAGE_MODELS entry; adding a platform is one more
branch here plus its adapter module.

Gemini is routed here too rather than being special-cased in main.py, which is
what lets config.IMAGE_MODELS_BY_MODE name it: setting a mode (or the
IMAGE_MODEL env var) to config.MODEL_ID is the rollback to the pre-DashScope
path, with no code change.
"""

import config
from generate import dashscope_image, gemini_image
from generate.http_client import ProviderError


def generate(model: str, prompt: str, aspect_ratio: str,
             n: int = 1) -> list[bytes]:
    spec = config.IMAGE_MODELS.get(model)
    if spec is None:
        # The Gemini id is deliberately not in IMAGE_MODELS: that table is the
        # set of DashScope models, and MODEL_ID is the way back off DashScope.
        if model == config.MODEL_ID:
            return gemini_image.generate(model, prompt, aspect_ratio, n)
        raise ProviderError(f"unknown image model {model!r}; valid: "
                            f"{sorted(set(config.IMAGE_MODELS) | {config.MODEL_ID})}")
    return dashscope_image.generate(model, prompt, aspect_ratio, n)

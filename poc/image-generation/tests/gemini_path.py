"""Point /generate at the Gemini model for the duration of a test.

Several suites drive /generate end to end while standing in for the Vertex SDK
(`main._get_client`). Since /generate moved to DashScope those mocks describe a
provider the request no longer reaches, so the suite has to say which path it is
testing.

Using it is not a workaround: `config.MODEL_ID` is the documented rollback off
DashScope (`IMAGE_MODEL=<that id>`), so every test wrapped in this is coverage
for the rollback still behaving the way it did before the move -- quota still
becoming a 503, a blank still being retried once, an expansion still running
again for that retry.

The DashScope side of the same behaviour is covered separately, in
tests/test_dashscope_failures.py, which mocks the provider layer instead.
"""

from unittest import mock

import config


def serving_gemini():
    """Context manager: all three /generate modes use config.MODEL_ID."""
    return mock.patch.dict(
        config.IMAGE_MODELS_BY_MODE,
        {mode: config.MODEL_ID for mode in config.IMAGE_MODELS_BY_MODE},
    )

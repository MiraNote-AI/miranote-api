"""Configuration for the photo-collections service.

Everything that is a number or a switch lives here; everything that is
Chinese text lives in prompts/*.txt (org Rule 3 keeps source ASCII-only).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PROMPT_DIR = HERE / "prompts"

# Loaded by absolute path so the service finds its key whatever the working
# directory is. Existing environment variables win.
load_dotenv(HERE / ".env")

# ---------- Model ----------
# The three move together: the OpenAI SDK falls back to api.openai.com when
# base_url is None, so the default URL is explicit (as in voice-to-text).
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY") or None
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or DASHSCOPE_BASE_URL
LLM_MODEL = os.getenv("LLM_MODEL") or "qwen3.7-flash"

# How each call talks to the model, settled by tools/probe_model.py
# (2026-10-31, see README.md "Model call modes"):
#   "thinking_stream" -- enable_thinking=true, thinking_budget, stream=true,
#                        no response_format; JSON parsed by the program
#   "json_object"     -- enable_thinking=false, response_format json_object
#   "json_schema"     -- enable_thinking=false, response_format json_schema
#                        (strict) with SCHEMA_B
CALL_A_MODE = os.getenv("PHOTO_CALL_A_MODE") or "thinking_stream"
CALL_B_MODE = os.getenv("PHOTO_CALL_B_MODE") or "json_schema"
CALL_C_MODE = os.getenv("PHOTO_CALL_C_MODE") or "json_object"
MODES = ("thinking_stream", "json_object", "json_schema")
THINKING_BUDGET = 1500
TEMPERATURE_A = 0.2
TEMPERATURE_B = 0.7
TEMPERATURE_C = 0.2

# The copy rules the memory was written under; the phone keys its AI cache
# on it, so a change of rules never serves old copy (copy refine handoff 5.4).
COPY_VERSION = "copy_v3"  # copy_v3: question refine 2026-11-05

# ---------- Prices (CNY per million tokens; thinking billed as output) ----------
PRICE_INPUT_PER_M = 0.2
PRICE_OUTPUT_PER_M = 0.8


def cost_cny(prompt_tokens: int, completion_tokens: int) -> float:
    """What one call cost, from the token counts the provider returned."""
    return (
        prompt_tokens / 1_000_000 * PRICE_INPUT_PER_M
        + completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_M
    )


# ---------- Time limits (seconds) ----------
CALL_A_TIMEOUT = 180.0  # user decision 2026-11-01 (spec said 90, then 120)
# Calls B and C see the photos (copy refine handoff 2.2, 2.4). Measured on
# the 80 test photos, 2026-11-04, 5 runs x 8 groups: B 5-26 s with two
# outliers at 81-84 s (16 jobs in flight), C 5-26 s. Limits are about 1.5x
# the usual top; an outlier is retried inside the 300 s job cap. C raised
# from 40 s on 2026-11-05: on a real phone's 7 groups at once it ran past
# 40 s twice in a row for one group.
CALL_B_TIMEOUT = 60.0
CALL_C_TIMEOUT = 60.0
# Calls B and C see the photos at most this many pixels on the long side
# (user decision 2026-11-05: a group of 10 landscape photos ran both past
# 60 s at full size). Call A keeps what the phone sent.
SMALL_IMAGE_PX = 512
JOB_CAP_SECONDS = 300.0  # user decision 2026-11-01 (spec said 240)
# Results are kept an hour: the phone may poll a batch of seven groups late.
JOB_TTL_SECONDS = float(os.getenv("PHOTO_JOB_TTL_SECONDS") or 3600)

# ---------- Concurrency ----------
# WORKERS jobs run at once; each job makes up to two model calls at a time
# (calls A, B and C within a job run one after another, but a timed-out request stays in
# the pool while its retry starts; see _calls in llm.py), so the model pool is
# sized at twice the job pool.
WORKERS = int(os.getenv("PHOTO_WORKERS") or 40)
LLM_WORKERS = int(os.getenv("PHOTO_LLM_WORKERS") or 80)

# ---------- Request limits ----------
MAX_PHOTOS = 10
MAX_PHOTO_BYTES = 5 * 1024 * 1024
# The push buckets the phone may send, in tie order (names live in names.txt).
BUCKET_IDS = ("food", "scenery", "animal", "city", "indoor", "event", "other")
MAX_TZ_CHARS = 64

# ---------- Log ----------
LOG_PATH = REPO_ROOT / "logs" / "photo-collections" / "compose.jsonl"

# ---------- Call B output schema (copy refine handoff 2.2) ----------
# No minItems / maxItems / maxLength: DashScope answers 400 for some of them.
# Counts and lengths are checked by the program instead.
SCHEMA_B = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cannot_write", "candidates", "question"],
    "properties": {
        "cannot_write": {"type": "boolean"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "photo_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "photo_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "question": {
            "type": "object",
            "additionalProperties": False,
            "required": ["ranked"],
            "properties": {
                "ranked": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "type", "why"],
                        "properties": {
                            "text": {"type": "string"},
                            "type": {"type": "string"},
                            "why": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

# ---------- Call C output schema (copy refine handoff 2.4) ----------
# Used only when CALL_C_MODE is "json_schema".
_VERDICT = {"type": "boolean"}
SCHEMA_C = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sentences", "questions"],
    "properties": {
        "sentences": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "grounded", "mentions_cover", "reason"],
            "properties": {"id": {"type": "string"}, "grounded": _VERDICT,
                           "mentions_cover": _VERDICT, "reason": {"type": "string"}},
        }},
        "questions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "grounded", "safe", "reason"],
            "properties": {"id": {"type": "string"}, "grounded": _VERDICT,
                           "safe": _VERDICT, "reason": {"type": "string"}},
        }},
    },
}

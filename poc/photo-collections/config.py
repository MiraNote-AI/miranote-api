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
MODES = ("thinking_stream", "json_object", "json_schema")
THINKING_BUDGET = 1500
TEMPERATURE_A = 0.2
TEMPERATURE_B = 0.7

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
CALL_B_TIMEOUT = 30.0
JOB_CAP_SECONDS = 300.0  # user decision 2026-11-01 (spec said 240)
JOB_TTL_SECONDS = 900.0
WORKERS = 4

# ---------- Request limits ----------
MAX_PHOTOS = 10
MAX_PHOTO_BYTES = 5 * 1024 * 1024

# ---------- Log ----------
LOG_PATH = REPO_ROOT / "logs" / "photo-collections" / "compose.jsonl"

# ---------- Call B output schema (AI doc 3.3) ----------
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
                "required": ["text", "photo_ids", "uses"],
                "properties": {
                    "text": {"type": "string"},
                    "photo_ids": {"type": "array", "items": {"type": "string"}},
                    "uses": {"type": "array", "items": {"type": "string"}},
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
                        "required": ["id", "fill", "why"],
                        "properties": {
                            "id": {"type": "string"},
                            "fill": {"type": "string"},
                            "why": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

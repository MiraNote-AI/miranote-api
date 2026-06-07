import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["PROJECT_ID"],
            location=os.environ["LOCATION"],
        )
    return _client


_PROMPT_DIR = Path(__file__).parent
SYSTEM_PROMPT = (_PROMPT_DIR / "sticker_system.txt").read_text(encoding="utf-8")


def expand(user_input: str, model: str) -> str:
    prompt = SYSTEM_PROMPT.replace("{{USER_INPUT}}", user_input)
    response = _get_client().models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text.strip()


BACKGROUND_SYSTEM_PROMPT = (_PROMPT_DIR / "background_system.txt").read_text(encoding="utf-8")


def expand_background(user_input: str, model: str) -> str:
    prompt = BACKGROUND_SYSTEM_PROMPT.replace("{user_prompt}", user_input)
    response = _get_client().models.generate_content(model=model, contents=prompt)
    return response.text.strip()


"""One-off real connectivity probe for the photo-collections model calls.

Settles which request shape works for call A (look at photos) and call B
(write text) against LLM_MODEL, before any code depends on it. Uses a
synthetic image drawn in memory -- never a personal photo. Costs well under
0.05 CNY. Not part of tests/ (it reaches the network on purpose).

Usage (from poc/photo-collections):
    .venv/bin/python tools/probe_model.py
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from openai import OpenAI  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def _test_image() -> str:
    img = Image.new("RGB", (768, 576), (40, 90, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse((250, 150, 520, 420), fill=(220, 40, 40))
    draw.rectangle((40, 460, 300, 540), fill=(250, 220, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _messages(image_url: str) -> list:
    return [
        {"role": "system", "content": "You describe images. Output JSON only."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "P1"},
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": 'Describe P1 as {"shapes": [{"shape": str, "color": str}], '
                    '"background": str}. JSON only.',
                },
            ],
        },
    ]


def _report(label: str, ok: bool, latency: float, usage, text: str, err: str = "") -> float:
    cost = 0.0
    if usage:
        cost = config.cost_cny(usage.prompt_tokens, usage.completion_tokens)
    print("--- {}: {}".format(label, "OK" if ok else "FAILED"))
    print("    latency {:.1f}s".format(latency))
    if usage:
        print("    tokens prompt={} completion={} cost={:.5f} CNY".format(
            usage.prompt_tokens, usage.completion_tokens, cost))
    if text:
        print("    text: " + text[:300].replace("\n", " "))
    if err:
        print("    error: " + err[:400])
    return cost


def probe_thinking_stream(client: OpenAI, image_url: str) -> float:
    t0 = time.time()
    text, reasoning_len, usage = "", 0, None
    try:
        stream = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=_messages(image_url),
            temperature=config.TEMPERATURE_A,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": True, "thinking_budget": config.THINKING_BUDGET},
            timeout=config.CALL_A_TIMEOUT,
        )
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in chunk.choices or []:
                delta = choice.delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_len += len(reasoning)
                if delta.content:
                    text += delta.content
        json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception as error:  # noqa: BLE001 -- a probe reports, never raises
        return _report("A1 thinking+stream", False, time.time() - t0, usage, text, str(error))
    print("    reasoning chars: {}".format(reasoning_len))
    return _report("A1 thinking+stream", True, time.time() - t0, usage, text)


def probe_json(client: OpenAI, image_url: str, label: str, response_format: dict,
               temperature: float, with_image: bool = True) -> float:
    t0 = time.time()
    messages = _messages(image_url)
    if not with_image:
        messages = [
            {"role": "system", "content": "You write short JSON answers."},
            {"role": "user", "content": "Write one candidate sentence about a red circle, "
             "and rank question ids B1 and B2. Fill must be empty."},
        ]
    try:
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            extra_body={"enable_thinking": False},
            timeout=config.CALL_A_TIMEOUT,
        )
        text = resp.choices[0].message.content or ""
        json.loads(text)
    except Exception as error:  # noqa: BLE001
        return _report(label, False, time.time() - t0, None, "", str(error))
    return _report(label, True, time.time() - t0, resp.usage, text)


def main() -> int:
    if not config.LLM_API_KEY:
        print("LLM_API_KEY is not set in poc/photo-collections/.env")
        return 2
    print("model={} base_url={}".format(config.LLM_MODEL, config.LLM_BASE_URL))
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    image_url = _test_image()
    total = 0.0
    total += probe_thinking_stream(client, image_url)
    total += probe_json(client, image_url, "A2 no-thinking+json_object",
                        {"type": "json_object"}, config.TEMPERATURE_A)
    schema = {"type": "json_schema", "json_schema": {
        "name": "collect_b", "strict": True, "schema": config.SCHEMA_B}}
    total += probe_json(client, image_url, "B1 no-thinking+json_schema strict",
                        schema, config.TEMPERATURE_B, with_image=False)
    total += probe_json(client, image_url, "B2 no-thinking+json_object",
                        {"type": "json_object"}, config.TEMPERATURE_B, with_image=False)
    print("total probe cost: {:.5f} CNY".format(total))
    return 0


if __name__ == "__main__":
    sys.exit(main())

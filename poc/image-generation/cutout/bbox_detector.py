import json
import re
from typing import Optional

from google.genai import types

from shared.vertex_client import _get_client


_SYSTEM_PROMPT = """You are an object detector for image cutouts.

Given the image and a target description, return the bounding box of the requested object.

Rules:
- Return ONLY a JSON object: {"box": [y_min, x_min, y_max, x_max]} with values normalized to 0-1000.
- Box the WHOLE thing the target names, including whatever is served in it, piled
  on it, or held by it. When the target names a container -- "the parfait glass",
  "the bowl of noodle soup" -- the box must also cover its contents, including
  any part that rises above the rim or hangs over the edge.
- If multiple instances of the target exist, return only the LARGEST one (by area).
- If the target object is NOT visible in the image, return exactly: {}
- No commentary, no markdown fences, no extra text.

Target: {{TARGET}}"""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse(raw: str) -> Optional[tuple[float, float, float, float]]:
    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    box = data.get("box") or data.get("box_2d")
    if isinstance(box, list) and len(box) == 4:
        try:
            y_min, x_min, y_max, x_max = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
    elif {"x_min", "y_min", "x_max", "y_max"}.issubset(data):
        try:
            x_min = float(data["x_min"])
            y_min = float(data["y_min"])
            x_max = float(data["x_max"])
            y_max = float(data["y_max"])
        except (TypeError, ValueError):
            return None
    else:
        return None

    if not (0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000):
        return None
    return y_min, x_min, y_max, x_max


def detect_bbox(image_bytes: bytes, target: str, model: str) -> Optional[tuple[float, float, float, float]]:
    prompt = _SYSTEM_PROMPT.replace("{{TARGET}}", target)
    mime = "image/png" if image_bytes[:8].startswith(b"\x89PNG") else "image/jpeg"
    response = _get_client().models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
        # Asks for a stable box; does not guarantee one. Measured 2026-09-26:
        # with these exact settings and identical bytes, four calls to
        # gemini-2.5-flash returned three different boxes, and varied the JSON
        # schema between {"box":..}, {"box_2d":..} and {"x_min":..} across them.
        # gemini-3.1-flash-lite (the current default) returned one box in the
        # requested schema on all four. So this config narrows the sampling, and
        # how much it actually buys is a property of the model, not of the flag
        # -- do not build an A/B on the assumption that one call is the model's
        # answer. Evidence: test_output/bbox_model_ab/20260926_134013/
        config=types.GenerateContentConfig(temperature=0, seed=0),
    )
    raw = response.text or ""
    print(f"[bbox_detector] target={target!r} raw={raw!r}")
    return _parse(raw)

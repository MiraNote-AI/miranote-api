"""GroundingDINO open-vocabulary detection: image + text → bbox."""
import io
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import config


_processor = None
_model = None


def preload() -> None:
    global _processor, _model
    device = config.GROUNDING_DEVICE
    if device == "mps" and not torch.backends.mps.is_available():
        print("[dino] MPS unavailable; falling back to CPU")
        device = "cpu"
    print(f"[dino] loading {config.GROUNDING_DINO_MODEL} on {device}...")
    _processor = AutoProcessor.from_pretrained(config.GROUNDING_DINO_MODEL)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.GROUNDING_DINO_MODEL
    ).to(device)
    _model.eval()
    print(f"[dino] ready on {device}")


def detect_all_boxes(image_bytes: bytes, target: str, threshold: float) -> list[tuple[tuple[float, float, float, float], float]]:
    """Returns list of (bbox_normalized_0_1000, score), sorted by score desc."""
    if _processor is None or _model is None:
        raise RuntimeError("grounding_dino.preload() must be called first")

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = pil_img.size

    text = target.strip()
    if not text.endswith("."):
        text = text + "."

    device = next(_model.parameters()).device
    inputs = _processor(images=pil_img, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _model(**inputs)

    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=config.GROUNDING_TEXT_THRESHOLD,
        target_sizes=[(h, w)],
    )

    if not results or len(results[0]["boxes"]) == 0:
        print(f"[dino] target={target!r} no boxes at threshold {threshold}")
        return []

    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    order = torch.argsort(scores, descending=True).tolist()

    out: list[tuple[tuple[float, float, float, float], float]] = []
    for i in order:
        x1, y1, x2, y2 = boxes[i].tolist()
        bbox_norm = (
            (y1 / h) * 1000,
            (x1 / w) * 1000,
            (y2 / h) * 1000,
            (x2 / w) * 1000,
        )
        out.append((bbox_norm, float(scores[i].item())))
    print(f"[dino] target={target!r} found {len(out)} candidates: scores={[f'{s:.2f}' for _b, s in out]}")
    return out

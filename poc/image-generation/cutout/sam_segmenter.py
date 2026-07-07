"""SAM 2.1 Large via official PyTorch package: image + bbox → full-image binary mask PNG."""
import io
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image, ImageFilter
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import config


_predictor: Optional[SAM2ImagePredictor] = None

# SAM2ImagePredictor is stateful: set_image() stores the image embedding on the
# predictor and predict() reads it back. Concurrent /cutout requests share this
# one predictor (each segment call runs in an asyncio.to_thread worker), so a
# second request's set_image() can land between the first's set_image() and
# predict() and return a mask for the wrong image. This lock keeps each
# set_image + predict pair atomic across threads.
_predict_lock = threading.Lock()


def _cache_dir() -> Path:
    return Path(os.path.expanduser(config.SAM_CACHE_DIR))


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[sam] downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with open(tmp, "wb") as f:
            for buf in resp.iter_content(chunk_size=1 << 20):
                if not buf:
                    continue
                f.write(buf)
                read += len(buf)
                if total:
                    print(f"[sam]   {read / 1e6:.1f}/{total / 1e6:.1f} MB", end="\r")
    tmp.rename(dest)
    print(f"\n[sam] saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def preload() -> None:
    global _predictor
    cache = _cache_dir()
    ckpt = cache / config.SAM2_CHECKPOINT_NAME
    _download(config.SAM2_CHECKPOINT_URL, ckpt)

    device = config.SAM2_DEVICE
    if device == "mps" and not torch.backends.mps.is_available():
        print("[sam] MPS unavailable; falling back to CPU")
        device = "cpu"

    print(f"[sam] building SAM 2 model on {device}...")
    model = build_sam2(config.SAM2_MODEL_CFG, str(ckpt), device=device)
    _predictor = SAM2ImagePredictor(model)
    print(f"[sam] SAM 2.1 Large ready on {device}")


def segment_with_bbox(image_bytes: bytes, bbox_pixels: tuple[float, float, float, float]) -> bytes:
    if _predictor is None:
        raise RuntimeError("sam_segmenter.preload() must be called first")

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.asarray(pil_img)
    orig_w, orig_h = pil_img.size

    x1, y1, x2, y2 = bbox_pixels  # (x1, y1, x2, y2)
    pad = config.SAM_BOX_PADDING_RATIO
    if pad > 0:
        dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    x1 = max(0.0, x1); y1 = max(0.0, y1)
    x2 = min(float(orig_w), x2); y2 = min(float(orig_h), y2)
    box = np.array([x1, y1, x2, y2], dtype=np.float32)

    # Hold the lock only for the stateful set_image + predict pair; the numpy
    # post-processing below works on returned arrays and is thread-local.
    with _predict_lock, torch.inference_mode():
        _predictor.set_image(np_img)
        logits, scores, _ = _predictor.predict(
            box=box, multimask_output=True, return_logits=True
        )

    best = int(np.argmax(scores))
    logit = logits[best]
    soft = 1.0 / (1.0 + np.exp(-logit.astype(np.float32) * config.SAM_MASK_GAIN))
    pct = float((soft > 0.5).mean())
    print(
        f"[sam] picked={best} score={float(scores[best]):.3f} "
        f"logit_min={logit.min():.2f} logit_max={logit.max():.2f} pct_foreground={pct:.3f}"
    )

    if config.SAM_BINARIZE_MASK:
        # Hard threshold removes the half-transparent "fuzzy" zones where SAM is unsure;
        # a 1px blur keeps the edge anti-aliased instead of jagged.
        binary = ((soft > 0.5) * 255).astype(np.uint8)
        alpha_img = Image.fromarray(binary, mode="L").filter(ImageFilter.GaussianBlur(1))
    else:
        alpha = (soft * 255).clip(0, 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha, mode="L")
    if alpha_img.size != (orig_w, orig_h):
        alpha_img = alpha_img.resize((orig_w, orig_h), Image.BILINEAR)

    buf = io.BytesIO()
    alpha_img.save(buf, format="PNG")
    return buf.getvalue()

"""
MiraNote POC -- Image Generation API
Sticker generation with Apple Vision background removal.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import asyncio
import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, ImageFilter
from pydantic import BaseModel
from rembg import remove, new_session

import beta_auth

import config
from shared.vertex_client import _get_client
from generate import (fallback, prompt_expander, generate_presets,
                      image_providers, dashscope_text)
from generate.http_client import ProviderError
from cutout import bbox_detector, sam_segmenter, grounding_dino, vision_matte
from stylize import stylizer, style_presets
from border import border, border_presets


# Concurrent sticker generations are capped so ten testers cannot queue enough
# CPU work to push a single request past Cloudflare's 125s edge timeout. That is
# the only thing this bounds. It is NOT a quota control and cannot be one: the
# Vertex image limit is a rate (2/min/{project}/{base_model}) while a semaphore
# bounds simultaneity, so any value here still permits far more than 2 calls in
# a minute. What keeps /generate inside the quota is NUMBER_OF_IMAGES = 1, which
# makes one request cost one call instead of two at once (api #69).
GENERATE_CONCURRENCY = 3
_generate_semaphore = None


def _generate_gate():
    """The /generate concurrency cap.

    Built on first use rather than at import time: on Python 3.9 an
    asyncio.Semaphore binds to whatever loop exists when it is constructed,
    which at import time is not the loop the app ends up serving on.
    """
    global _generate_semaphore
    if _generate_semaphore is None:
        _generate_semaphore = asyncio.Semaphore(GENERATE_CONCURRENCY)
    return _generate_semaphore


# Deliberately not the provider's own wording: "RESOURCE_EXHAUSTED" tells a
# tester nothing. Deliberately not a retry either -- config.py records that
# users who retry wedge the queue, and section 7 of the deploy spec rules out
# automatic retries on a saturated host.
QUOTA_DETAIL = "image generation is busy right now; wait a moment and try again"
# The two ways a response can carry no image. Kept apart because the advice
# differs: a refusal will not succeed on a retry of the same prompt, an empty
# answer usually will. Neither repeats the provider's own words --
# "PROHIBITED_CONTENT" tells a tester nothing. The reason goes to the log,
# where it is useful.
REFUSED_DETAIL = "the image model would not draw that one; try describing it differently"
EMPTY_DETAIL = "no picture came back that time; try again"

# One extra attempt when the model answers with nothing and did not refuse.
# Not a general retry policy: quota rejections are never retried (section 7 of
# the deploy spec, and a 429 means the budget is already gone), and refusals are
# not retried because the second answer is the first one again.
EMPTY_RESPONSE_RETRIES = 1


def _quota_exhausted(model: str, error: Exception) -> HTTPException:
    print(f"[generate] quota exhausted on {model}: {str(error)[:120]}")
    return HTTPException(status_code=503, detail=QUOTA_DETAIL)


# How a DashScope failure maps onto the three outcomes above. Gemini reports all
# of this in the response object (fallback.is_safety_refusal and friends);
# DashScope has no equivalent, so the only signal is the error text.
#
# Both lists hold ONLY codes that have actually been observed. Guessing more in
# would mean a wrong status for a failure nobody has seen, which is worse than
# the generic path -- so anything unmatched propagates as a 502 carrying the
# provider's own words.
_DASHSCOPE_QUOTA_MARKERS = (
    "Arrearage",      # account out of credit. Seen 2026-09-11: all three modes
                      # answered 400 Arrearage at once, mid-benchmark.
    "429",            # survived http_client's four backoff attempts
)
_DASHSCOPE_REFUSAL_MARKERS = (
    "DataInspectionFailed",   # content filter, on the request or the output
)


def _classify_provider_error(error: Exception) -> str:
    """"quota" | "refused" | "unknown" for a ProviderError."""
    text = str(error)
    if any(marker in text for marker in _DASHSCOPE_QUOTA_MARKERS):
        return "quota"
    if any(marker in text for marker in _DASHSCOPE_REFUSAL_MARKERS):
        return "refused"
    return "unknown"


def _one_image(model: str, prompt: str, aspect_ratio: str) -> tuple[bytes | None, bool]:
    """One image from one call: (image, refused).

    Split out of _call_model so the retry loop around it is identical on both
    providers. The Gemini branch is the pre-DashScope code verbatim, which is
    what makes IMAGE_MODEL=<config.MODEL_ID> a real rollback rather than an
    untested one.

    Raises HTTPException(503) when the provider says the budget is gone, and
    lets anything it cannot classify propagate to _call_model.
    """
    if model == config.MODEL_ID:
        response = _get_client().models.generate_content(
            model=model,
            contents=fallback.build_prompt(prompt, aspect_ratio),
        )
        parts = fallback.image_parts(response)
        if parts:
            return parts[0], False
        print(f"[generate] empty response from {model}: "
              f"{fallback.empty_reason(response)}")
        return None, fallback.is_safety_refusal(response)

    try:
        images = image_providers.generate(model, prompt, aspect_ratio, 1)
    except ProviderError as error:
        outcome = _classify_provider_error(error)
        if outcome == "quota":
            raise _quota_exhausted(model, error)
        if outcome == "refused":
            print(f"[generate] {model} refused the prompt: {str(error)[:200]}")
            return None, True
        raise
    if images:
        return images[0], False
    # A 200 that carried no image. Distinct from every branch above: it is the
    # one failure worth another attempt, so it comes back as "empty".
    print(f"[generate] {model} answered with no image")
    return None, False


def _call_model(prompt: str, aspect_ratio: str, model: str,
                reprompt=None) -> list[bytes]:
    """Generate NUMBER_OF_IMAGES images with `model`.

    `reprompt`, when given, produces a fresh prompt for a retry. /generate
    passes the expander so the second attempt does not send the string that
    just drew a blank; it passes None when the caller asked for no expansion,
    because then there is nothing to vary.
    """

    def _next_prompt(previous: str) -> str:
        """A fresh prompt for the retry, or the previous one if that fails.

        The expander is a network call. Losing it must not also lose the
        retry, which is the expensive half -- an image call out of a bucket
        holding two a minute.
        """
        if reprompt is None:
            return previous
        try:
            return reprompt()
        except Exception as error:
            print(f"[generate] re-expansion failed, retrying with the same "
                  f"prompt: {str(error)[:120]}")
            return previous

    def _one() -> tuple[bytes | None, bool]:
        """(image, refused). Refused is carried alongside because an empty
        answer is only a failure once every call has come back empty.

        A blank gets one more attempt; a refusal gets none. An image call is
        expensive enough that the second one is only worth spending where it can
        succeed: an empty answer can produce a picture on a second try, while a
        refusal buys another refusal (#78). Bounded at one extra attempt --
        NUMBER_OF_IMAGES = 1 left a single blank with nothing to hide behind,
        and the point is to cover that, not to grind against the provider.
        """
        current = prompt
        for attempt in range(1 + EMPTY_RESPONSE_RETRIES):
            if attempt:
                current = _next_prompt(current)
            image, refused = _one_image(model, current, aspect_ratio)
            if image:
                return image, False
            if refused:
                return None, True
        return None, False

    # The images are independent; generate them concurrently so the whole
    # request stays comfortably inside client timeouts.
    try:
        with ThreadPoolExecutor(max_workers=config.NUMBER_OF_IMAGES) as pool:
            results = list(pool.map(lambda _: _one(), range(config.NUMBER_OF_IMAGES)))
    except HTTPException:
        # _one_image already chose the status (503 for a spent budget). Re-raising
        # it unchanged matters: the branches below would relabel it 502, and the
        # app keys its message off the status code alone.
        raise
    except ProviderError as error:
        # Anything _classify_provider_error could not place. The provider's own
        # words go in the detail because nobody has triaged this shape yet.
        raise HTTPException(status_code=502, detail=f"{model}: {error}")
    except Exception as error:
        if fallback.is_rate_limited(error):
            raise _quota_exhausted(model, error)
        raise
    images = [image for image, _ in results if image]
    if not images:
        if any(refused for _, refused in results):
            raise HTTPException(status_code=502, detail=REFUSED_DETAIL)
        raise HTTPException(status_code=502, detail=EMPTY_DETAIL)
    return images


# The startup probe's image. demo_data/ is tracked, unlike test_input/, so the
# probe works on a fresh clone. Any photo with a subject does; the probe checks
# that the model answers at all, not what it answers.
_PROBE_IMAGE = Path(__file__).parent / "demo_data" / "17.jpeg"

_rembg_session = None

# Whether the Vision helper came up. Nothing branches on it -- the defaults are
# Vision either way, see config.ROLLING BACK TO REMBG -- but /health reports it,
# which is the only cheap way to tell "Vision is serving every cutout" from
# "every cutout is 503ing" from outside the process.
_vision_ready = False

# Whether the bbox detector answered at startup: None until probed, then the
# error text, or "" for ok. Nothing branches on it either, for a different and
# worse reason: a detector that stops answering does not fail the request, it
# quietly removes the disambiguation step. _cutout_via_hybrid_sam gathers the
# two detectors with return_exceptions=True, so a dead Gemini becomes
# chosen_path="dino-only" and a 200, with /health still green.
#
# That is the shape the voice service was broken in for weeks (#73). One cheap
# call at boot is what turns it into something a person can see.
_bbox_detector_error = None

# Same for DashScope: None until probed, the error text, or "" for ok. A third
# value matters here -- "unused" -- because the documented rollback
# (IMAGE_MODEL=<config.MODEL_ID>) puts /generate back on Vertex, and warning
# about a credential that path never reads would be noise.
_dashscope_error = None


def _dashscope_is_configured() -> bool:
    """Whether anything this process will actually call lives on DashScope.

    Two independent users: the image models, and the prompt expander. Either
    one alone makes the credential required, and the rollback to Gemini only
    moves the first -- PROMPT_EXPANDER_MODEL stays on qwen unless it is changed
    too, which is why this asks about both rather than about IMAGE_MODEL.
    """
    if any(m in config.IMAGE_MODELS for m in config.IMAGE_MODELS_BY_MODE.values()):
        return True
    return config.PROMPT_EXPANDER_MODEL.lower().startswith("qwen")


def _rembg():
    """The rembg session, built on first use.

    Lazy because the defaults are Vision-only now: on a normal run nothing
    reaches rembg, and eagerly loading a background-removal model that never
    runs costs startup time and resident memory for nothing. It stays one call
    away for ?mode=auto, matte="rembg", and a rollback of the config defaults.
    """
    global _rembg_session
    if _rembg_session is None:
        print(f"Loading rembg session ({config.REMBG_MODEL})...")
        _rembg_session = new_session(config.REMBG_MODEL)
        print("rembg session ready.")
    return _rembg_session


class _NotFound(Exception):
    pass


@contextmanager
def _stage(timings: dict, name: str):
    """Record one /cutout stage's wall-clock ms into `timings`.

    Only the prompt-guided branch is instrumented; the timings ride back on
    the response so a caller can see which stage dominates without reading
    server logs. It is the cheapest way to tell a slow detector from a slow
    matte on a host nobody can attach a profiler to.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - start) * 1000, 1)


def _timed(timings: dict, name: str, fn, *args):
    """asyncio.to_thread target that times `fn` inside the worker thread.

    Timing here rather than around the await measures the call itself, not
    the time it spent waiting for a thread-pool slot. Concurrent workers
    write distinct keys, so the shared dict needs no lock.
    """
    with _stage(timings, name):
        return fn(*args)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ay1, ax1, ay2, ax2 = a
    by1, bx1, by2, bx2 = b
    iy1, ix1 = max(ay1, by1), max(ax1, bx1)
    iy2, ix2 = min(ay2, by2), min(ax2, bx2)
    if iy2 <= iy1 or ix2 <= ix1:
        return 0.0
    inter = (iy2 - iy1) * (ix2 - ix1)
    area_a = (ay2 - ay1) * (ax2 - ax1)
    area_b = (by2 - by1) * (bx2 - bx1)
    return inter / (area_a + area_b - inter)


def _union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Smallest box covering both. Boxes in 0-1000 (y_min, x_min, y_max, x_max)."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    """Largest box inside both, or None if they do not overlap."""
    y_min, x_min = max(a[0], b[0]), max(a[1], b[1])
    y_max, x_max = min(a[2], b[2]), min(a[3], b[3])
    if y_max <= y_min or x_max <= x_min:
        return None
    return (y_min, x_min, y_max, x_max)


def _is_transparent(image_bytes: bytes) -> bool:
    """Whether an image actually has see-through pixels.

    Not "does it have an alpha band": a fully opaque RGBA PNG has one and is
    still a photo. The app stores every image under a .png name -- including
    photos, whose bytes are JPEG from PhotoTreatments.downscaled() -- so the
    filename says nothing and the band alone would misread the day that
    encoding changes. What /stylize needs to know is whether anything is
    see-through, so that is what this asks.
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == "P":
        image = image.convert("RGBA")
    if "A" not in image.getbands():
        return False
    return image.getchannel("A").getextrema()[0] < 255


def _alpha_bbox(rgba_png_bytes: bytes):
    """Tight box around a matte's opaque pixels, normalised to 0-1000.

    In the prebg modes everything outside this box is background by
    construction -- flatten paints it a flat colour -- so it is a hard upper
    bound on where the subject can be, and clipping a detector box to it can
    only tighten, never loosen.
    """
    alpha = Image.open(io.BytesIO(rgba_png_bytes)).convert("RGBA").getchannel("A")
    solid = alpha.point(lambda v: 255 if v > 127 else 0)
    box = solid.getbbox()
    if box is None:
        return None
    left, upper, right, lower = box
    width, height = alpha.size
    return (upper / height * 1000, left / width * 1000,
            lower / height * 1000, right / width * 1000)


def _normalized_to_pixels(image_bytes: bytes, bbox_norm: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    y_min, x_min, y_max, x_max = bbox_norm
    return ((x_min / 1000) * w, (y_min / 1000) * h, (x_max / 1000) * w, (y_max / 1000) * h)


def _flatten_on_bg(rgba_png_bytes: bytes, color: tuple[int, int, int]) -> bytes:
    """Composite an RGBA cutout onto a solid background color, return RGB PNG.

    Downstream detectors (DINO/SAM) convert("RGB"), which turns rembg's transparent
    pixels black; flatten onto white/gray instead so the background is clean but neutral.
    """
    img = Image.open(io.BytesIO(rgba_png_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, color + (255,))
    flat = Image.alpha_composite(bg, img).convert("RGB")
    buf = io.BytesIO()
    flat.save(buf, format="PNG")
    return buf.getvalue()


async def _cutout_via_prebg(image_bytes: bytes, prompt: str, timings: dict,
                            matte, stage: str, label: str, clip_to_matte: bool = False):
    """Full-image background removal -> flatten onto solid gray -> hybrid_sam_union.

    Removing the background first stops the union step from spanning competing
    subjects in cluttered/multi-subject scenes -- where plain hybrid_sam_union
    tends to grab the whole frame. Downstream detectors convert("RGB"), so we
    flatten onto a neutral solid color instead of leaving the transparency
    (which would otherwise become black).

    `matte` is the only difference between the two prebg modes, so an A/B
    between them measures the background remover and nothing else. The one that
    ran is reported back as extras["prebg"].
    """
    with _stage(timings, stage):
        removed = await asyncio.to_thread(matte, image_bytes)
    with _stage(timings, "flatten"):
        flattened = _flatten_on_bg(removed, (128, 128, 128))
    clip = _alpha_bbox(removed) if clip_to_matte else None
    out, bbox, extras = await _cutout_via_hybrid_sam(flattened, prompt, timings, clip)
    extras = extras or {}
    extras["prebg"] = label
    if clip is not None:
        extras["matte_bbox"] = [round(v, 1) for v in clip]
    return out, bbox, extras


def _rembg_matte(image_bytes: bytes) -> bytes:
    return remove(image_bytes, session=_rembg())


async def _cutout_via_hybrid_prebg(image_bytes: bytes, prompt: str, timings: dict):
    return await _cutout_via_prebg(image_bytes, prompt, timings,
                                   _rembg_matte, "rembg", "gray")


async def _cutout_via_hybrid_prebg_vision(image_bytes: bytes, prompt: str, timings: dict):
    """Same as the rembg prebg mode with Apple Vision producing the matte.

    Vision costs 0.03 s against rembg's 12.55 s on the 17-image bench, and was
    better on three of them and worse on none. It raises rather than falling
    back, so a run of this mode is always a run of Vision.
    """
    return await _cutout_via_prebg(image_bytes, prompt, timings,
                                   vision_matte.remove_background, "vision_matte", "vision",
                                   clip_to_matte=True)


def _apply_fullsize_mask(image_bytes: bytes, mask_png: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    mask = Image.open(io.BytesIO(mask_png)).convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
    img.putalpha(mask)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _cutout_via_hybrid_sam(image_bytes: bytes, prompt: str, timings: dict,
                                 clip_bbox=None):
    """Detector boxes -> SAM. `clip_bbox` bounds where the subject can be.

    A box covering most of the frame tells SAM almost nothing, and on a prebg
    image -- subject over one flat colour -- SAM then answers "the flat field"
    with high confidence: on 06_parfait all three candidates came back as the
    background, none overlapping the subject at all. Clipping the box to the
    matte's extent fixed it (candidates went to 0.998 overlap). The flat colour
    itself is not the trigger; grey, black, white and green all failed
    identically, while the untouched photo with the same loose box was fine.
    """
    dino_task = asyncio.to_thread(
        _timed, timings, "dino",
        grounding_dino.detect_all_boxes, image_bytes, prompt, config.HYBRID_DINO_THRESHOLD
    )
    gemini_task = asyncio.to_thread(
        _timed, timings, "gemini",
        bbox_detector.detect_bbox, image_bytes, prompt, config.BBOX_DETECTOR_MODEL
    )
    # The two detectors run concurrently, so dino + gemini exceeds detect_wall;
    # detect_wall is the wall-clock cost this step actually adds.
    #
    # return_exceptions=True so one detector failing degrades to the other
    # instead of killing the request: the dino-only / gemini-only branches
    # below already handle a missing box, and a transient Vertex 429/500 used
    # to 500 a cutout that GroundingDINO alone could have served.
    with _stage(timings, "detect_wall"):
        dino_candidates, gemini_bbox = await asyncio.gather(
            dino_task, gemini_task, return_exceptions=True
        )

    dino_error = gemini_error = None
    if isinstance(dino_candidates, BaseException):
        dino_error = dino_candidates
        dino_candidates = []
        print(f"[hybrid] dino failed, falling back: {dino_error!r}")
    if isinstance(gemini_bbox, BaseException):
        gemini_error = gemini_bbox
        gemini_bbox = None
        print(f"[hybrid] gemini failed, falling back: {gemini_error!r}")

    if not dino_candidates and gemini_bbox is None:
        # Both detectors down is an outage, not a miss -- surface the real
        # error rather than a misleading "not found".
        if dino_error is not None or gemini_error is not None:
            raise dino_error or gemini_error
        raise _NotFound(f"hybrid_sam: '{prompt}' not found")

    if gemini_bbox is None:
        chosen_bbox = dino_candidates[0][0]
        used_path = "dino-only"
    elif not dino_candidates:
        chosen_bbox = gemini_bbox
        used_path = "gemini-only"
    else:
        rated = [(box, score, _iou(gemini_bbox, box)) for box, score in dino_candidates]
        passing = [(b, s, m) for b, s, m in rated if m >= config.HYBRID_IOU_THRESHOLD]
        if passing:
            box, score, m = max(passing, key=lambda t: t[2])   # highest IoU wins
            chosen_bbox = _union(box, gemini_bbox)
            used_path = f"dino(iou={m:.2f}, score={score:.2f})+union"
        else:
            chosen_bbox = gemini_bbox
            best = max((m for _b, _s, m in rated), default=0.0)
            used_path = f"gemini(best_iou={best:.2f})"

    clipped_to = None
    if clip_bbox is not None:
        tightened = _intersect(chosen_bbox, clip_bbox)
        # No overlap means the detectors and the matte disagree completely;
        # trust the detectors rather than hand SAM an empty box.
        if tightened is not None and tightened != chosen_bbox:
            clipped_to = chosen_bbox
            chosen_bbox = tightened

    print(f"[hybrid] {used_path} bbox={chosen_bbox}"
          + (f" (clipped from {clipped_to})" if clipped_to else ""))

    bbox_pixels = _normalized_to_pixels(image_bytes, chosen_bbox)
    with _stage(timings, "sam"):
        mask_png = await asyncio.to_thread(
            sam_segmenter.segment_with_bbox, image_bytes, bbox_pixels
        )
    with _stage(timings, "apply_mask"):
        out = _apply_fullsize_mask(image_bytes, mask_png)
    extras = {
        "dino_bboxes": [list(b) for b, _s in dino_candidates],
        "dino_scores": [round(s, 3) for _b, s in dino_candidates],
        "gemini_bbox": list(gemini_bbox) if gemini_bbox else None,
        "chosen_path": used_path,
    }
    if clipped_to is not None:
        extras["bbox_before_matte_clip"] = list(clipped_to)
    if dino_error is not None or gemini_error is not None:
        extras["detector_errors"] = {
            name: f"{type(err).__name__}: {str(err)[:200]}"
            for name, err in (("dino", dino_error), ("gemini", gemini_error))
            if err is not None
        }
    return out, chosen_bbox, extras


PROMPT_CUTOUT_MODES = {
    "hybrid_sam_union": _cutout_via_hybrid_sam,             # DINO candidates disambiguated by Gemini box (IoU + union) -> SAM
    "hybrid_sam_prebg_gray": _cutout_via_hybrid_prebg,     # rembg -> gray bg -> hybrid_sam_union
    "hybrid_sam_prebg_vision": _cutout_via_hybrid_prebg_vision,  # DEFAULT: as above, Apple Vision matte instead of rembg (macOS only)
}

# Whole-foreground modes, for a /cutout with no prompt. Unlike the prompted
# modes these are not a dispatch table -- each is two lines inline in the
# endpoint -- but naming the valid set here keeps a typo'd `mode` a 400 rather
# than a silent rembg run, which is how it would read once the default is Vision.
AUTO_CUTOUT_MODES = {"vision", "auto"}

# Background removers for a generated sticker (/generate). Not a /cutout mode
# set: this path has no prompt, no detector and no SAM, it is just the matte.
#
# "none" returns the sticker with its background still on. It is how a client
# that runs Apple Vision itself -- the iOS app does, on the user's own device --
# asks for the generated image and nothing else. Leaving the matte to the phone
# costs the server a Vision call per image, and gives the app a failure it can
# handle locally instead of a 503 that discards an image already paid for.
STICKER_MATTE_MODES = {"vision", "rembg", "none"}

# Modes that cannot run without the Vision helper. Named or defaulted to, they
# raise when the helper is missing and answer 503; nothing downgrades them.
#
# There is deliberately no startup step-down to rembg. A machine that could not
# build the helper would then serve rembg at 12-80 s per image while every log
# line and response still looked normal -- the slowdown was the only symptom,
# and nothing named it. Rolling back is a deliberate edit of the three config
# defaults instead; see ROLLING BACK TO REMBG in config.py.
VISION_PROMPT_CUTOUT_MODES = {"hybrid_sam_prebg_vision"}
VISION_AUTO_CUTOUT_MODES = {"vision"}
VISION_STICKER_MATTE_MODES = {"vision"}


def _parse_hex(color: str) -> tuple[int, int, int, int]:
    """'#RRGGBB' or '#RRGGBBAA' -> (r, g, b, a). Raises HTTPException(400) on bad input."""
    s = color.lstrip("#")
    if len(s) not in (6, 8):
        raise HTTPException(status_code=400, detail=f"invalid hex color '{color}'")
    try:
        rgb = tuple(int(s[i:i + 2], 16) for i in range(0, 6, 2))
        a = int(s[6:8], 16) if len(s) == 8 else 255
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid hex color '{color}'")
    return rgb + (a,)


def _safe_debug_dir(debug_dir: str) -> str | None:
    """Confine /border debug frames under a fixed test_output/ root.

    debug_dir is a client-supplied request field and ai_outline_border writes PNG
    frames into it, so an unchecked value lets a caller create directories and
    write files anywhere on the server. Reject absolute paths and parent-directory
    traversal; nest any accepted relative path under test_output/.
    """
    if not debug_dir:
        return None
    norm = os.path.normpath(debug_dir)
    if os.path.isabs(norm) or norm == ".." or norm.startswith(".." + os.sep):
        raise HTTPException(
            status_code=400,
            detail="debug_dir must be a relative path under test_output/",
        )
    root = "test_output"
    if norm != root and not norm.startswith(root + os.sep):
        norm = os.path.join(root, norm)
    return norm


def _erode_alpha(png_bytes: bytes, radius: int) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    r, g, b, a = img.split()
    a = a.filter(ImageFilter.MinFilter(radius * 2 + 1))
    img = Image.merge("RGBA", (r, g, b, a))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to Vertex AI (genai client)...")
    _get_client()
    print("Vertex AI client ready.")
    # rembg is no longer preloaded: nothing reaches it on the default paths, so
    # see _rembg() for why it is built on first use instead.
    print("Loading SAM ONNX sessions...")
    sam_segmenter.preload()
    print("SAM ready.")
    print("Loading GroundingDINO...")
    grounding_dino.preload()
    print("GroundingDINO ready.")
    # Every cutout needs it. Starting it here keeps its ~1.4 s model load off the
    # first request; failing to start it must still not stop the server booting,
    # because /stylize, /describe and /border have nothing to do with the matte
    # and taking them down too would turn one broken feature into four.
    global _vision_ready
    _vision_ready = False
    error = None
    if vision_matte.available():
        try:
            vision_matte.preload()
            _vision_ready = True
            print("Vision matte helper ready.")
        except Exception as e:                      # noqa: BLE001 - reported below
            error = e
    else:
        error = "not built for this machine"

    if _vision_ready:
        print(f"[cutout] prompted={config.DEFAULT_PROMPT_CUTOUT_MODE} "
              f"auto={config.DEFAULT_AUTO_CUTOUT_MODE} "
              f"sticker_matte={config.DEFAULT_STICKER_MATTE}")
    else:
        # Loud, because the service is Vision-only: there is no step-down to
        # rembg, so this is not a slow mode, it is no cutout at all.
        print(f"[cutout] !! VISION HELPER DOWN ({error}) -- every /cutout and "
              f"every sticker /generate will answer 503 until it is back. "
              f"Build it with: swiftc -O -parse-as-library "
              f"vision_bench/vision_bench.swift -o vision_bench/vision_bench "
              f"(macOS 15+). To serve rembg instead, see ROLLING BACK TO REMBG "
              f"in config.py.")
    # The prompted /cutout disambiguates GroundingDINO's candidates with this
    # model. It is deliberately not fatal: the pipeline really does degrade to
    # dino-only rather than failing, and /stylize, /describe and /border do not
    # use the detector at all. What is NOT acceptable is that happening
    # silently, which is all this probe fixes.
    global _bbox_detector_error
    try:
        await asyncio.to_thread(
            bbox_detector.detect_bbox,
            Path(_PROBE_IMAGE).read_bytes(), "a subject",
            config.BBOX_DETECTOR_MODEL,
        )
        _bbox_detector_error = ""
        print(f"[cutout] bbox detector ready ({config.BBOX_DETECTOR_MODEL}).")
    except Exception as e:                          # noqa: BLE001 - reported below
        _bbox_detector_error = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"[cutout] !! BBOX DETECTOR DOWN ({config.BBOX_DETECTOR_MODEL}): "
              f"{_bbox_detector_error} -- every prompted /cutout will still "
              f"answer 200, but with GroundingDINO alone and no disambiguation. "
              f"Check the model id and that Vertex is reachable.")

    # /generate runs on DashScope, and http_client.api_key() only looks at the
    # environment when a request is already in flight. Without this a host that
    # forgot the key boots clean, answers /health 200 and reports its image
    # models, then 502s every /generate -- and the app says "AI server is not
    # running", which is false. Exactly the shape the voice service was broken
    # in for weeks (#73) before #77 gave it a startup check.
    #
    # A real round trip, not a presence check, for the reason _check_llm states
    # in voice-to-text: what matters is whether this key works for this base
    # URL and this model. It also catches a revoked key and an account in
    # arrears, which a presence check cannot and which has really happened.
    global _dashscope_error
    if not _dashscope_is_configured():
        _dashscope_error = "unused"
        print("[generate] DashScope not in use (Gemini rollback); "
              "DASHSCOPE_API_KEY not required.")
    else:
        try:
            await asyncio.to_thread(dashscope_text.complete, "ping",
                                    config.PROMPT_EXPANDER_MODEL)
            _dashscope_error = ""
            print(f"[generate] DashScope ready "
                  f"({config.PROMPT_EXPANDER_MODEL}, "
                  f"{config.DASHSCOPE_BASE_URL}).")
        except Exception as e:                      # noqa: BLE001 - reported below
            _dashscope_error = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[generate] !! DASHSCOPE NOT WORKING: {_dashscope_error} -- "
                  f"every /generate will answer 502 and the app will say the "
                  f"AI server is not running, which will not be true. Check "
                  f"DASHSCOPE_API_KEY in this host's .env (see .env.example).")

    try:
        yield
    finally:
        # Otherwise the helper outlives the server -- see vision_matte.shutdown.
        vision_matte.shutdown()


app = FastAPI(title="MiraNote Image Generation", version="0.1.0", lifespan=lifespan)

# Reachable from the public internet through the Cloudflare tunnel, so every
# request needs a beta token. Installed before CORSMiddleware on purpose: the
# most recently added middleware is the outermost, and CORS must stay outside
# the gate to answer a browser preflight, which carries no Authorization
# header.
beta_auth.install(app)


def _remove_sticker_bg(raw: bytes, matte: str) -> bytes:
    """Cut a generated sticker out. Raises VisionMatteUnavailable on failure.

    There is deliberately no fallback to rembg when Vision fails. The image is
    already generated and paid for, so answering 503 throws that away over a
    background rembg could still have removed -- but a matte the caller cannot
    identify is worse than a failure they can retry, and a quietly-rembg'd
    sticker was indistinguishable from a Vision one in the page it landed on.
    The discarded generation is the accepted cost.

    rembg still runs when the caller asks for it by name, and is one config
    edit from being the default again (ROLLING BACK TO REMBG in config.py).
    """
    if matte == "vision":
        cut = vision_matte.remove_background(raw)
    else:
        cut = remove(raw, session=_rembg())
    # Eroding here rather than in the endpoint keeps all of the pixel work on
    # the worker thread; it is off by default, but it is a full-image filter.
    if config.REMBG_ERODE_RADIUS > 0:
        cut = _erode_alpha(cut, config.REMBG_ERODE_RADIUS)
    return cut


class GenerateRequest(BaseModel):
    command: str        # "sticker" | "background"
    prompt: str = ""    # user-written prompt for sticker
    expand: bool = True   # if True, expand prompt via LLM before generation
    matte: str = ""     # sticker background remover: "vision" (default) | "rembg" | "none"


@app.post("/generate")
async def generate_images(req: GenerateRequest):
    async with _generate_gate():
        return await _generate(req)


# command -> (prompt_expander attribute, generate_presets attribute). Held as
# names and resolved when called, not captured at import, so the retry path
# below can build a prompt as many times as it needs -- and so patching either
# module reaches the handler.
_COMMANDS = {
    "sticker":    ("expand",            "build_sticker_prompt"),
    "background": ("expand_background", "build_background_prompt"),
    "art":        ("expand_art",        "build_art_prompt"),
}


async def _generate(req: GenerateRequest):
    spec = _COMMANDS.get(req.command)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown command: {req.command}")
    if not req.prompt:
        raise HTTPException(
            status_code=400, detail=f"prompt is required for {req.command}"
        )
    expander_name, builder_name = spec

    matte = req.matte or config.DEFAULT_STICKER_MATTE
    mattable = config.REMOVE_BG and req.command == "sticker"
    if mattable and matte not in STICKER_MATTE_MODES:
        # Checked before generating: a typo should not cost an image call out
        # of a bucket holding two a minute.
        raise HTTPException(
            status_code=400,
            detail=f"unknown matte '{matte}'; valid: {sorted(STICKER_MATTE_MODES)}",
        )
    # Validated as a matte above, then honoured by not running one.
    remove_bg = mattable and matte != "none"

    def _fresh_prompt() -> str:
        """Expand and dress the user's words. Called once normally, and once
        more if the first attempt draws a blank -- expansion is not
        deterministic, so the retry gets a different string to send. It runs on
        a text model, metered separately from the 2/min image quota, so varying
        the prompt costs nothing that was scarce (#80)."""
        core = (
            getattr(prompt_expander, expander_name)(
                req.prompt, config.PROMPT_EXPANDER_MODEL
            )
            if req.expand
            else req.prompt
        )
        return getattr(generate_presets, builder_name)(core)

    prompt = await asyncio.to_thread(_fresh_prompt)
    ratio = config.ASPECT_RATIOS.get(req.command, "1:1")
    model = config.IMAGE_MODELS_BY_MODE[req.command]
    # Nothing to vary when the caller asked for no expansion.
    reprompt = _fresh_prompt if req.expand else None
    images = await asyncio.to_thread(_call_model, prompt, ratio, model, reprompt)

    encoded = []
    for raw in images:
        if remove_bg:
            try:
                # to_thread, not a bare call: background removal is pixel work
                # off the event loop either way, and rembg (still reachable by
                # name) is seconds of CPU per image.
                processed = await asyncio.to_thread(_remove_sticker_bg, raw, matte)
            except vision_matte.VisionMatteUnavailable as e:
                raise HTTPException(status_code=503, detail=str(e))
        else:
            processed = raw
        encoded.append(base64.b64encode(processed).decode())

    response = {"command": req.command, "prompt": prompt, "raw_input": req.prompt,
                "images": encoded, "count": len(encoded)}
    if mattable:
        # Reported for "none" too, so a caller can tell "you asked me not to"
        # from "this build does not matte stickers".
        response["matte_used"] = matte
    return response


def _shrink_for_model(raw: bytes, max_side: int = 1536) -> bytes:
    """Defensively cap input resolution. Clients should downscale, but one
    oversized upload (a 4320px phone photo made it through once) turns a
    cutout into minutes of compute and wedges every request queued behind
    it. Preserves alpha; no-op for images already within the cap."""
    img = Image.open(io.BytesIO(raw))
    if max(img.size) <= max_side:
        return raw
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/cutout")
async def cutout_image(
    file: UploadFile,
    prompt: str = "",
    mode: str = "",
):
    timings: dict = {}
    request_start = time.perf_counter()
    with _stage(timings, "read"):
        uploaded = await file.read()
    with _stage(timings, "shrink"):
        raw = _shrink_for_model(uploaded)

    if prompt:
        chosen = mode or config.DEFAULT_PROMPT_CUTOUT_MODE
        if chosen not in PROMPT_CUTOUT_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown mode '{chosen}'; valid: {list(PROMPT_CUTOUT_MODES)}",
            )
        try:
            processed, bbox, extras = await PROMPT_CUTOUT_MODES[chosen](raw, prompt, timings)
            used = chosen
        except _NotFound as e:
            raise HTTPException(status_code=404, detail=str(e))
        except vision_matte.VisionMatteUnavailable as e:
            # A Vision mode never silently becomes a rembg one, so say what broke
            # instead of letting a bare 500 look like a bug in the cutout itself.
            raise HTTPException(status_code=503, detail=str(e))
        with _stage(timings, "encode"):
            encoded = base64.b64encode(processed).decode()
        timings["total_server"] = round(time.perf_counter() - request_start, 3)
        response = {
            "image": encoded,
            "mode_used": used,
            "prompt": prompt,
            "bbox": bbox,
            # Per-stage ms. Kept out of `extras` because read/shrink/encode
            # happen here, outside the mode function that builds `extras`.
            "timings": timings,
        }
        if extras:
            response.update(extras)
        return response

    # No prompt: whole-foreground removal, Apple Vision by default and rembg
    # under mode="auto". Same discipline as the prompted branch above -- every
    # mode, defaulted to or named, is served or refused, never substituted.
    chosen = mode or config.DEFAULT_AUTO_CUTOUT_MODE
    if chosen not in AUTO_CUTOUT_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown mode '{chosen}'; valid without a prompt: "
                   f"{sorted(AUTO_CUTOUT_MODES)}",
        )
    if chosen == "vision":
        try:
            processed = await asyncio.to_thread(vision_matte.remove_background, raw)
        except vision_matte.VisionMatteUnavailable as e:
            raise HTTPException(status_code=503, detail=str(e))
    else:
        processed = await asyncio.to_thread(_rembg_matte, raw)
    if config.REMBG_ERODE_RADIUS > 0:
        processed = _erode_alpha(processed, config.REMBG_ERODE_RADIUS)
    return {
        "image": base64.b64encode(processed).decode(),
        "mode_used": chosen,
    }


@app.post("/stylize")
async def stylize_image(
    file: UploadFile,
    style: str = "",          # preset key, e.g. "impressionist"
    prompt: str = "",         # custom style description (used when no/unknown preset)
    temperature: float | None = None,  # 0 = faithful to original; higher = more creative
):
    raw = _shrink_for_model(await file.read())
    try:
        # Transparency is how a sticker announces itself. The app never tells us
        # which it sent, and it does not have to: a photo is opaque, a sticker is
        # a cutout. Asking a photo for a flat backdrop would replace the scene
        # the user wanted edited, so this must not fire on one.
        instruction = style_presets.build_instruction(
            style=style, prompt=prompt,
            cut_out_afterwards=_is_transparent(raw))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    temp = config.STYLE_TEMPERATURE if temperature is None else temperature
    processed = await asyncio.to_thread(
        stylizer.stylize, raw, instruction, config.STYLE_MODEL, temp
    )
    return {
        "image": base64.b64encode(processed).decode(),
        "style_used": style or "custom",
        "instruction": instruction,
        "temperature": temp,
    }


@app.post("/describe")
async def describe_image(file: UploadFile, prompt: str = None):
    """Vision over one image.

    Default: one warm sentence about the photo, for the app's page
    context -- so the journaling chat can "see" what sits on the page.
    Canvas mode passes its own `prompt` to ask what a whole page looks
    like instead."""
    raw = await file.read()
    question = config.describe_question(prompt)

    def _describe() -> str:
        from google.genai import types
        response = _get_client().models.generate_content(
            # Not PROMPT_EXPANDER_MODEL: this call sends image bytes, and that
            # constant now names a text-only model. See config.DESCRIBE_MODEL.
            model=config.DESCRIBE_MODEL,
            contents=[
                types.Part.from_bytes(data=raw, mime_type=file.content_type or "image/png"),
                question,
            ],
        )
        return (response.text or "").strip()

    description = await asyncio.to_thread(_describe)
    if not description:
        raise HTTPException(status_code=502, detail="the model returned no description")
    return {"description": description}


@app.post("/border")
async def border_image(
    file: UploadFile,
    mode: str = "outline",        # "outline" | "ai_outline"
    # outline mode
    color: str = config.BORDER_COLOR,
    width: int = config.BORDER_WIDTH,
    # ai_outline mode
    style: str = "",
    prompt: str = "",
    band_ratio: float = config.BORDER_BAND_RATIO,
    paste_back: bool = True,  # False: keep the model's own subject (skip cutout paste-back)
    white_edge: bool | None = None,  # None: ai_outline default (off)
    white_edge_width: int = config.WHITE_EDGE_WIDTH,
    shadow: bool = True,
    temperature: float | None = None,
    debug_dir: str = "",
):
    raw = await file.read()

    if mode == "outline":
        out = await asyncio.to_thread(border.outline_cutout, raw, _parse_hex(color), width)
    elif mode == "ai_outline":
        try:
            instruction = border_presets.build_outline_instruction(style=style, prompt=prompt)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        temp = config.BORDER_TEMPERATURE if temperature is None else temperature
        we = False if white_edge is None else white_edge  # ai_outline: white edge off by default
        out = await asyncio.to_thread(
            border.ai_outline_border, raw, instruction, config.BORDER_MODEL,
            band_ratio, (255, 255, 255, 255), border._DEFAULT_BG,
            paste_back, we, white_edge_width, shadow, temp,
            config.BORDER_WORK_SIZE, _safe_debug_dir(debug_dir), config.BORDER_GUIDE_MIN_RATIO,
        )
    else:
        raise HTTPException(status_code=400, detail=f"unknown mode '{mode}'; valid: outline, ai_outline")

    return {"image": base64.b64encode(out).decode(), "mode_used": mode}


@app.get("/health")
async def health():
    return {"status": "ok",
            # Per mode, not one `model` key. That key used to name the single
            # /generate model; now MODEL_ID is only the rollback target, so
            # reporting it would name a model the service never calls -- worse
            # than reporting none. `image_models` is also how you confirm an
            # IMAGE_MODEL override actually took effect.
            "image_models": config.IMAGE_MODELS_BY_MODE,
            "prompt_expander": config.PROMPT_EXPANDER_MODEL,
            # "" once DashScope answered at boot, "unused" on the Gemini
            # rollback, the error text otherwise. Worth a field for the same
            # reason as cutout.bbox_detector_error: the failure it names does
            # not stop the server or show up anywhere else until a user hits
            # /generate.
            "dashscope_error": _dashscope_error,
            "describe": config.DESCRIBE_MODEL,
            # Reported for the same reason as image_models: a STYLE_MODEL
            # override is otherwise invisible from outside the process.
            "stylize": config.STYLE_MODEL,
            "border": config.BORDER_MODEL,
            # Whether the Vision helper came up. Worth a health field precisely
            # because nothing steps down: a false here means every cutout is
            # answering 503, and the defaults beside it say what the server
            # would be serving if it were true. start_backends.sh reads this.
            "cutout": {"vision_ready": _vision_ready,
                       "prompt_mode": config.DEFAULT_PROMPT_CUTOUT_MODE,
                       "auto_mode": config.DEFAULT_AUTO_CUTOUT_MODE,
                       "sticker_matte": config.DEFAULT_STICKER_MATTE,
                       # Reported for the same reason as image_models and
                       # stylize: a BBOX_DETECTOR_MODEL override is otherwise
                       # invisible from outside the process, and an A/B between
                       # detectors is worthless if you cannot prove which one
                       # the arm actually ran.
                       "bbox_detector": config.BBOX_DETECTOR_MODEL,
                       # "" once the detector answered at boot, the error text
                       # if it did not, null if the probe never ran. Worth a
                       # field because a dead detector does not fail requests --
                       # it silently drops the disambiguation step.
                       "bbox_detector_error": _bbox_detector_error}}

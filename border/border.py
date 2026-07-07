"""
Sticker borders for the /border endpoint.

Two modes:
  * outline (outline_cutout): a solid-color die-cut stroke that hugs the object's
    silhouette. Pure Pillow, no AI.
  * ai_outline (ai_outline_border): a decorative band that hugs the silhouette --
    the band shape is enforced deterministically in Pillow and only its *pattern*
    comes from Gemini, so the result is stable regardless of what the model paints.
    Finishes with an optional white die-cut edge and drop shadow.

Its Gemini image wrapper (`_gemini_generate_image`) stays local to this module --
it adds retries that the style-transfer path doesn't need -- but the low-level
pieces it shares with stylizer (the Vertex client singleton and the
response-extraction helper) live in vertex_client.
"""

import io
import os

from google.genai import types
from PIL import Image, ImageChops, ImageFilter

from shared.vertex_client import _get_client, _extract_image_bytes

# Solid background color for the canvas handed to Gemini (ai_outline). Magenta is
# rare in real subjects/borders, so it contrasts cleanly with the decoration.
_DEFAULT_BG = (255, 0, 255)
# Largest single MaxFilter step (kernel = 2*step+1) used when dilating.
_MAX_DILATE_STEP = 5


def _make_dump(debug_dir):
    """Return a dump(name, img) that writes debug frames only when debug_dir is set."""
    def _dump(name, img):
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            img.save(os.path.join(debug_dir, name))
    return _dump


# --------------------------------------------------------------------------- #
# Gemini image call
# --------------------------------------------------------------------------- #

def _gemini_generate_image(image_bytes: bytes | None, instruction: str, model: str,
                           seed: int = 0, temperature: float = 0, retries: int = 3) -> bytes:
    """Generate a PNG from a Gemini image model.

    ``image_bytes`` is optional: pass an image for edit/img2img (ai_outline passes
    the subject-plus-band canvas), or ``None`` for pure text-to-image.

    Retries with a bumped seed when the model returns no image -- common patterns
    intermittently hit IMAGE_RECITATION (a copyright/recitation filter), which a
    different seed usually clears. Successful calls never retry.
    """
    contents = []
    if image_bytes is not None:
        mime = "image/png" if image_bytes[:8].startswith(b"\x89PNG") else "image/jpeg"
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
    contents.append(instruction)
    last_err = None
    for attempt in range(retries):
        response = _get_client().models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                # Gemini 3.x image models expect TEXT+IMAGE; _extract_image_bytes
                # ignores any text part and returns the inline image.
                response_modalities=["TEXT", "IMAGE"],
                temperature=temperature,
                seed=seed + attempt,
            ),
        )
        try:
            raw = _extract_image_bytes(response)
        except RuntimeError as e:
            last_err = e
            continue
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    raise last_err


# --------------------------------------------------------------------------- #
# Pillow helpers
# --------------------------------------------------------------------------- #

def dilate_alpha(alpha: Image.Image, px: int) -> Image.Image:
    """Grow an L-mode mask outward by ~px pixels via iterated MaxFilter.

    Pillow caps kernel size, so large widths are applied in small odd-kernel
    steps rather than one big filter.
    """
    if px <= 0:
        return alpha
    out = alpha
    remaining = px
    while remaining > 0:
        step = min(remaining, _MAX_DILATE_STEP)
        out = out.filter(ImageFilter.MaxFilter(step * 2 + 1))
        remaining -= step
    return out


def erode_alpha(alpha: Image.Image, px: int) -> Image.Image:
    """Shrink an L-mode mask inward by ~px pixels via iterated MinFilter.

    Mirror of ``dilate_alpha``: Pillow caps kernel size, so the shrink is applied
    in small odd-kernel steps. Used to shrink the subject hole so the decorated
    band ring tucks a few px under the pasted cutout and leaves no seam.
    """
    if px <= 0:
        return alpha
    out = alpha
    remaining = px
    while remaining > 0:
        step = min(remaining, _MAX_DILATE_STEP)
        out = out.filter(ImageFilter.MinFilter(step * 2 + 1))
        remaining -= step
    return out


def smooth_mask(mask: Image.Image, blur: float, lo: int = 112, hi: int = 144) -> Image.Image:
    """Round off the stair-stepping left by box-kernel dilation/erosion.

    Blur the L-mode mask, then re-map the alpha through a steep ramp centered at
    128: this collapses the soft feathered halo a plain blur leaves behind into a
    crisp edge while keeping ~a couple px of anti-aliasing so it isn't razor-hard.
    """
    if blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return mask.point(
        lambda v: 0 if v <= lo else 255 if v >= hi else round((v - lo) * 255 / (hi - lo))
    )


def _to_rgba(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGBA")


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _crop_to_alpha(img: Image.Image) -> Image.Image:
    """Crop an RGBA image to the bounding box of its non-transparent pixels."""
    bbox = img.split()[3].getbbox()
    return img.crop(bbox) if bbox else img


# --------------------------------------------------------------------------- #
# outline mode: die-cut solid outline
# --------------------------------------------------------------------------- #

def outline_cutout(png_bytes: bytes, color=(255, 255, 255, 255), width: int = 12,
                   smooth: bool = True) -> bytes:
    """Add a solid-color stroke hugging the object's silhouette. Returns PNG bytes."""
    img = _crop_to_alpha(_to_rgba(png_bytes))
    pad = width + 2
    w, h = img.size
    canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    canvas.alpha_composite(img, (pad, pad))

    mask = dilate_alpha(canvas.split()[3], width)
    if smooth:
        mask = smooth_mask(mask, max(0.6, width * 0.1))

    stroke = Image.new("RGBA", canvas.size, tuple(color[:3]) + (0,))
    stroke.putalpha(mask)

    out = Image.alpha_composite(stroke, canvas)
    return _to_png_bytes(_crop_to_alpha(out))


# --------------------------------------------------------------------------- #
# ai_outline finishing helpers: white edge + drop shadow
# --------------------------------------------------------------------------- #

def add_white_edge(png_bytes: bytes, width: int = 8,
                   color=(255, 255, 255, 255)) -> bytes:
    """Outermost die-cut white stroke around the whole sticker (reuses outline_cutout)."""
    return outline_cutout(png_bytes, color=color, width=width, smooth=True)


def add_drop_shadow(png_bytes: bytes, offset=(0, 6), blur: int = 12,
                    opacity: int = 110) -> bytes:
    """Place a soft shadow under the whole sticker. Returns PNG bytes."""
    img = _crop_to_alpha(_to_rgba(png_bytes))
    w, h = img.size
    pad = blur * 2 + max(abs(offset[0]), abs(offset[1]))
    canvas_size = (w + 2 * pad, h + 2 * pad)

    shadow_alpha = img.split()[3].point(lambda v: int(v * opacity / 255))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(shadow_alpha)

    shadow_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    shadow_canvas.alpha_composite(shadow, (pad + offset[0], pad + offset[1]))
    shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(blur))

    obj_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    obj_canvas.alpha_composite(img, (pad, pad))

    out = Image.alpha_composite(shadow_canvas, obj_canvas)
    return _to_png_bytes(_crop_to_alpha(out))


# --------------------------------------------------------------------------- #
# ai_outline mode: outline-hugging AI border (silhouette-following decoration)
# --------------------------------------------------------------------------- #

def ai_outline_border(cutout_png: bytes, instruction: str, model: str,
                      band_ratio: float = 0.06, band_color=(255, 255, 255, 255),
                      bg_color=_DEFAULT_BG, paste_back: bool = True,
                      white_edge: bool = False,
                      white_edge_width: int = 8, shadow: bool = True,
                      temperature: float = 1.0, work_size: int = 1024,
                      debug_dir=None, guide_min_ratio: float = 0.10) -> bytes:
    """Decorative border that hugs the subject's silhouette.

    The shape is enforced deterministically (Pillow) and only the *pattern* comes
    from the image model -- so the result is stable regardless of how the model
    behaves:

      1. Draw a solid contrasting band that traces the subject's outline and show
         it to the model on a removable background (anchors it to the silhouette).
      2. Model redraws the band as the requested pattern.
      3. Use our own smoothed mask (subject + outline band) as the final alpha and
         take the model's RGB through it. We do NOT chroma-key the model output:
         these models often ignore the "plain background" instruction and flood the
         decoration to the corners, which would make a corner-sampled color-key
         delete the border instead of the background. Driving the alpha from our own
         shape is immune to whatever background the model paints.
      4. Optional white die-cut edge + drop shadow.

    Guide vs final band are decoupled. The model is fed a *guide* band that is
    always wide enough (``guide_min_ratio`` floor, and never narrower than the final
    band) so it reliably paints decoration across the whole band instead of treating
    a thin band as a subject outline and flooding the background. The user-facing
    ``band_ratio`` only sets the *final* (displayed) band width: the final alpha is a
    narrower ``final_mask`` sitting well inside the model's painted guide band, so its
    outer edge never reaches the model's background -> no halo. ``band_ratio`` can be
    made arbitrarily small for a thin clean border without bleed.

    ``paste_back=True`` (default): the band ring is the model's pattern and the
    subject opening gets the real cutout pasted back. ``paste_back=False``: the
    whole final band region (model's own subject + border) is kept as-is.
    """
    _dump = _make_dump(debug_dir)

    subject = _crop_to_alpha(_to_rgba(cutout_png))
    sub_w, sub_h = subject.size
    # Work at a capped resolution: the band/dilation math scales with the subject's
    # pixel size, so on a full-res photo the guide band balloons to ~150px and the
    # box-kernel dilation stair-steps badly; downscaling keeps the band a sane width,
    # the morphology smooth, and stops the model's ~1024 output being upsampled.
    if max(sub_w, sub_h) > work_size:
        scale = work_size / max(sub_w, sub_h)
        subject = subject.resize((round(sub_w * scale), round(sub_h * scale)), Image.LANCZOS)
        sub_w, sub_h = subject.size
    _dump("0_subject.png", subject)
    max_side = max(sub_w, sub_h)
    # final_width: what the user sees. guide_width: what the model is shown -- always
    # at least guide_min_ratio of the subject AND always wider than final by `safety`,
    # so the final band edge sits inside the model's painted decoration and never
    # reaches its background. The gap (guide - final) is the "waste" ring we crop off.
    final_width = max(8, round(band_ratio * max_side))
    safety = max(8, round(final_width * 0.3))
    guide_width = max(round(guide_min_ratio * max_side), final_width) + safety
    pad = guide_width + max(8, round(guide_width * 0.3))  # room for the guide band + smoothing
    offset = (pad, pad)
    canvas_size = (sub_w + 2 * pad, sub_h + 2 * pad)

    # 1: subject silhouette on the canvas, plus a solid band hugging its outline.
    subj_alpha = Image.new("L", canvas_size, 0)
    subj_alpha.paste(subject.split()[3], offset)
    # Smooth the band: box-kernel dilation leaves a faceted/stair-stepped outer
    # edge, and that edge is the template the model traces, so round it first.
    guide_mask = smooth_mask(dilate_alpha(subj_alpha, guide_width), max(1.0, guide_width * 0.15))
    # final_mask is the narrower displayed band; its outer edge is >= `safety` px inside
    # the guide band, deep within the model's decoration, so no background leaks in.
    final_mask = smooth_mask(dilate_alpha(subj_alpha, final_width), max(1.0, final_width * 0.15))

    prepped = Image.new("RGBA", canvas_size, tuple(bg_color[:3]) + (255,))
    band_layer = Image.new("RGBA", canvas_size, tuple(band_color[:3]) + (0,))
    band_layer.putalpha(guide_mask)            # wide guide band shown to the model
    prepped.alpha_composite(band_layer)        # contrasting band tracing the outline
    prepped.alpha_composite(subject, offset)   # real subject inside the band
    _dump("1_prepped.png", prepped)

    # 2: model redraws the band as the requested pattern.
    frame_bytes = _gemini_generate_image(
        _to_png_bytes(prepped.convert("RGB")), instruction, model, temperature=temperature
    )
    frame = _to_rgba(frame_bytes).resize(canvas_size, Image.LANCZOS)
    _dump("2_gemini.png", frame)

    # 3: drive the final alpha from our OWN smoothed final_mask, not from the model's
    #    background. (See docstring: corner-sampled color-keying breaks when the model
    #    floods the decoration to the corners.)
    if paste_back:
        # Keep only the band ring (final_mask minus the subject opening); the real
        # cutout is pasted into the opening next, so its color stays faithful.
        # Erode (not dilate) the hole so the ring tucks a few px *under* the subject
        # the pasted cutout covers it -- dilating left a transparent seam at the edge.
        hole = erode_alpha(subj_alpha, max(2, round(0.01 * max(canvas_size))))
        border_alpha = ImageChops.subtract(final_mask, hole)
        frame.putalpha(border_alpha)
        _dump("3_shaped.png", frame)

        subject_full = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        subject_full.alpha_composite(subject, offset)
        out = Image.alpha_composite(frame, subject_full)
        _dump("5_pasted.png", out)
    else:
        # Keep the whole final_mask region (model's own subject + border together).
        frame.putalpha(final_mask)
        _dump("3_shaped.png", frame)
        out = frame
        _dump("4_model_output.png", frame)
    out_bytes = _to_png_bytes(_crop_to_alpha(out))
    _dump("6_composed.png", _to_rgba(out_bytes))

    # 6: finishing touches.
    if white_edge:
        out_bytes = add_white_edge(out_bytes, white_edge_width)
        _dump("7_white_edge.png", _to_rgba(out_bytes))
    if shadow:
        out_bytes = add_drop_shadow(out_bytes)
        _dump("8_shadow.png", _to_rgba(out_bytes))
    return out_bytes

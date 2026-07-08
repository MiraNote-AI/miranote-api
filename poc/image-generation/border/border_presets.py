"""
Border-style presets for the /border endpoint (ai_outline mode).

Each preset maps a key to a natural-language style description. A request either
names a preset key or passes a free-form ``prompt``; both are wrapped in the
outline skeleton below that tells the model to repaint the silhouette-hugging band
as that style. The band shape itself is enforced in Pillow (see border.py), so the
skeleton only has to steer the *pattern*.
"""

BORDER_PRESETS = {
    "floral": (
        "a hand-drawn floral frame of small dense flowers and leaves, soft pastel "
        "colors, cute and delicate"
    ),
    "lace": (
        "an elegant white lace doily frame with fine scalloped openwork edges, "
        "delicate and ornamental"
    ),
    "film_strip": (
        "a retro 35mm film strip frame with sprocket holes along the edges, matte "
        "black with a subtle vintage grain"
    ),
    "ribbon": (
        "a glossy satin ribbon frame with a small bow, soft folds and gentle "
        "highlights, cute and gift-like"
    ),
    "dashed_tape": (
        "a playful washi-tape and dashed-line journaling frame, pastel paper tape "
        "pieces at the corners, casual hand-made look"
    ),
    "gold_frame": (
        "an ornate vintage gold picture frame with baroque carved scrollwork, warm "
        "metallic sheen, elegant and luxurious"
    ),
}


# Outline-hugging skeleton (ai_outline mode). The input image already shows a solid
# band tracing the subject's silhouette, so the model only has to repaint that band
# -- this anchors the decoration to the contour and stops it from drawing a
# rectangular picture frame.
_OUTLINE_SKELETON = (
    "The image shows a subject wrapped in a plain, solid-color band that traces "
    "its exact outline/silhouette. "
    "Redraw ONLY that band as a decorative border in this style: {style}. "
    "Follow the band's exact shape, which hugs the subject's contour, so the "
    "border curves with the subject. "
    "Do NOT draw a rectangular frame or any straight-edged border. "
    "Do not cover, crop, redraw, or alter the subject itself. "
    "Keep the background a plain, solid, flat single-color that strongly "
    "contrasts with the border so it can be cleanly removed. "
    "Do not add any text, watermark, or signature."
)


def _build(skeleton: str, style: str, prompt: str) -> str:
    if style:
        if style not in BORDER_PRESETS:
            raise ValueError(
                f"unknown border style '{style}'; valid: {list(BORDER_PRESETS)}"
            )
        return skeleton.format(style=BORDER_PRESETS[style])
    if prompt:
        return skeleton.format(style=prompt)
    raise ValueError("either 'style' (preset key) or 'prompt' (custom) is required")


def build_outline_instruction(style: str = "", prompt: str = "") -> str:
    """Return the full instruction for a preset key or a custom prompt (ai_outline).

    Raises ValueError if neither is usable so the endpoint can map it to a 400.
    """
    return _build(_OUTLINE_SKELETON, style, prompt)

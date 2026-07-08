"""
Style-transfer presets for the /stylize endpoint.

Each preset maps a style key to a natural-language instruction. Every
instruction insists on preserving the original composition/subject/pose so the
model restyles the photo rather than repainting new content. Custom styles are
wrapped in the same "keep composition, only change style" skeleton.
"""

# Shared skeleton: applied to both presets and free-form custom styles so the
# model treats every request as a restyle, not a fresh generation.
_SKELETON = (
    "Restyle this photo in the following art style: {style}. "
    "Keep the original composition, subjects, poses, and layout exactly the same; "
    "only change the rendering style. Do not add, remove, or rearrange objects. "
    "Do not add any text, watermark, signature, or border."
)

STYLE_PRESETS = {
    "impressionist": (
        "an impressionist oil painting with visible loose brushstrokes, soft "
        "dappled light, rich layered color, in the style of Claude Monet"
    ),
    "cartoon": (
        "a clean cartoon / anime illustration with bold smooth outlines, flat "
        "cel-shaded color blocks, and simplified shapes"
    ),
    "sketch": (
        "a hand-drawn pencil sketch with graphite shading, cross-hatching, and "
        "delicate sketchy linework on paper"
    ),
    "line_art": (
        "minimalist line art: clean black ink outlines on a plain white "
        "background, no shading, no color fills"
    ),
}


def build_instruction(style: str = "", prompt: str = "") -> str:
    """Return the full instruction for a preset key or a custom prompt.

    Raises ValueError if neither is usable so the endpoint can map it to a 400.
    """
    if style:
        if style not in STYLE_PRESETS:
            raise ValueError(
                f"unknown style '{style}'; valid: {list(STYLE_PRESETS)}"
            )
        return _SKELETON.format(style=STYLE_PRESETS[style])
    if prompt:
        return _SKELETON.format(style=prompt)
    raise ValueError("either 'style' (preset key) or 'prompt' (custom) is required")

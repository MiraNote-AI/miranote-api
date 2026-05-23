QUALITY_SUFFIX = (
    "flat design, white background, high quality, no text, no watermark, no shadow"
)

def sticker(subject: str, color_palette: str = "pastel") -> str:
    return (
        f"Cute kawaii {subject} sticker illustration, {color_palette} color palette, "
        f"Japanese journaling aesthetic, clean outline, {QUALITY_SUFFIX}"
    )


def background(season: str = "spring", mood: str = "cozy") -> str:
    return (
        f"{season.capitalize()} themed journaling page background, {mood} atmosphere, "
        f"watercolor texture, soft colors, subtle pattern, {QUALITY_SUFFIX}"
    )


def divider(style: str = "floral") -> str:
    return (
        f"A horizontal {style} decorative divider line for digital journaling. "
        f"The centerpiece divider consists of green leaves, white blossoms, and gold vine elements. "
        f"The background is a solid, clean, uniform bright magenta color. "
        f"High contrast, distinct separation between the divider and the background, "
        f"flat vector illustration, minimalist planner asset, crisp edges"
    )


SAMPLE_BATCH = [
    ("sticker", sticker("cherry blossom")),
    ("sticker", sticker("strawberry", "red and pink")),
    ("sticker", sticker("moon and stars", "purple and gold")),
    ("background", background("spring", "dreamy")),
    ("background", background("autumn", "warm and cozy")),
    ("divider", divider("floral")),
    ("divider", divider("ribbon")),
]

"""
Style-transfer presets for the /stylize endpoint.

Two wrappers, because /stylize is asked for two different things and one
sentence cannot honestly frame both:

  `style`   a preset key. The value is an art style written as a noun phrase
            ("an impressionist oil painting with..."), so it needs a sentence
            built around it, and "only change the rendering style" is true.

  `prompt`  free text the user typed. This is the whole of what the app sends --
            "Tell AI what to change" in the photo panel, the instruction box in
            the sticker panel, and the user's own words passed through by Mira
            chat -- and it is usually an edit, not a style.

They shared _SKELETON until a 52-call comparison showed what that costs. An edit
request dropped into the "{style}" slot reads as a style name, so
"give it a red scarf" arrived as "Restyle this photo in the following art style:
give it a red scarf ... only change the rendering style", and the models did as
asked: gemini-3.1-flash-lite-image re-rendered a flat cartoon apple sticker as
felt. Sent through _INSTRUCTION instead, the same model kept the artwork and
added the scarf. Evidence:
test_output/stylize_prompt_arms/20260925_221942/contact_sheet_edit.png
"""

# Presets only. Unchanged: for a genuine style name every clause here is true,
# and nothing the app sends reaches it today.
_SKELETON = (
    "Restyle this photo in the following art style: {style}. "
    "Keep the original composition, subjects, poses, and layout exactly the same; "
    "only change the rendering style. Do not add, remove, or rearrange objects. "
    "Do not add any text, watermark, signature, or border."
)

# Free text. Says nothing about what kind of change was asked for, and forbids
# nothing the user may have just requested.
#
# The "keep what was not mentioned" clause is the honest version of _SKELETON's
# protective half, and it serves both intents from one sentence with no
# classifier deciding which this is: a restyle mentions the rendering, so the
# rendering may change; it does not mention the subject, so the subject may not.
#
# The last line is not decoration. Dropping the wrapper entirely also passed the
# edit cases, but on "make it look like a Monet oil painting" it let
# gemini-3.1-flash-lite-image return a photograph of a framed oil painting --
# gilt frame, canvas texture and all -- where this line keeps the result an
# image of the subject. See contact_sheet_restyle.png beside the sheet above.
_INSTRUCTION = (
    "{instruction}. "
    "Change only what that asks for: keep the composition, the subject, and "
    "everything the instruction does not mention exactly as they are. "
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

    The two take different wrappers; see the module docstring for why.

    Raises ValueError if neither is usable so the endpoint can map it to a 400.
    """
    if style:
        if style not in STYLE_PRESETS:
            raise ValueError(
                f"unknown style '{style}'; valid: {list(STYLE_PRESETS)}"
            )
        return _SKELETON.format(style=STYLE_PRESETS[style])
    if prompt:
        return _INSTRUCTION.format(instruction=prompt)
    raise ValueError("either 'style' (preset key) or 'prompt' (custom) is required")

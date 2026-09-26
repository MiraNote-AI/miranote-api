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

# Appended when the caller is going to cut the result out again -- which, in
# practice, means the input was a sticker.
#
# This is the request generate/sticker_suffix.txt has been making on the
# CREATION side all along, and it is the whole reason a generated sticker cuts
# out cleanly while an edited one did not. Left to itself the model draws a
# sticker on white, which is also the colour of the die-cut edge the sticker
# carries, and the matte that has to find that edge has only a faint drop shadow
# to go on. Measured over five stickers, the edge came back 0-1% saturated
# against 49-57% for the alternative -- flattening the input onto a colour of
# our own, which tints every semi-transparent edge pixel.
#
# Two wordings here are deliberate:
#
#   "edge", not "border". _INSTRUCTION already says "Do not add any text,
#   watermark, signature, or border", and a second sentence using that word
#   would contradict the first -- the exact defect that started this whole
#   investigation.
#
#   "keep", not "add", and only "if the subject already has" one. A sticker
#   without a die-cut edge must not be given one.
#
# The second sentence earns its place: without it the painterly sticker of the
# five came back as bare artwork with the edge gone.
#   poc/image-generation/test_output/stylize_model_bg/20260926_073455/
_SOLID_BACKGROUND = (
    " Place the result on a solid flat single-colour background. That colour "
    "must not appear anywhere in the subject itself and must contrast strongly "
    "with the subject's palette, so the background can be removed cleanly "
    "afterwards."
    " If the subject already has a white die-cut edge around its outline, keep "
    "that edge."
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


def build_instruction(style: str = "", prompt: str = "",
                      cut_out_afterwards: bool = False) -> str:
    """Return the full instruction for a preset key or a custom prompt.

    The two take different wrappers; see the module docstring for why.

    `cut_out_afterwards` says the caller will matte the result -- the sticker
    path -- and adds the backdrop request that makes that matte possible. It is
    off by default because a photo must not be asked for a flat backdrop: that
    would replace the scene the user wanted edited.

    Raises ValueError if neither is usable so the endpoint can map it to a 400.
    """
    if style:
        if style not in STYLE_PRESETS:
            raise ValueError(
                f"unknown style '{style}'; valid: {list(STYLE_PRESETS)}"
            )
        return _SKELETON.format(style=STYLE_PRESETS[style])
    if prompt:
        sent = _INSTRUCTION.format(instruction=prompt)
        return sent + _SOLID_BACKGROUND if cut_out_afterwards else sent
    raise ValueError("either 'style' (preset key) or 'prompt' (custom) is required")

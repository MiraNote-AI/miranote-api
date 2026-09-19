# Configuration for the image service (main.py).
# Grouped by the pipeline that uses each setting; a shared section holds the
# values used by more than one pipeline. Changing a value here changes behavior
# for every pipeline listed above its section.

# --------------------------------------------------------------------------- #
# Shared across pipelines
# --------------------------------------------------------------------------- #
# /generate output. Two predecessors sat here: Imagen 4 until 2026-09-14, which
# is enabled per project on Vertex and was enabled on none of ours, so every
# process paid a 404 on its first request; then gemini-2.5-flash-image until
# 2026-09-15, which answered but ignored the aspect ratio in the prompt and
# returned 1024x1024 for every command, making each "background" a square.
# This model honors it (9:16 measured as 768x1376) and returns a ~90 KB JPEG
# rather than a ~900 KB PNG, which is most of the wall clock on a phone.
#
# It answers only on the global endpoint: us-central1 returns 404 NOT_FOUND for
# it, so LOCATION must be "global" (see .env.example). Every other model named
# in this file was verified to answer there too.
#
# Deliberately NOT the same id as STYLE_MODEL / BORDER_MODEL. Vertex meters
# image generation as 1/min/{project}/{base_model}, so keeping /generate on its
# own model gives it a quota bucket that /stylize and /border cannot drain.
MODEL_ID = "gemini-3.1-flash-lite-image"
PROMPT_EXPANDER_MODEL = "gemini-2.5-flash"  # prompt expansion: /generate + /describe

# /describe default: one sentence about a photo, for the app's page
# context. Every ImageRef.summary the app has ever stored was written
# with this wording, so changing it changes what old pages "say".
DESCRIBE_PROMPT = (
    "Describe this photo in one warm, concrete sentence "
    "(what is in it, the mood). Answer with the sentence only."
)


def describe_question(prompt=None):
    """What /describe should ask about the image.

    Callers may bring their own question -- canvas mode asks what a
    whole page looks like rather than what a photo shows. Photo import
    passes nothing and keeps the default.
    """
    return (prompt or "").strip() or DESCRIBE_PROMPT


REMBG_MODEL = "birefnet-general-lite"  # the named-only background remover now that Vision is the default everywhere; nothing reaches it unless a caller asks for it by name or ROLLING BACK TO REMBG below is applied. options: u2net, birefnet-general-lite, birefnet-general. full birefnet takes ~80s/cutout on an M-series Mac and starves the event loop -- too slow for interactive use (phone times out at 150s and users retry, wedging the queue)
REMBG_ERODE_RADIUS = 0  # pixels to erode alpha edge inward; 0 to disable. /generate + /cutout

# --------------------------------------------------------------------------- #
# /generate  (sticker & background image generation)
# --------------------------------------------------------------------------- #
# One image per request, not two. Vertex meters image generation at
# 1/min/{project}/{base_model} with an effective limit of 2 (api #69), and the
# two calls were dispatched concurrently, so a single /generate needed the whole
# minute's budget banked at the same instant and failed outright against a
# partially refilled bucket -- measured as five consecutive 503s at 45s spacing
# with no other traffic, while a lone call succeeded in 3.6s immediately after.
# The app's picker renders whatever count comes back, so this turns "pick one of
# two" into "keep it or discard it" rather than breaking anything.
NUMBER_OF_IMAGES = 1
ASPECT_RATIOS = {
    "sticker":    "1:1",
    "background": "9:16",
    "art":        "1:1",
}
REMOVE_BG = True  # set to False to skip background removal for testing

# Which background remover a generated sticker goes through. On the 12-image
# sticker bench (test_output/sticker_matte/) Vision was 94.6x faster --
# 0.16 s against rembg's 15.37 s per image -- at equal or better quality:
# identical coverage (alpha IoU 0.96-0.998), no background fringe on either,
# and a body at a true alpha 255 where rembg sits at 254 across the whole
# sticker. The one regression is that Vision drops large pale low-contrast
# areas inside a sticker (26 holes, 2.1% of the subject, on one of the twelve).
# Vision-only, like the two /cutout defaults: a Vision failure here answers 503
# and discards the generated image rather than quietly re-cutting it with rembg.
# See ROLLING BACK TO REMBG below.
DEFAULT_STICKER_MATTE = "vision"       # Apple Vision foreground matte ("rembg" to roll back)

# --------------------------------------------------------------------------- #
# /stylize  (image-to-image style transfer)
# --------------------------------------------------------------------------- #
STYLE_MODEL = "gemini-2.5-flash-image"  # Nano Banana image-to-image; verify GA/preview id on Vertex
STYLE_TEMPERATURE = 0  # default for /stylize; lower = more faithful to the original photo

# --------------------------------------------------------------------------- #
# /cutout  (background removal + prompt-guided cutout)
# --------------------------------------------------------------------------- #
# Default for a prompted /cutout. On the 17-image bench the Vision matte beat
# rembg on both axes -- 23.0 s -> 5.5 s end to end, better on 3 images (parfait,
# noodle soup, mango ice) and worse on none.
DEFAULT_PROMPT_CUTOUT_MODE = "hybrid_sam_prebg_vision"  # vision matte -> gray bg -> hybrid_sam_union

# Same for a /cutout with no prompt, where the whole foreground is the subject
# and no detector or SAM runs. The gap is even wider here than on the prompted
# path, because there is no 5 s of DINO + SAM to hide it: the bench put the
# Vision matte at 0.12 s against rembg's 12.55 s, on a matte whose median alpha
# was a uniform 254.
DEFAULT_AUTO_CUTOUT_MODE = "vision"        # Apple Vision foreground matte

# ROLLING BACK TO REMBG
#
# The service is Vision-only: nothing downgrades itself. There is deliberately
# no startup step-down, because a machine that could not build the helper would
# then serve rembg at 12-80 s per image while every log line and response still
# looked normal. Rolling back is a deliberate edit of these three constants:
#
#   DEFAULT_PROMPT_CUTOUT_MODE = "hybrid_sam_prebg_gray"   # rembg -> gray bg -> hybrid_sam_union
#   DEFAULT_AUTO_CUTOUT_MODE   = "auto"                    # rembg REMBG_MODEL
#   DEFAULT_STICKER_MATTE      = "rembg"                   # rembg REMBG_MODEL
#
# Every rembg code path is still present and tested; only the defaults point
# away from it. A caller can also reach it per-request with ?mode= / "matte".

# Apple Vision matte helper (hybrid_sam_prebg_vision, and /cutout?mode=vision).
# Build it with:
#   swiftc -O -parse-as-library vision_bench/vision_bench.swift -o vision_bench/vision_bench
# macOS 15+ only -- vision_bench.swift uses the Swift-native Vision API
# (ImageRequestHandler / GenerateForegroundInstanceMaskRequest), which does not
# compile on macOS 14. Without this binary every cutout fails; scripts/
# start_backends.sh builds it, and /health reports whether it came up.
VISION_MATTE_BIN = "vision_bench/vision_bench"

# Seconds to wait for one matte before giving up on the helper and killing it.
#
# The helper is one conversation over one pipe, so a request holds it for the
# whole round trip -- meaning a helper that hangs (alive, but never answering)
# would block not just its own request but every cutout and sticker behind it,
# forever, until someone restarted the server. This bound is what turns that
# into one failed request. Vision answers in 0.03-0.12 s, so 30 s can only be
# hit by a hang, never by a slow image.
VISION_MATTE_TIMEOUT = 30

# Gemini bbox detector (hybrid disambiguator)
BBOX_DETECTOR_MODEL = "gemini-2.5-flash"

# SAM 2.1 segmenter
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_DEVICE = "mps"
SAM_MASK_GAIN = 5.0
SAM_BOX_PADDING_RATIO = 0.0  # expand the box fed to SAM by this fraction per side so edges aren't clipped (0 = off)
SAM_BINARIZE_MASK = False  # True: hard alpha (crisp edges); False: soft sigmoid alpha (keeps fuzzy/hair edges)
SAM_CACHE_DIR = "~/.miranote_sam"

# GroundingDINO open-vocabulary detector
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
GROUNDING_TEXT_THRESHOLD = 0.25
GROUNDING_DEVICE = "cpu"

# Hybrid matching (DINO candidates disambiguated by the Gemini box)
HYBRID_DINO_THRESHOLD = 0.20
HYBRID_IOU_THRESHOLD = 0.5  # min IoU for a DINO box to be accepted over the Gemini box

# --------------------------------------------------------------------------- #
# /border  (sticker frames / borders)
# --------------------------------------------------------------------------- #
BORDER_MODEL = "gemini-2.5-flash-image"  # ai_outline: Nano Banana img2img; same model as STYLE_MODEL (verified available)
BORDER_TEMPERATURE = 1.0  # ai_outline; higher = more creative decoration
BORDER_WIDTH = 12  # outline: default stroke width (px)
BORDER_COLOR = "#FFFFFF"  # outline: default stroke color
WHITE_EDGE_WIDTH = 8  # ai_outline: outermost die-cut white edge (px)
# ai_outline (outline-hugging AI border)
BORDER_BAND_RATIO = 0.06  # final (displayed) band width as a fraction of the subject's max side
BORDER_GUIDE_MIN_RATIO = 0.10  # min width of the wide guide band shown to the model (AI reliability floor); decoupled from the displayed width to avoid background halo at small band_ratio
BORDER_WORK_SIZE = 1024  # cap subject max side to this before processing (speed + smoothness)

# Configuration for the image service (main.py).
# Grouped by the pipeline that uses each setting; a shared section holds the
# values used by more than one pipeline. Changing a value here changes behavior
# for every pipeline listed above its section.
import os

# --------------------------------------------------------------------------- #
# Shared across pipelines
# --------------------------------------------------------------------------- #
# GEMINI MODEL LAUNCH STAGE, checked 2026-09-29.
#
# Vertex encodes the stage in the id: preview and experimental models carry
# "-preview" or "-exp" (the catalogue for this project returns
# gemini-2.5-flash-preview-04-17, gemini-2.5-pro-exp-03-25,
# gemini-3.1-flash-image-preview, gemini-3.1-pro-preview and others). None of
# the four ids this file uses carries either marker, and all four resolve
# through models.get():
#
#   gemini-3.1-flash-lite         BBOX_DETECTOR_MODEL
#   gemini-2.5-flash              DESCRIBE_MODEL
#   gemini-3.1-flash-lite-image   MODEL_ID, STYLE_MODEL
#   gemini-2.5-flash-image        BORDER_MODEL
#
# What that does and does not establish: the naming convention is Google's own
# and it is consistent across the 27 models this project can see, so the
# absence of a marker is real evidence. It is NOT a published GA declaration --
# the SDK exposes only name and version, with no launch-stage field, so a
# documentation page is the only authoritative source and it was not reachable
# from here. Treat these as "not marked preview" rather than "confirmed GA",
# and re-check before relying on a deprecation window.
#
# Note gemini-2.5-flash-image is absent from models.list() while models.get()
# resolves it, so that listing is not exhaustive and absence from it means
# nothing on its own.
#
# This replaces the "# verify GA/preview id on Vertex" note that lived on
# origin/dev's STYLE_MODEL line and was dropped when that block was rewritten.
# The Gemini image model, and the way back off DashScope.
#
# /generate no longer calls it by default -- see IMAGE_MODELS_BY_MODE below --
# but this id stays because it is the rollback: IMAGE_MODEL=<this> restores the
# pre-DashScope path with no code change, and image_providers.py routes it to
# generate/gemini_image.py.
#
# History, kept because two of these were paid for: Imagen 4 until 2026-09-14,
# enabled per project on Vertex and enabled on none of ours, so every process
# paid a 404 on its first request; then gemini-2.5-flash-image until 2026-09-15,
# which answered but ignored the aspect ratio in the prompt and returned
# 1024x1024 for every command, making each "background" a square. This model
# honors it (9:16 measured as 768x1376) and returns a ~90 KB JPEG rather than a
# ~900 KB PNG.
#
# It answers only on the global endpoint: us-central1 returns 404 NOT_FOUND for
# it, so LOCATION must be "global" (see .env.example). Every other Gemini model
# named in this file was verified to answer there too.
#
# Deliberately NOT the same id as STYLE_MODEL / BORDER_MODEL. Vertex meters
# image generation as 1/min/{project}/{base_model}, so a rollback of /generate
# lands in its own quota bucket rather than draining /stylize and /border.
MODEL_ID = "gemini-3.1-flash-lite-image"

# /generate prompt expansion. A text-only job, so a text-only model is enough.
# Moved to qwen-turbo with the image models: on the 90-image bench it produced
# expansions of the same shape and length as gemini-2.5-flash at 1/14th the cost
# and half the latency. It is about 1% of what a /generate call costs either way
# -- the money is in the image.
PROMPT_EXPANDER_MODEL = "qwen-turbo"

# /describe is a VISION call -- it sends image bytes -- so this model must
# accept images. It is deliberately NOT PROMPT_EXPANDER_MODEL: the two shared
# one constant until this change, and pointing that constant at qwen-turbo (text
# only) would have broken /describe silently, which poc/chatbot depends on for
# canvas mode.
DESCRIBE_MODEL = "gemini-2.5-flash"

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
# Left at 1 through the move to DashScope. The Vertex quota that forced it here
# no longer applies, but DashScope's own concurrency limits have not been
# measured, and this is not the change to find them with. Raising it is one line.
NUMBER_OF_IMAGES = 1
ASPECT_RATIOS = {
    "sticker":    "1:1",
    "background": "9:16",
    "art":        "1:1",
}
REMOVE_BG = True  # set to False to skip background removal for testing

# --------------------------------------------------------------------------- #
# /generate  --  image providers
#
# /generate runs on DashScope (Aliyun Bailian) as of this change. Vertex is
# still required by /describe, /stylize and /border, so this is not a
# Google-free service -- ADC and PROJECT_ID are still mandatory.
# --------------------------------------------------------------------------- #
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL",
                               "https://dashscope.aliyuncs.com/api/v1")  # Beijing

# The DashScope text-to-image models, and the request shape each one answers on.
# Two shapes, because the platform genuinely has two endpoints:
#   mm_sync    multimodal-generation/generation -- answers immediately
#   t2i_async  text2image/image-synthesis -- returns a task id to poll
# Shapes are documented in generate/dashscope_image.py. Availability is
# region-dependent and the vendor docs disagree about which endpoint the older
# wan models still answer on, so treat an id here as "was verified once", not
# "is guaranteed".
IMAGE_MODELS = {
    "z-image-turbo":     {"provider": "dashscope", "shape": "mm_sync"},
    "qwen-image-3.0":    {"provider": "dashscope", "shape": "mm_sync"},
    "wan2.2-t2i-flash":  {"provider": "dashscope", "shape": "t2i_async"},
    "wanx2.0-t2i-turbo": {"provider": "dashscope", "shape": "t2i_async"},
}

# Every model is asked for the same pixel budget, so a side-by-side review does
# not read "bigger" as "better". DashScope has a real `size` parameter, unlike
# Nano Banana which took the aspect ratio in prose and ignored it.
TARGET_SIZES = {"1:1": (1024, 1024), "9:16": (720, 1280)}


def size_for(model: str, aspect_ratio: str) -> tuple[int, int]:
    """The (width, height) to request from `model` for this aspect ratio."""
    return TARGET_SIZES[aspect_ratio]


# List prices per image, checked on the date below. Re-check it when you touch
# these: a stale cost column reads as authoritative whether or not it is.
#   help.aliyun.com/zh/model-studio/model-pricing
#   ai.google.dev/gemini-api/docs/pricing
PRICES_CHECKED = "2026-09-05"
USD_TO_CNY = 7.1

# Per image, in CNY. Only the DashScope models are here.
#
# MODEL_ID is deliberately absent: nobody has looked up what
# gemini-3.1-flash-lite-image costs, and a guessed number in a price table is
# worse than a gap, because a cost column reads as authoritative either way.
# Fill it in before using this table to argue about the rollback.
IMAGE_PRICE_CNY = {
    # 0.10 only because PROMPT_EXTEND is False; it is 0.20 with rewriting on.
    "z-image-turbo":     0.10,
    "qwen-image-3.0":    0.20,
    "wan2.2-t2i-flash":  0.14,
    "wanx2.0-t2i-turbo": 0.04,
}

# (input, output) CNY per 1M tokens, for the two text models. The Gemini row is
# converted from $0.30 / $2.50 at the rate above, so it is an estimate; the qwen
# row is a Bailian list price.
TEXT_PRICE_CNY_PER_MTOK = {
    "qwen-turbo":       (0.30, 0.60),
    "gemini-2.5-flash": (0.30 * USD_TO_CNY, 2.50 * USD_TO_CNY),
}

# DashScope will rewrite the prompt for you if allowed. Off for two reasons:
# it would undo the expansion PROMPT_EXPANDER_MODEL just produced, and on
# z-image-turbo it is also the difference between 0.10 and 0.20 CNY per image --
# turning this on doubles the /generate bill. A unit test guards it.
PROMPT_EXTEND = False

POLL_INTERVAL = 2.0    # seconds between task-status polls on the async shape
IMAGE_TIMEOUT = 180.0  # per-image ceiling, including polling and download

# Which model generates each /generate mode. One entry per mode rather than one
# global constant, because the 90-image bench says these three will not stay
# equal: z-image-turbo was the fastest and cheapest of six everywhere, but it
# was also the weakest at following decorative and spatial instructions, which
# only background asks for (it flattened "film border", "trees in four corners"
# and "travel route" into plain gradients where wan2.2-t2i-flash drew them).
# Shaped like ASPECT_RATIOS above, so switching one mode later is this one line.
IMAGE_MODELS_BY_MODE = {
    "sticker":    "z-image-turbo",
    "background": "z-image-turbo",   # candidate to revisit: wan2.2-t2i-flash
    "art":        "z-image-turbo",
}

# One model for every mode, for walking a ladder by restarting the server
# instead of editing this file mid-experiment. Accepts any key of IMAGE_MODELS
# plus MODEL_ID, which is also the rollback:
#   IMAGE_MODEL=gemini-3.1-flash-lite-image
# restores the pre-DashScope path for all three modes.
_image_model_override = os.getenv("IMAGE_MODEL", "").strip()
if _image_model_override:
    IMAGE_MODELS_BY_MODE = {mode: _image_model_override
                            for mode in IMAGE_MODELS_BY_MODE}

# Fail at import, not on the first paid call.
_KNOWN_IMAGE_MODELS = set(IMAGE_MODELS) | {MODEL_ID}
for _mode, _model in IMAGE_MODELS_BY_MODE.items():
    if _model not in _KNOWN_IMAGE_MODELS:
        raise ValueError(f"IMAGE_MODELS_BY_MODE[{_mode!r}]={_model!r} is not one "
                         f"of {sorted(_KNOWN_IMAGE_MODELS)}")
# A mode present in one dict and missing from the other would either 500 on that
# command or silently generate at the wrong aspect ratio, so the two are
# required to describe the same set of modes.
if set(IMAGE_MODELS_BY_MODE) != set(ASPECT_RATIOS):
    raise ValueError(f"IMAGE_MODELS_BY_MODE covers {sorted(IMAGE_MODELS_BY_MODE)} "
                     f"but ASPECT_RATIOS covers {sorted(ASPECT_RATIOS)}")

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
# Image-to-image for /stylize, which the app reaches from "Ask AI" on a photo
# and from "Edit sticker".
#
# Deliberately NOT the same id as BORDER_MODEL, for the same reason MODEL_ID is
# not: Vertex meters image generation at 1/min/{project}/{base_model}, so two
# pipelines sharing an id share a bucket and starve each other. They were both
# on gemini-2.5-flash-image until /generate moved off Vertex and left this id's
# bucket empty. A test pins them apart -- "tidying" them back together would
# silently halve the rate either one can sustain.
STYLE_MODEL = "gemini-3.1-flash-lite-image"
STYLE_TEMPERATURE = 0  # default for /stylize; lower = more faithful to the original photo

# Same ladder switch as IMAGE_MODEL above: point /stylize at another model by
# restarting the server rather than editing this file mid-experiment (an edit
# between runs is how a comparison ends up labelled with the wrong model).
# STYLE_MODEL=gemini-2.5-flash-image is the rollback.
_style_model_override = os.getenv("STYLE_MODEL", "").strip()
if _style_model_override:
    STYLE_MODEL = _style_model_override

# CNY per /stylize call, for the two ids this endpoint can be pointed at. Also
# fills the gap IMAGE_PRICE_CNY still has: nobody had looked up what
# gemini-3.1-flash-lite-image costs, and both pipelines can now cite it.
#
# Neither figure comes from Google. The pricing page truncated when fetched, so
# 2.5 is this repo's own conversion (1290 tokens at 30 USD/1M) and 3.1-lite is
# third-party aggregators. Re-check both against Google's page before quoting
# them in an argument -- an unverified number in a cost table still reads as
# authoritative.
STYLE_PRICES_CHECKED = "2026-09-25"
STYLE_PRICE_CNY = {
    "gemini-2.5-flash-image":      0.039 * USD_TO_CNY,   # repo conversion
    "gemini-3.1-flash-lite-image": 0.0336 * USD_TO_CNY,  # third-party
}

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

# Gemini bbox detector (hybrid disambiguator), for a /cutout with a prompt.
#
# This is a vision-in / text-out call -- it reads an image and answers with JSON
# -- so it is NOT restricted to the ids that can draw. The image-capable Gemini
# models stop at 3.1; a detector may go past that.
#
# Moved off gemini-2.5-flash on the date below. What decided it was not accuracy
# -- on a 5-image A/B the final cutouts were pixel-identical on 3, better on 1
# and worse on 1 -- but RELIABILITY OF THE ANSWER. Given the same bytes and the
# same temperature=0/seed=0, four calls to 2.5 returned three different boxes and
# varied the JSON schema between {"box":..}, {"box_2d":..} and {"x_min":..}; one
# call answered with malformed JSON outright. 3.1-flash-lite returned one box,
# in the requested schema, every time. It is also about 25% faster on the
# detector call (median 2190 ms -> 1644 ms).
#
# An unreadable answer is not loud: _parse returns None and main.py degrades to
# dino-only, serving a quieter cutout with nothing in the response to say the
# disambiguator dropped out.
#
# It was retested over all 17 cases on 2026-09-28. 17/17 answer 200 and 16/17
# take the healthy dino+union path. The one regression that A/B found --
# 06_parfait, where 3.1-lite boxed the glass and left the dessert above the rim
# outside it -- was the system prompt, not the model, and is fixed: the
# whole-object clause in cutout/bbox_detector.py restored 73% more of that
# subject while leaving the other 16 cutouts pixel-identical.
#
# 08_noodle_soup still comes back as a bare bowl rim. That one is NOT the
# detector: the box handed to SAM differs from the 2.5 run by about 5%
# ([152, 51, 995, 941] against [111, 0, 1000, 941]) and SAM returns a third as
# many pixels, so it is SAM's sensitivity on that image. Both arms were sampled
# once, and 2.5's box is not reproducible, so which model "passes" 08 is not
# established. Tracked separately, out of scope here.
#
# Evidence: test_output/bbox_model_ab/20260926_134013/ (the A/B),
# test_output/bbox_17_newprompt/ (all 17 on the shipped configuration)
#
# BBOX_DETECTOR_MODEL=gemini-2.5-flash is the rollback.
BBOX_DETECTOR_MODEL = "gemini-3.1-flash-lite"
BBOX_DETECTOR_CHECKED = "2026-09-26"   # date the id above was last seen answering

# Same ladder switch as IMAGE_MODEL and STYLE_MODEL above: point the bbox
# detector at another model by restarting the server rather than editing this
# line between runs, which is how a comparison ends up labelled with the wrong
# model. /health reports the value in effect.
_bbox_detector_override = os.getenv("BBOX_DETECTOR_MODEL", "").strip()
if _bbox_detector_override:
    BBOX_DETECTOR_MODEL = _bbox_detector_override

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
# ai_outline only. The other /border mode (outline) is pure Pillow and calls
# nothing, and the app only ever asks for that one -- ImageStudio.swift sends
# mode=outline and "ai_outline" appears nowhere in the iOS repo. So this model
# is not on any path a user reaches today.
#
# Left on 2.5 while STYLE_MODEL moved to 3.1: the two ids must differ to keep
# separate Vertex quota buckets (see STYLE_MODEL), and there is nothing to gain
# from re-verifying a model for a pipeline nobody calls.
BORDER_MODEL = "gemini-2.5-flash-image"  # Nano Banana img2img
BORDER_TEMPERATURE = 1.0  # ai_outline; higher = more creative decoration
BORDER_WIDTH = 12  # outline: default stroke width (px)
BORDER_COLOR = "#FFFFFF"  # outline: default stroke color
WHITE_EDGE_WIDTH = 8  # ai_outline: outermost die-cut white edge (px)
# ai_outline (outline-hugging AI border)
BORDER_BAND_RATIO = 0.06  # final (displayed) band width as a fraction of the subject's max side
BORDER_GUIDE_MIN_RATIO = 0.10  # min width of the wide guide band shown to the model (AI reliability floor); decoupled from the displayed width to avoid background halo at small band_ratio
BORDER_WORK_SIZE = 1024  # cap subject max side to this before processing (speed + smoothness)

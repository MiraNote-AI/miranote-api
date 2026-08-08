# Configuration for the image service (main.py).
# Grouped by the pipeline that uses each setting; a shared section holds the
# values used by more than one pipeline. Changing a value here changes behavior
# for every pipeline listed above its section.

# --------------------------------------------------------------------------- #
# Shared across pipelines
# --------------------------------------------------------------------------- #
MODEL_ID = "imagen-4.0-generate-001"   # Imagen 4: /generate output
FALLBACK_IMAGE_MODEL = "gemini-2.5-flash-image"  # /generate fallback when Imagen is project-gated
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


REMBG_MODEL = "birefnet-general"  # background removal: /generate (sticker) + /cutout. options: u2net, birefnet-general-lite, birefnet-general
REMBG_ERODE_RADIUS = 0  # pixels to erode alpha edge inward; 0 to disable. /generate + /cutout

# --------------------------------------------------------------------------- #
# /generate  (sticker & background image generation)
# --------------------------------------------------------------------------- #
NUMBER_OF_IMAGES = 2
ASPECT_RATIOS = {
    "sticker":    "1:1",
    "background": "9:16",
    "art":        "1:1",
}
REMOVE_BG = True  # set to False to skip background removal for testing

# --------------------------------------------------------------------------- #
# /stylize  (image-to-image style transfer)
# --------------------------------------------------------------------------- #
STYLE_MODEL = "gemini-2.5-flash-image"  # Nano Banana image-to-image; verify GA/preview id on Vertex
STYLE_TEMPERATURE = 0  # default for /stylize; lower = more faithful to the original photo

# --------------------------------------------------------------------------- #
# /cutout  (background removal + prompt-guided cutout)
# --------------------------------------------------------------------------- #
DEFAULT_PROMPT_CUTOUT_MODE = "hybrid_sam_prebg_gray"  # rembg -> gray bg -> hybrid_sam_union

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

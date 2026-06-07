MODEL_ID = "imagen-4.0-generate-001"
NUMBER_OF_IMAGES = 2
ASPECT_RATIOS = {
    "sticker":    "1:1",
    "background": "9:16",
}
OUTPUT_DIR = "output"
REMBG_MODEL = "birefnet-general"  # options: u2net, birefnet-general-lite, birefnet-general
REMBG_HUMAN_MODEL = "birefnet-general"
REMOVE_BG = True  # set to False to skip background removal for testing
REMBG_ERODE_RADIUS = 0  # pixels to erode alpha edge inward; set to 0 to disable
PROMPT_BUILDER_MODEL = "gemini-2.5-flash"

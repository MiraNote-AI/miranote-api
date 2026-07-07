import requests
import base64
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://localhost:8001"


def _bbox_norm_to_xyxy(bbox_norm, w, h):
    y_min, x_min, y_max, x_max = bbox_norm
    return [(x_min / 1000) * w, (y_min / 1000) * h, (x_max / 1000) * w, (y_max / 1000) * h]


def _draw_dashed_rectangle(draw, xyxy, outline, width, dash=12, gap=8):
    x1, y1, x2, y2 = xyxy
    def dashed_line(ax, ay, bx, by):
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length == 0:
            return
        ux, uy = (bx - ax) / length, (by - ay) / length
        pos = 0.0
        while pos < length:
            seg = min(dash, length - pos)
            sx, sy = ax + ux * pos, ay + uy * pos
            ex, ey = ax + ux * (pos + seg), ay + uy * (pos + seg)
            draw.line([sx, sy, ex, ey], fill=outline, width=width)
            pos += dash + gap
    dashed_line(x1, y1, x2, y1)
    dashed_line(x2, y1, x2, y2)
    dashed_line(x2, y2, x1, y2)
    dashed_line(x1, y2, x1, y1)


def _save_bbox_debug(image_path: str, chosen_bbox, prefix: str,
                     dino_bboxes=None, gemini_bbox=None, dino_scores=None, chosen_path=None):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    base = max(2, int(min(w, h) * 0.003))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(14, int(min(w, h) * 0.03)))
    except OSError:
        font = ImageFont.load_default()

    if dino_bboxes:
        for i, b in enumerate(dino_bboxes):
            x1, y1, x2, y2 = _bbox_norm_to_xyxy(b, w, h)
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=base)
            if dino_scores and i < len(dino_scores):
                draw.text((x1 + base, y1 + base), f"{dino_scores[i]:.2f}",
                          fill="yellow", font=font, stroke_width=2, stroke_fill="black")
    if chosen_bbox:
        cx1, cy1, cx2, cy2 = _bbox_norm_to_xyxy(chosen_bbox, w, h)
        draw.rectangle([cx1, cy1, cx2, cy2], outline="red", width=base * 3)
        # Label the final chosen box (e.g. "dino(iou=0.62, score=0.58)+union"); in union
        # mode the red box is synthetic, so this is the only place its provenance shows.
        if chosen_path:
            draw.text((cx1 + base, max(0, cy1 - int(min(w, h) * 0.04))), chosen_path,
                      fill="red", font=font, stroke_width=2, stroke_fill="black")
    # Gemini box drawn last as a dashed line so it stays visible even when it
    # coincides with the chosen (red) box — red shows through the gaps.
    if gemini_bbox:
        _draw_dashed_rectangle(draw, _bbox_norm_to_xyxy(gemini_bbox, w, h),
                               outline="blue", width=base * 2)

    path = Path(f"test_output/{prefix}_bbox.png")
    path.parent.mkdir(exist_ok=True)
    img.save(path)
    print(f"  Saved bbox debug {path}")

def save_images(data: dict, prefix: str):
    for i, img_b64 in enumerate(data["images"]):
        path = Path(f"test_output/{prefix}_{i}.png")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(base64.b64decode(img_b64))
        print(f"  Saved {path}")

def _test_generate(command: str, prompt: str, prefix: str):
    print(f"\nTesting generate ({command})...")
    start = time.time()
    resp = requests.post(f"{BASE_URL}/generate", json={"command": command, "prompt": prompt})
    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    print(f"  Prompt: {data['prompt']}")
    print(f"  Time: {elapsed:.1f}s")
    save_images(data, prefix)


def test_generate_sticker(prompt: str, prefix: str = "sticker"):
    _test_generate("sticker", prompt, prefix)


def test_generate_background(prompt: str, prefix: str = "background"):
    _test_generate("background", prompt, prefix)

def test_cutout(image_path: str, prompt: str = "", mode: str = "", prefix: str = "cutout"):
    label = "auto" + (f", prompt='{prompt}'" if prompt else "") + (f", mode={mode}" if mode else "")
    print(f"\nTesting cutout ({label})...")
    start = time.time()
    params = {}
    if prompt:
        params["prompt"] = prompt
    if mode:
        params["mode"] = mode
    with open(image_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/cutout", files={"file": f}, params=params)
    if resp.status_code in (400, 404):
        print(f"  {resp.status_code}: {resp.json().get('detail')}")
        return
    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    path = Path(f"test_output/{prefix}.png")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(base64.b64decode(data["image"]))
    print(f"  Time: {elapsed:.1f}s  mode_used={data.get('mode_used')}")
    print(f"  Saved {path}")
    bbox = data.get("bbox")
    if bbox:
        _save_bbox_debug(
            image_path, bbox, prefix,
            dino_bboxes=data.get("dino_bboxes"),
            gemini_bbox=data.get("gemini_bbox"),
            dino_scores=data.get("dino_scores"),
            chosen_path=data.get("chosen_path"),
        )


def test_stylize(image_path: str, style: str = "", prompt: str = "", temperature: float = None, prefix: str = "stylize"):
    label = (f"style={style}" if style else "") + (f", prompt='{prompt}'" if prompt else "") + (f", temp={temperature}" if temperature is not None else "")
    print(f"\nTesting stylize ({label})...")
    start = time.time()
    params = {}
    if style:
        params["style"] = style
    if prompt:
        params["prompt"] = prompt
    if temperature is not None:
        params["temperature"] = temperature
    with open(image_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/stylize", files={"file": f}, params=params)
    if resp.status_code in (400, 404):
        print(f"  {resp.status_code}: {resp.json().get('detail')}")
        return
    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    path = Path(f"test_output/{prefix}.png")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(base64.b64decode(data["image"]))
    print(f"  Time: {elapsed:.1f}s  style_used={data.get('style_used')}  temperature={data.get('temperature')}")
    print(f"  Saved {path}")


def test_border(image_path: str, mode: str = "outline", prefix: str = "border", **params):
    label = f"mode={mode}" + "".join(f", {k}={v}" for k, v in params.items())
    print(f"\nTesting border ({label})...")
    start = time.time()
    params = {"mode": mode, **params}
    with open(image_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/border", files={"file": f}, params=params)
    if resp.status_code in (400, 404):
        print(f"  {resp.status_code}: {resp.json().get('detail')}")
        return
    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    path = Path(f"test_output/{prefix}.png")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(base64.b64decode(data["image"]))
    print(f"  Time: {elapsed:.1f}s  mode_used={data.get('mode_used')}")
    print(f"  Saved {path}")


print("Health:", requests.get(f"{BASE_URL}/health").json())

# Each pipeline below has its own section. One line per sub-mode is left
# UNCOMMENTED so a bare run exercises every pipeline once; the rest are a
# commented catalog of extra examples. Sections run top-to-bottom, so /cutout
# comes before /border (it produces the 2cut.png that the border tests need).

# ==========================================================================
# /generate  --  Imagen 4 sticker & background generation.
# expand=True by default, so the short Chinese prompts also exercise LLM
# prompt expansion. Sub-modes: sticker (transparent) | background (9:16).
# ==========================================================================
# -- sticker: background auto-removed to a transparent PNG --
test_generate_sticker("棕色铅笔素描小象", "sticker_elephant")
# test_generate_sticker("生成一个可爱的草莓蛋糕贴纸，甜一点", "sticker_strawberry_cake")
# test_generate_sticker("小花花贴纸，手绘感，随性", "sticker_small_floral")
# test_generate_sticker("复古风的邮票贴纸，有点旧旧的纸张质感", "sticker_vintage_stamp")
# test_generate_sticker("马克杯贴纸，颜色淡淡的，不要太复杂", "sticker_ceramic_mug")
# test_generate_sticker("印象派style的湖中花花", "sticker_lake_flowers")

# -- background: keeps its background --
test_generate_background("生成复古陈旧的背景", "bg_vintage")
# test_generate_background("生成有手绘感颜色比较可爱带点随意感的背景", "bg_handdrawn")
# test_generate_background("生成有一条旅行轨迹（杭州-苏州）的深蓝色带点高级感的背景", "bg_travel")
# test_generate_background("生成带有胶片感，有年代感和轻微颗粒感的暖色背景，背景四周有胶片边框", "bg_film")
# test_generate_background("生成樱花粉色系、带一点日系手帐感的背景，背景底部有樱花瓣飘落", "bg_sakura")
# test_generate_background("生成森林绿色调、自然治愈的背景，四个角落有小树", "bg_forest")


# ==========================================================================
# /cutout  --  background removal + subject cutout.
# Sub-modes: auto (no prompt) | prompt-guided.
# ==========================================================================
# -- auto: no prompt -> full-image rembg background removal.
#    Also writes test_output/2cut.png, the transparent input the /border tests use. --
test_cutout("test_input/2.jpeg", prefix="2cut")
# test_cutout("test_input/1.jpeg", prefix="1cut")
# -- prompt-guided: default mode hybrid_sam_prebg_gray (rembg -> gray bg ->
#    hybrid_sam_union). Omit mode= for the default; pass mode="hybrid_sam_union"
#    to compare the no-prebg path. --
# test_cutout("test_input/13.jpeg", prompt="the yellow mango shaved ice on the right", prefix="prebg_mango")
# test_cutout("test_input/14.jpeg", prompt="the rabbit on the right", prefix="prebg_rabbit_right")
test_cutout("test_input/17.jpeg", prompt="the boy on the left", prefix="prebg_left_boy")
# test_cutout("test_input/11.jpeg", prompt="the girl in blue dress", mode="hybrid_sam_union", prefix="gsam_girl_union")


# ==========================================================================
# /stylize  --  image-to-image style transfer via Gemini 2.5 Flash Image.
# Sub-modes: preset style (key) | custom prompt (free-form).
# ==========================================================================
# -- preset style: key from style_presets --
# test_stylize("test_input/17.jpeg", style="impressionist", prefix="style_imp")
# test_stylize("test_input/17.jpeg", style="cartoon",       prefix="style_cartoon")
test_stylize("test_input/17.jpeg", style="sketch",        prefix="style_sketch")
# test_stylize("test_input/17.jpeg", style="line_art",      prefix="style_line")

# -- custom prompt: free-form style description --
test_stylize("test_input/17.jpeg", prompt="哥特风", prefix="style_custom")


# ==========================================================================
# /border  --  sticker frame; needs a transparent cutout PNG (uses 2cut.png
# produced by /cutout above). Sub-modes: outline (Pillow) | ai_outline (AI).
# ==========================================================================
# -- outline: die-cut solid outline, pure Pillow --
# test_border("test_output/2cut.png", mode="outline", color="#FF5C8A", width=24, prefix="border_outline_pink")
test_border("test_output/2cut.png", mode="outline", color="#FFFFFF", width=32, prefix="border_outline_white")

# -- ai_outline: outline-hugging AI border (shape via Pillow, pattern via AI, DIY prompt).
#    paste_back=False keeps the model's own subject (just drops the background). --
test_border("test_output/2cut.png", mode="ai_outline", prompt="白色的褶皱纸", band_ratio=0.03, paste_back=False, debug_dir=None, prefix="border_ai_outline_paper")
# test_border("test_output/2cut.png", mode="ai_outline", prompt="手绘小花边框", band_ratio=0.12, debug_dir="test_output/border_outline_debug", prefix="border_ai_outline_flower")
# test_border("test_output/2cut.png", mode="ai_outline", prompt="粉色底白色小波点", band_ratio=0.10, debug_dir="test_output/border_outline_debug", prefix="border_ai_outline_dots")
# test_border("test_output/2cut.png", mode="ai_outline", prompt="白色底小碎樱花", band_ratio=0.03, paste_back=False, debug_dir="test_output/outline_debug_2", prefix="border_ai_outline_sakura")
# test_border("test_output/2cut.png", mode="ai_outline", prompt="豹纹", band_ratio=0.03, paste_back=False, debug_dir="test_output/outline_debug_4", prefix="border_ai_outline_leopard")
# test_border("test_output/2cut.png", mode="ai_outline", prompt="陈旧的褶皱纸", band_ratio=0.08, paste_back=False, debug_dir="test_output/outline_debug_3", prefix="border_ai_outline_oldpaper")

print("\nDone. Open test_output/ to view images.")

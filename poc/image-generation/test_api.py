import requests
import base64
import time
from pathlib import Path

BASE_URL = "http://localhost:8001"

def save_images(data: dict, prefix: str):
    for i, img_b64 in enumerate(data["images"]):
        path = Path(f"test_output/{prefix}_{i}.png")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(base64.b64decode(img_b64))
        print(f"  Saved {path}")

def test(payload: dict, prefix: str):
    print(f"\nTesting {payload['command']}...")
    start = time.time()
    resp = requests.post(f"{BASE_URL}/generate", json=payload)
    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    print(f"  Prompt: {data['prompt']}")
    print(f"  Time: {elapsed:.1f}s")
    save_images(data, prefix)

def test_cutout(image_path: str, type: str = "general", prefix: str = "cutout"):
    print(f"\nTesting cutout ({type})...")
    start = time.time()
    with open(image_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/cutout", files={"file": f}, params={"type": type})
    resp.raise_for_status()
    elapsed = time.time() - start
    data = resp.json()
    path = Path(f"test_output/{prefix}.png")
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(base64.b64decode(data["image"]))
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved {path}")


print("Health:", requests.get(f"{BASE_URL}/health").json())

# BG_INSTRUCTION = "placed on a solid flat single-color background; the background color must not appear anywhere in the sticker itself and must strongly contrast with the sticker's palette to ensure clean background removal"
# STICKER_SUFFIX = f"for digital journaling, isolated object, clean edges, no text, no watermark, {BG_INSTRUCTION}"

test({"command": "sticker", "prompt": "A kawaii sticker of a small brown bear holding a coffee cup, cute and cozy, soft pastel colors, rounded shapes, simple flat illustration."}, "sticker_bear")
# test({"command": "sticker", "prompt": "A Korean aesthetic sticker of a ceramic mug with small flowers, clean and minimal, pastel blue and cream tones, soft and fresh illustration style."}, "sticker_mug")
# test({"command": "sticker", "prompt": "A Japanese journaling sticker of a cherry blossom bookmark, delicate and elegant, soft pink and beige palette, clean hand-drawn illustration."}, "sticker_bookmark")
# test({"command": "sticker", "prompt": "A vintage collage-style sticker of antique postage stamps and a small handwritten label, warm brown and muted green tones, paper texture look, nostalgic and elegant."}, "sticker_stamps")
# test({"command": "sticker", "prompt": "A hand-drawn doodle sticker of a bunch of tiny flowers, playful and casual, pastel color palette, sketchy line art, cute imperfect handmade look."}, "sticker_doodle")
# test({"command": "sticker", "prompt": "A pencil-sketch sticker of a brown elephant, hand-drawn pencil illustration, soft brown tones, sketchy shading, delicate linework, cute and gentle."}, "sticker_elephant")
# test({"command": "sticker", "prompt": "A gothic-style sticker of an ornate black rose with dark elegant details, gothic aesthetic, dramatic and refined, deep black and burgundy tones, intricate decorative linework."}, "sticker_gothic_rose")
# test({"command": "sticker", "prompt": "A Chinese ink-wash landscape sticker, traditional guofeng aesthetic, misty mountains, flowing river, pine trees, soft ink brush texture, elegant and poetic, muted earthy tones."}, "sticker_inkwash")
# test({"command": "sticker", "prompt": "生成一个可爱的草莓蛋糕贴纸，甜一点"}, "sticker_strawberry_cake")
# test({"command": "sticker", "prompt": "小花花贴纸，手绘感，随性"}, "sticker_small_floral")
# test({"command": "sticker", "prompt": "复古风的邮票贴纸，有点旧旧的纸张质感"}, "sticker_vintage_stamp")
# test({"command": "sticker", "prompt": "马克杯贴纸，颜色淡淡的，不要太复杂"}, "sticker_ceramic_mug")
# test({"command": "sticker", "prompt": "棕色铅笔素描小象"}, "sticker_elephant")
# test({"command": "sticker", "prompt": "印象派style的湖中花花"}, "sticker_lake_flowers")
test({"command": "background", "prompt": "生成复古陈旧的背景"}, "bg_vintage")
# test({"command": "background", "prompt": "生成有手绘感颜色比较可爱带点随意感的背景"}, "bg_handdrawn")
# test({"command": "background", "prompt": "生成有一条旅行轨迹（杭州-苏州）的深蓝色带点高级感的背景"}, "bg_travel")
# test({"command": "background", "prompt": "生成带有胶片感，有年代感和轻微颗粒感的暖色背景，背景四周有胶片边框"}, "bg_film")
# test({"command": "background", "prompt": "生成樱花粉色系、带一点日系手帐感的背景，背景底部有樱花瓣飘落"}, "bg_sakura")
# test({"command": "background", "prompt": "生成森林绿色调、自然治愈的背景，四个角落有小树"}, "bg_forest")


test_cutout("test_input/1.jpeg", type="person", prefix="1cutout_general")
# test_cutout("test_input/2.jpeg", type="general", prefix="2cutout_general")
# test_cutout("test_input/3.jpeg", type="person", prefix="3cutout_general")
# test_cutout("test_input/4.jpeg", type="general", prefix="4cutout_general")
# test_cutout("test_input/5.jpeg", type="general", prefix="5cutout_general")
# test_cutout("test_input/6.jpeg", type="person", prefix="6cutout_general")
# test_cutout("test_input/7.jpeg", type="general", prefix="7cutout_general")
# test_cutout("test_input/8.jpeg", type="general", prefix="8cutout_general")
# test_cutout("test_input/9.jpeg", type="person", prefix="9cutout_general")
# test_cutout("test_input/10.jpeg", type="person", prefix="10cutout_general")
# test_cutout("test_input/11.png", type="general", prefix="11cutout_general")

print("\nDone. Open test_output/ to view images.")

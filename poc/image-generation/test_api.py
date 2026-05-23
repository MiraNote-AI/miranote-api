import requests
import base64
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
    resp = requests.post(f"{BASE_URL}/generate", json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Prompt: {data['prompt']}")
    save_images(data, prefix)

print("Health:", requests.get(f"{BASE_URL}/health").json())

test({"command": "sticker", "subject": "cherry blossom", "palette": "pastel"}, "sticker")
test({"command": "background", "season": "spring", "mood": "cozy"}, "background")
test({"command": "divider", "style": "floral"}, "divider")


print("\nDone. Open test_output/ to view images.")

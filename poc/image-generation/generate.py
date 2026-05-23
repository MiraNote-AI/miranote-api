"""
MiraNote POC -- Image Generation CLI
Generates journal stickers, backgrounds, and dividers using Vertex AI Imagen 3.
"""

import json
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

import config
from prompts import journal_styles

load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
LOCATION = os.environ["LOCATION"]


def init_model() -> ImageGenerationModel:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return ImageGenerationModel.from_pretrained(config.MODEL_ID)


def save_images(images, output_dir: Path, base_name: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, image in enumerate(images):
        path = output_dir / f"{base_name}_{i}.png"
        image.save(str(path))
        saved_paths.append(str(path))
    return saved_paths


def update_manifest(manifest_path: Path, entry: dict) -> None:
    entries = []
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries.append(entry)
    manifest_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def _call_model(model: ImageGenerationModel, prompt: str, aspect_ratio: str):
    return model.generate_images(
        prompt=prompt,
        number_of_images=config.NUMBER_OF_IMAGES,
        aspect_ratio=aspect_ratio,
    )


def generate(model: ImageGenerationModel, prompt: str, category: str, base_name: str) -> list[str]:
    images = _call_model(model, prompt, config.ASPECT_RATIOS.get(category, "1:1"))

    output_dir = Path(config.OUTPUT_DIR) / category
    saved_paths = save_images(images, output_dir, base_name)

    entry = {
        "prompt": prompt,
        "category": category,
        "paths": saved_paths,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(config.OUTPUT_DIR) / "manifest.json"
    update_manifest(manifest_path, entry)

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate journal images with Imagen 3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sticker_cmd = subparsers.add_parser("sticker", help="Generate a sticker")
    sticker_cmd.add_argument("subject", help="e.g. 'cherry blossom'")
    sticker_cmd.add_argument("--palette", default="pastel")

    bg_cmd = subparsers.add_parser("background", help="Generate a page background")
    bg_cmd.add_argument("--season", default="spring")
    bg_cmd.add_argument("--mood", default="cozy")

    div_cmd = subparsers.add_parser("divider", help="Generate a decorative divider")
    div_cmd.add_argument("--style", default="floral")

    subparsers.add_parser("batch", help="Run the full sample batch from journal_styles.SAMPLE_BATCH")

    args = parser.parse_args()
    model = init_model()

    if args.command == "sticker":
        prompt = journal_styles.sticker(args.subject, args.palette)
        paths = generate(model, prompt, "sticker", args.subject.replace(" ", "_"))
    elif args.command == "background":
        prompt = journal_styles.background(args.season, args.mood)
        paths = generate(model, prompt, "background", f"{args.season}_{args.mood}")
    elif args.command == "divider":
        prompt = journal_styles.divider(args.style)
        paths = generate(model, prompt, "divider", args.style)
    elif args.command == "batch":
        paths = []
        for category, prompt in journal_styles.SAMPLE_BATCH:
            base_name = prompt.split(",")[0].replace(" ", "_")[:40]
            paths.extend(generate(model, prompt, category, base_name))

    print(f"Saved {len(paths)} image(s):")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

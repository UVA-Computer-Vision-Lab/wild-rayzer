#!/usr/bin/env python3
"""Center crop raw frames and resize to 256×256 for dynamic RE10K clips.

Usage:
    python scripts/center_crop_and_resize.py \
        ./data/dynamic_re10k/test/images/davis_rollerblade \
        ... (other scene directories)

Each scene directory must contain a ``raw`` subdirectory with the original
frames. The script writes processed PNGs (256×256) beside the ``raw`` folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

TARGET_SIZE = 256


def center_crop_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


def process_scene(scene_dir: Path, overwrite: bool) -> None:
    raw_dir = scene_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Expected 'raw' folder inside {scene_dir}")

    output_dir = scene_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p
        for p in raw_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not image_paths:
        print(f"No raw images found in {raw_dir}")
        return

    for raw_path in image_paths:
        with Image.open(raw_path) as img:
            img = img.convert("RGB")
            cropped = center_crop_square(img)
            resized = cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        output_path = output_dir / f"{raw_path.stem}.png"
        if output_path.exists() and not overwrite:
            print(f"Skip existing file {output_path}")
            continue

        resized.save(output_path)
        print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Center crop + resize raw frames to 256×256")
    parser.add_argument("scenes", nargs="+", type=Path, help="Scene directories to process")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed images",
    )
    args = parser.parse_args()

    for scene in args.scenes:
        if not scene.exists():
            print(f"Scene directory does not exist: {scene}")
            continue
        process_scene(scene, overwrite=args.overwrite)


if __name__ == "__main__":
    main()


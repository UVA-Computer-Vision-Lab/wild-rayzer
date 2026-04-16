"""
Annotate Dynamic RE10K Test Set with Human Masks using YOLOv11

This script uses the trained YOLO segmentation model to extract binary human masks.
It keeps all detection classes but only outputs the 'person' class (ID=0) as binary masks.

Usage:
    python annotate_dynamic_re10k.py
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from ultralytics import YOLO


# Configuration — override via env vars WILDRAYZER_YOLO_MODEL / WILDRAYZER_DATA_ROOT
YOLO_MODEL_PATH = Path(os.environ.get("WILDRAYZER_YOLO_MODEL", "./pretrained_weights/yolo_person.pt"))
DATA_ROOT = Path(os.environ.get("WILDRAYZER_DATA_ROOT", "./data/dynamic_re10k/test"))

METADATA_DIR = DATA_ROOT / "metadata"
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR = DATA_ROOT / "gt_masks"
BACKUP_DIR = DATA_ROOT / "metadata_backup"

PERSON_CLASS_ID = 0  # 'person' class in COCO dataset


def segment_human_yolo(model, img_path, original_size=None):
    """
    Segment human from image using YOLO.

    Args:
        model: YOLO model
        img_path: Path to input image
        original_size: (width, height) tuple for output size, or None to match input

    Returns:
        mask_binary: Binary mask (uint8), 1=human, 0=background
    """
    # Load image
    img = Image.open(img_path).convert("RGB")
    if original_size is None:
        original_size = img.size

    # Run inference
    results = model(img, verbose=False)[0]

    # Initialize empty mask
    h, w = results.orig_shape
    mask_combined = np.zeros((h, w), dtype=np.uint8)

    # Extract masks for 'person' class only
    if results.masks is not None:
        for i, cls_id in enumerate(results.boxes.cls):
            if int(cls_id) == PERSON_CLASS_ID:
                # Get mask for this person instance
                mask = results.masks.data[i].cpu().numpy()
                # Resize to original image size if needed
                if mask.shape != (h, w):
                    mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_resized = mask_resized.resize((w, h), Image.NEAREST)
                    mask = (np.array(mask_resized) > 127).astype(np.uint8)
                else:
                    mask = (mask > 0.5).astype(np.uint8)
                # Combine all person instances
                mask_combined = np.maximum(mask_combined, mask)

    # Resize to target size if needed
    if mask_combined.shape[:2] != (original_size[1], original_size[0]):
        mask_pil = Image.fromarray(mask_combined * 255)
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
        mask_combined = (np.array(mask_pil) > 127).astype(np.uint8)

    return mask_combined


def check_scene_processed(metadata_path):
    """Check if scene already has human_mask_path in metadata."""
    with open(metadata_path, "r") as f:
        data = json.load(f)

    if not data.get("frames"):
        return False

    # Check if first frame has human_mask_path
    return "human_mask_path" in data["frames"][0]


def main():
    print("=" * 80)
    print("Starting Dynamic RE10K Test Set Annotation with YOLOv11")
    print("=" * 80)

    # Load YOLO model
    print("\n[0/5] Loading YOLO model...")
    model = YOLO(str(YOLO_MODEL_PATH))
    print(f"✓ Model loaded: {YOLO_MODEL_PATH}")
    print(f"✓ Classes: {len(model.names)} categories")
    print(f"✓ Person class ID: {PERSON_CLASS_ID}")

    # Step 1: Backup metadata (if not already backed up)
    if not BACKUP_DIR.exists():
        print("\n[1/5] Backing up metadata...")
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for json_file in METADATA_DIR.glob("*.json"):
            shutil.copy(json_file, BACKUP_DIR / json_file.name)
        num_backed_up = len(list(BACKUP_DIR.glob("*.json")))
        print(f"✓ Backed up {num_backed_up} metadata files to {BACKUP_DIR}")
    else:
        print(f"\n[1/5] ✓ Metadata backup already exists at {BACKUP_DIR}")

    # Step 2: Discover all scenes
    print("\n[2/5] Discovering scenes...")
    all_scenes = sorted(METADATA_DIR.glob("*.json"))
    print(f"✓ Found {len(all_scenes)} scenes")

    # Step 3: Filter out already processed scenes
    print("\n[3/5] Checking processed scenes...")
    scenes_to_process = []
    scenes_skipped = []
    for scene_path in all_scenes:
        if check_scene_processed(scene_path):
            scenes_skipped.append(scene_path.stem)
        else:
            scenes_to_process.append(scene_path)

    print(f"✓ {len(scenes_to_process)} scenes to process")
    print(f"✓ {len(scenes_skipped)} scenes already processed (skipped)")

    if len(scenes_to_process) == 0:
        print("\n✅ All scenes already processed!")
        return

    # Step 4: Process each scene
    print("\n[4/5] Processing scenes...")

    total_frames = 0
    total_with_human = 0

    for scene_path in tqdm(scenes_to_process, desc="Scenes"):
        # Load metadata
        with open(scene_path, "r") as f:
            metadata = json.load(f)

        scene_name = metadata["scene_name"]

        # Create output directory
        scene_mask_dir = MASKS_DIR / scene_name
        scene_mask_dir.mkdir(parents=True, exist_ok=True)

        # Process each frame
        for frame_data in tqdm(metadata["frames"], desc=f"  {scene_name}", leave=False):
            # Get image path (already absolute in metadata)
            img_path = Path(frame_data["image_path"])

            # Get original size
            with Image.open(img_path) as img:
                original_size = img.size

            # Segment human using YOLO
            mask_binary = segment_human_yolo(model, img_path, original_size)

            # Save PNG (255=human, 0=background)
            frame_name = Path(frame_data["image_path"]).stem
            mask_png_path = scene_mask_dir / f"{frame_name}.png"
            Image.fromarray(mask_binary * 255).save(mask_png_path)

            # Save NPY (1=human, 0=background)
            mask_npy_path = scene_mask_dir / f"{frame_name}.npy"
            np.save(mask_npy_path, mask_binary)

            # Update metadata with absolute paths (matching the format of image_path)
            frame_data["human_mask_path"] = str(mask_png_path)
            frame_data["human_mask_npy_path"] = str(mask_npy_path)

            # Statistics
            total_frames += 1
            if mask_binary.sum() > 0:
                total_with_human += 1

        # Save updated metadata (overwrite)
        with open(scene_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # Step 5: Summary
    print("\n[5/5] Summary")
    print("=" * 80)
    print(f"✓ Processed {len(scenes_to_process)} scenes")
    print(f"✓ Total frames: {total_frames}")
    human_pct = total_with_human / total_frames * 100 if total_frames > 0 else 0
    print(f"✓ Frames with humans: {total_with_human} ({human_pct:.1f}%)")
    print(f"✓ Masks saved to: {MASKS_DIR}")
    print(f"✓ Metadata updated in: {METADATA_DIR}")
    print(f"✓ Backup available at: {BACKUP_DIR}")
    print("=" * 80)
    print("✅ All done!")


if __name__ == "__main__":
    main()

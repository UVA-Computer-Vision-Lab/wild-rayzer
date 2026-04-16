#!/usr/bin/env python3
"""
Generate and save augmented sequences for visualization.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
import json
from pathlib import Path


def save_sequence(batch, output_dir, seq_idx):
    """Save a sequence with visualizations."""

    seq_dir = output_dir / f"sequence_{seq_idx:03d}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "scene_name": batch["scene_name"],
        "num_views": batch["image"].shape[0],
        "image_shape": list(batch["image"].shape),
    }

    masks = batch.get("copy_paste_mask")
    if masks is not None:
        mask_np = masks.cpu().numpy()  # (num_views, 1, H, W)
        metadata["has_copy_paste_mask"] = True
    else:
        mask_np = None
        metadata["has_copy_paste_mask"] = False

    with open(seq_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save images
    images = batch["image"]  # (num_views, 3, H, W)
    num_views = images.shape[0]

    # Individual images
    for view_idx in range(num_views):
        img = images[view_idx].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img = (img * 255).astype(np.uint8)

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"View {view_idx}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(seq_dir / f"view_{view_idx:02d}.png", dpi=150, bbox_inches="tight")
        plt.close()

        if mask_np is not None:
            mask_img = (mask_np[view_idx, 0] > 0).astype(np.uint8) * 255
            Image.fromarray(mask_img, mode="L").save(seq_dir / f"mask_{view_idx:02d}.png")

    # Grid visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for view_idx in range(min(num_views, 8)):
        img = images[view_idx].permute(1, 2, 0).cpu().numpy()
        axes[view_idx].imshow(img)
        axes[view_idx].set_title(f"View {view_idx}")
        axes[view_idx].axis("off")

    # Hide unused subplots
    for view_idx in range(num_views, 8):
        axes[view_idx].axis("off")

    plt.suptitle(f"Sequence {seq_idx}: {batch['scene_name']}", fontsize=16)
    plt.tight_layout()
    plt.savefig(seq_dir / "grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    if mask_np is not None:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for view_idx in range(min(num_views, 8)):
            mask_img = mask_np[view_idx, 0]
            axes[view_idx].imshow(mask_img, cmap="gray", vmin=0, vmax=1)
            axes[view_idx].set_title(f"Mask {view_idx}")
            axes[view_idx].axis("off")

        for view_idx in range(num_views, 8):
            axes[view_idx].axis("off")

        plt.suptitle(f"Overlay Masks {seq_idx}: {batch['scene_name']}", fontsize=16)
        plt.tight_layout()
        plt.savefig(seq_dir / "mask_grid.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"✓ Saved sequence {seq_idx} to {seq_dir}")


def main():
    print("=" * 80)
    print("Generate Augmented Sequences")
    print("=" * 80)

    # Load config
    config_path = "./configs/wildrayzer_stage3_joint_copy_paste.yaml"
    print(f"\n1. Loading config from {config_path}...")

    config = OmegaConf.load(config_path)

    # Override for testing
    config.training.copy_paste.paste_probability = 1.0  # Always apply
    config.training.copy_paste.generate_overlay_masks = True
    if "per_view_objects_prob" not in config.training.copy_paste:
        config.training.copy_paste.per_view_objects_prob = 0.5
    config.inference.if_inference = False

    print(f"   ✓ Config loaded")
    print(f"   Copy-paste enabled: {config.training.copy_paste.enabled}")
    print(f"   Paste probability: {config.training.copy_paste.paste_probability}")

    # Create dataset
    print("\n2. Creating dataset...")
    try:
        from data.dataset_mixed_re10k_official import Dataset

        dataset = Dataset(config)
        print(f"   ✓ Dataset created with {len(dataset)} sequences")

        # Check if copy-paste is enabled in child datasets
        if hasattr(dataset, "static_dataset"):
            print(f"\n   Checking copy-paste status:")
            print(f"   - Static dataset has copy-paste: {dataset.static_dataset.use_copy_paste}")
            if dataset.static_dataset.use_copy_paste:
                print(f"     ✓ Copy-paste extractor initialized")
            else:
                print(f"     ✗ Copy-paste NOT initialized!")

            if hasattr(dataset, "dynamic_dataset"):
                print(
                    f"   - Dynamic dataset has copy-paste: {dataset.dynamic_dataset.use_copy_paste}"
                )

    except Exception as e:
        print(f"   ✗ Failed to create dataset: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create output directory
    output_dir = Path("./experiments/augmented_sequences")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n3. Output directory: {output_dir}")

    # Generate sequences
    print("\n4. Generating 10 augmented sequences...")

    num_sequences = 10
    for i in range(num_sequences):
        try:
            # Load sequence
            batch = dataset[i]

            # Save
            save_sequence(batch, output_dir, i)

        except Exception as e:
            print(f"   ✗ Failed to generate sequence {i}: {e}")
            import traceback

            traceback.print_exc()

    # Create summary
    print("\n5. Creating summary...")
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Augmented Sequences Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total sequences: {num_sequences}\n")
        f.write(f"Copy-paste enabled: {config.training.copy_paste.enabled}\n")
        f.write(f"Paste probability: {config.training.copy_paste.paste_probability}\n")
        f.write(f"Animals per scene: {config.training.copy_paste.num_animals_per_scene}\n")
        f.write(f"Animal categories: {config.training.copy_paste.animal_categories}\n")
        f.write(f"Scale range: {config.training.copy_paste.scale_range}\n")
        f.write(
            f"Per-view objects prob: {config.training.copy_paste.get('per_view_objects_prob', 0.0)}\n"
        )
        f.write(
            f"Generate overlay masks: {config.training.copy_paste.get('generate_overlay_masks', False)}\n"
        )
        f.write("\nSequences:\n")

        for i in range(num_sequences):
            seq_dir = output_dir / f"sequence_{i:03d}"
            if (seq_dir / "metadata.json").exists():
                with open(seq_dir / "metadata.json", "r") as mf:
                    metadata = json.load(mf)
                f.write(
                    f"  {i:03d}: {metadata['scene_name']} "
                    f"({metadata['num_views']} views, mask={metadata.get('has_copy_paste_mask', False)})\n"
                )

    print(f"   ✓ Summary saved to {summary_path}")

    print("\n" + "=" * 80)
    print("✓ Generation complete!")
    print("=" * 80)
    print(f"\nOutput location: {output_dir}")
    print("\nGenerated files:")
    print("  - sequence_XXX/")
    print("    - grid.png          (all views in one image)")
    print("    - view_XX.png       (individual view images)")
    print("    - mask_XX.png       (binary overlay mask per view, if enabled)")
    print("    - mask_grid.png     (overlay masks in a grid, if enabled)")
    print("    - metadata.json     (sequence metadata)")
    print("  - summary.txt         (overall summary)")
    print("=" * 80)


if __name__ == "__main__":
    main()

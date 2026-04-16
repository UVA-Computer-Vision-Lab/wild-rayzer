"""
RE10K Mixed Dataset (Official Version)

Combines Dynamic RE10K (dynamic scenes with learned masks) and
Original RE10K (static scenes with zero masks) for balanced training.

This version uses dataset_scene_official.Dataset for deterministic frame sampling
compatible with the official RAYZAR model.

Default: 50% Dynamic RE10K + 50% Static RE10K
"""

import random
from pathlib import Path
from typing import Dict
from easydict import EasyDict as edict
import torch
from torch.utils.data import Dataset as TorchDataset

try:
    from .dataset_scene_official import Dataset as RE10KDataset
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dataset_scene_official import Dataset as RE10KDataset


class RE10KMixedDataset(TorchDataset):
    """
    Mixed dataset that combines:
    1. Dynamic RE10K - scenes with motion (learns masks via DINOv3)
    2. Static RE10K - original static multi-view scenes (zero masks)

    Samples are drawn with a configurable ratio to balance training.

    Uses dataset_scene_official.Dataset for deterministic frame sampling.
    """

    def __init__(
        self,
        dynamic_config,  # Config for dynamic RE10K
        static_config,  # Config for static RE10K
        mix_ratio: float = 0.5,  # Ratio of dynamic samples (0.5 = 50% dynamic, 50% static)
    ):
        """
        Args:
            dynamic_config: Config object for Dynamic RE10K dataset
            static_config: Config object for Static RE10K dataset
            mix_ratio: Ratio of dynamic samples in the mix (0.0 to 1.0)
        """
        self.mix_ratio = float(mix_ratio)
        assert 0.0 <= self.mix_ratio <= 1.0, "mix_ratio must be between 0 and 1"

        print("=" * 80)
        print("Initializing RE10K Mixed Dataset (Official Version)")
        print("=" * 80)

        # Initialize Dynamic RE10K dataset with dynamic view selector
        print("\n[1/2] Loading Dynamic RE10K dataset (dynamic scenes)...")
        dyn_min = dynamic_config.training.view_selector.min_frame_dist
        dyn_max = dynamic_config.training.view_selector.max_frame_dist
        print(f"  - View selector: min={dyn_min}, max={dyn_max}")
        print(f"  - Scene scale factor: {dynamic_config.training.scene_scale_factor}")
        self.dynamic_dataset = RE10KDataset(dynamic_config)

        # Initialize Static RE10K dataset with static view selector
        print("\n[2/2] Loading Static RE10K dataset (static scenes)...")
        static_min = static_config.training.view_selector.min_frame_dist
        static_max = static_config.training.view_selector.max_frame_dist
        print(f"  - View selector: min={static_min}, max={static_max}")
        print(f"  - Scene scale factor: {static_config.training.scene_scale_factor}")
        self.static_dataset = RE10KDataset(static_config)

        # Compute dataset sizes
        self.dynamic_size = len(self.dynamic_dataset)
        self.static_size = len(self.static_dataset)

        # Total size
        self.total_size = self.dynamic_size + self.static_size

        print("\n" + "=" * 80)
        print("Mixed Dataset Summary (Official Version)")
        print("=" * 80)
        print(f"Dynamic scenes: {self.dynamic_size:5d} ({mix_ratio * 100:.1f}% target)")
        print(f"Static scenes:  {self.static_size:5d} ({(1 - mix_ratio) * 100:.1f}% target)")
        print(f"Total size:     {self.total_size:5d}")
        print("Using dataset_scene_official.Dataset (deterministic frame sampling)")
        print("=" * 80)

    def update_step(self, step):
        """Forward step update to child datasets for deterministic sampling."""
        if hasattr(self.dynamic_dataset, "update_step"):
            self.dynamic_dataset.update_step(step)
        if hasattr(self.static_dataset, "update_step"):
            self.static_dataset.update_step(step)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a sample from either Dynamic or Static dataset based on mix_ratio.

        IMPORTANT: Uses deterministic sampling based on idx for cache consistency!

        Returns:
            dict with keys:
                - image: [V, 3, H, W]
                - index: [V, 3] (frame_idx, time, scene_idx)
                - dataset_source: "dre10k" or "re10k"
                - ... (other dataset-specific keys)
        """
        # DETERMINISTIC: Use idx to decide which dataset and which sample
        # This ensures same idx always returns the same sample (critical for mask caching!)

        # Split total dataset into dynamic and static regions based on mix_ratio
        num_dynamic_slots = int(self.total_size * self.mix_ratio)

        if idx < num_dynamic_slots:
            # Sample from Dynamic RE10K dataset (deterministic mapping)
            dynamic_idx = idx % self.dynamic_size
            sample = self.dynamic_dataset[dynamic_idx]
            # Add dataset source identifier (string, will be collated into list by DataLoader)
            sample["dataset_source"] = "dre10k"  # Dynamic scenes

            # Ensure copy_paste_mask exists (for collation compatibility with static dataset)
            if "copy_paste_mask" not in sample:
                # Create zero mask matching image shape
                if "image" in sample:
                    v, _, h, w = sample["image"].shape
                    sample["copy_paste_mask"] = torch.zeros((v, 1, h, w), dtype=torch.float32)

            # Ensure image_clean exists (for collation compatibility)
            if "image_clean" not in sample:
                # No augmentation, so clean image = original image
                sample["image_clean"] = sample["image"].clone()
        else:
            # Sample from Static RE10K dataset (deterministic mapping)
            static_idx = (idx - num_dynamic_slots) % self.static_size
            sample = self.static_dataset[static_idx]
            # Add dataset source identifier (string, will be collated into list by DataLoader)
            sample["dataset_source"] = "re10k"  # Static scenes (zero masks)

            # Ensure copy_paste_mask exists (for collation compatibility)
            if "copy_paste_mask" not in sample:
                # Create zero mask matching image shape
                if "image" in sample:
                    v, _, h, w = sample["image"].shape
                    sample["copy_paste_mask"] = torch.zeros((v, 1, h, w), dtype=torch.float32)

            # Ensure image_clean exists (for collation compatibility)
            if "image_clean" not in sample:
                # No augmentation, so clean image = original image
                sample["image_clean"] = sample["image"].clone()

        return sample


# Adapter class for config-based instantiation
class Dataset:
    """
    Adapter class that matches the expected API for the training pipeline.
    Uses dataset_scene_official.Dataset for deterministic frame sampling.
    """

    def __new__(cls, config):
        """
        Create a RE10KMixedDataset from config.

        Expected config keys:
            - training.dynamic_re10k_path: Path to Dynamic RE10K scene list
            - training.static_re10k_path: Path to Static RE10K scene list
            - training.dataset_mix_ratio: Mix ratio (0.0 to 1.0, default 0.5)
            - training.num_input_views: Number of input views
            - training.num_target_views: Number of target views
            - training.view_selector_dynamic: View selector config for dynamic RE10K
            - training.view_selector_static: View selector config for static RE10K (optional)
            - model.image_tokenizer.image_size: Image size
            - model.image_tokenizer.patch_size: Patch size
        """

        # Extract parameters from config
        total_views = config.training.num_input_views + config.training.num_target_views

        # Get view_selector configs for dynamic and static datasets
        view_selector_dynamic_cfg = config.training.get("view_selector_dynamic", None)
        if view_selector_dynamic_cfg is None:
            view_selector_dynamic_cfg = edict(
                {
                    "type": "two_frame",
                    "min_frame_dist": 8,
                    "max_frame_dist": 12,
                }
            )

        view_selector_static_cfg = config.training.get("view_selector_static", None)
        if view_selector_static_cfg is None:
            # Use the regular view_selector for static if not specified
            view_selector_static_cfg = config.training.get("view_selector", None)
            if view_selector_static_cfg is None:
                view_selector_static_cfg = edict(
                    {
                        "type": "two_frame",
                        "min_frame_dist": 25,
                        "max_frame_dist": 192,
                    }
                )

        # Get scale factors (allow separate configs or use defaults)
        dynamic_scale_factor = config.training.get(
            "dynamic_scene_scale_factor", config.training.get("scene_scale_factor", 1.0)
        )
        static_scale_factor = config.training.get("static_scene_scale_factor", 1.35)

        # Create Dynamic RE10K config
        dynamic_config = edict(
            {
                "training": {
                    "dataset_path": config.training.dynamic_re10k_path,
                    "num_views": total_views,
                    "num_input_views": config.training.num_input_views,
                    "num_target_views": config.training.num_target_views,
                    "target_has_input": config.training.get("target_has_input", False),
                    "random_split": config.training.get("random_split", True),
                    "square_crop": config.training.get("square_crop", True),
                    "scene_scale_factor": dynamic_scale_factor,
                    "view_selector": view_selector_dynamic_cfg,  # Use dynamic view selector
                    "deterministic_views": config.training.get("deterministic_views", False),
                    "deterministic_views_refresh_every": config.training.get(
                        "deterministic_views_refresh_every", 3000
                    ),
                },
                "model": {
                    "image_tokenizer": {
                        "image_size": config.model.image_tokenizer.image_size,
                        "patch_size": config.model.image_tokenizer.patch_size,
                    }
                },
                "inference": {
                    "if_inference": False,
                },
            }
        )

        # Create Static RE10K config (same as dynamic but different path and scale)
        static_config = edict(
            {
                "training": {
                    "dataset_path": config.training.static_re10k_path,
                    "num_views": total_views,
                    "num_input_views": config.training.num_input_views,
                    "num_target_views": config.training.num_target_views,
                    "target_has_input": config.training.get("target_has_input", False),
                    "random_split": config.training.get("random_split", True),
                    "square_crop": config.training.get("square_crop", True),
                    "scene_scale_factor": static_scale_factor,  # Different scale for static RE10K
                    "view_selector": view_selector_static_cfg,  # Use static view selector
                    "deterministic_views": config.training.get("deterministic_views", False),
                    "deterministic_views_refresh_every": config.training.get(
                        "deterministic_views_refresh_every", 3000
                    ),
                },
                "model": {
                    "image_tokenizer": {
                        "image_size": config.model.image_tokenizer.image_size,
                        "patch_size": config.model.image_tokenizer.patch_size,
                    }
                },
                "inference": {
                    "if_inference": False,
                },
            }
        )

        # Pass copy_paste config if present
        if "copy_paste" in config.training:
            static_config.training["copy_paste"] = config.training.copy_paste

        return RE10KMixedDataset(
            dynamic_config=dynamic_config,
            static_config=static_config,
            mix_ratio=config.training.get("dataset_mix_ratio", 0.5),
        )

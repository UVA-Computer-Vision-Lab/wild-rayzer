# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F

# Copy-paste augmentation
try:
    from data.copy_paste_utils import COCOAnimalExtractor, apply_copy_paste_to_views

    COPY_PASTE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Copy-paste augmentation not available: {e}")
    COPY_PASTE_AVAILABLE = False


class Dataset(Dataset):  # Original PNG/JSON dataset
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            with open(self.config.training.dataset_path, "r") as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]

        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e

        # Deterministic view selection for cache-friendly training
        self.deterministic_views = self.config.training.get("deterministic_views", False)
        self.deterministic_views_refresh_every = self.config.training.get(
            "deterministic_views_refresh_every",
            3000,
        )
        self.current_step = 0  # Will be updated by trainer
        if self.deterministic_views:
            print(
                "✓ Deterministic view selection enabled "
                f"(refresh every {self.deterministic_views_refresh_every} steps)"
            )

        # Initialize copy-paste augmentation
        self.use_copy_paste = False
        self.copy_paste_extractor = None
        if not self.config.inference.get("if_inference", False):  # Only for training
            copy_paste_config = self.config.training.get("copy_paste", None)
            if copy_paste_config is not None and copy_paste_config.get("enabled", False):
                if COPY_PASTE_AVAILABLE:
                    try:
                        print("[Dataset] Initializing copy-paste augmentation...")
                        default_animals = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
                        self.copy_paste_extractor = COCOAnimalExtractor(
                            coco_root=copy_paste_config["coco_root"],
                            coco_ann_file=copy_paste_config["coco_ann_file"],
                            animal_categories=set(
                                copy_paste_config.get("animal_categories", default_animals)
                            ),
                        )
                        self.use_copy_paste = True
                        self.copy_paste_config = copy_paste_config
                        prob = copy_paste_config.get("paste_probability", 0.5)
                        print(f"✓ Copy-paste augmentation enabled (probability={prob})")
                    except Exception as e:
                        print(f"⚠️  Failed to initialize copy-paste augmentation: {e}")
                        self.use_copy_paste = False
                else:
                    print("⚠️  Copy-paste augmentation requested but not available")

        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, "r") as f:
                        self.view_idx_list = json.load(f)
                        # Filter out scenes without specified input/target view indices
                        self.view_idx_list_filtered = [
                            k for k, v in self.view_idx_list.items() if v is not None
                        ]

                    # Expand scene paths to include splits
                    # For new naming: metadata path already includes full split name
                    # e.g., wildrayzer_computer_context_2_split_0.json
                    expanded_scene_paths = []
                    for scene_path in self.all_scene_paths:
                        file_name = scene_path.split("/")[-1]
                        scene_name_from_file = file_name.split(".")[
                            0
                        ]  # e.g., wildrayzer_computer_context_2_split_0

                        # Check if this exact scene name is in view_idx_list
                        if scene_name_from_file in self.view_idx_list_filtered:
                            # Direct match - use the scene name from the file
                            expanded_scene_paths.append((scene_path, scene_name_from_file))

                    self.all_scene_paths = expanded_scene_paths
                    if len(expanded_scene_paths) == 0:
                        print("⚠️  WARNING: Found 0 matches between manifest and view_idx!")
                        print(f"   Manifest scenes: {len(self.all_scene_paths)} total")
                        print(f"   View idx keys: {len(self.view_idx_list_filtered)} total")
                        if len(self.all_scene_paths) > 0:
                            sample_scene = self.all_scene_paths[0].split("/")[-1].split(".")[0]
                            print(f"   Sample manifest scene: {sample_scene}")
                        if len(self.view_idx_list_filtered) > 0:
                            print(f"   Sample view_idx key: {self.view_idx_list_filtered[0]}")
                    print(f"✓ Expanded {len(expanded_scene_paths)} sequences from view_idx")
                else:
                    # No view_idx, keep original format (just paths)
                    self.all_scene_paths = [(path, None) for path in self.all_scene_paths]
            else:
                # No view_idx, keep original format
                self.all_scene_paths = [(path, None) for path in self.all_scene_paths]
        else:
            # Training mode: keep original format
            self.all_scene_paths = [(path, None) for path in self.all_scene_paths]

    def __len__(self):
        return len(self.all_scene_paths)

    def update_step(self, step):
        """Update current training step for deterministic view selection."""
        self.current_step = step

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        # Use rgb_tokenizer if available, otherwise fall back to image_tokenizer for compatibility
        tokenizer_config = getattr(
            self.config.model, "rgb_tokenizer", getattr(self.config.model, "image_tokenizer", None)
        )
        if tokenizer_config is None:
            raise AttributeError(
                "Config must have either 'rgb_tokenizer' or 'image_tokenizer' section"
            )

        resize_h = tokenizer_config.image_size
        patch_size = tokenizer_config.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size

            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)

        # Handle different pose formats: w2c, transform_matrix (c2w), or dummy identity
        c2ws_list = []
        for frame in frames_chosen:
            if "w2c" in frame:
                w2c = np.array(frame["w2c"])
                c2w = np.linalg.inv(w2c)
            elif "transform_matrix" in frame:
                # transform_matrix is typically c2w in NeRF-style datasets
                c2w = np.array(frame["transform_matrix"])
            else:
                # Fallback to identity matrix (dummy poses for evaluation)
                c2w = np.eye(4)
            c2ws_list.append(c2w)

        c2ws = np.stack(c2ws_list)  # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # Align coordinate system (OpenCV) to the mean camera center and orientation.
        # Average direction vectors are computed from all cameras (average down and forward).
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(
            in_c2ws[:, :3, 2].mean(0), dim=-1
        )  # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0)  # average down direction (y of opencv camera)
        avg_right = F.normalize(
            torch.cross(avg_down, avg_forward, dim=-1), dim=-1
        )  # (x of opencv camera)
        avg_down = F.normalize(
            torch.cross(avg_forward, avg_right, dim=-1), dim=-1
        )  # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device)  # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose)  # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames, scene_idx=None, scene_path=None):
        if len(frames) < self.config.training.num_views:
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            return None

        # Deterministic sampling for cache-friendly training
        rng = None
        if self.deterministic_views and scene_path is not None:
            # Seed based on scene_path + step_group (refreshes every N steps)
            import hashlib

            step_group = self.current_step // self.deterministic_views_refresh_every
            seed_str = f"{scene_path}_{step_group}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
        elif scene_idx is not None and not self.config.training.get("random_split", False):
            # Fallback: Use scene_idx as seed (original behavior)
            rng = random.Random(scene_idx)

        if rng is not None:
            # Deterministic sampling
            frame_dist = rng.randint(min_frame_dist, max_frame_dist)
            if len(frames) <= frame_dist:
                return None
            start_frame = rng.randint(0, len(frames) - frame_dist - 1)
            end_frame = start_frame + frame_dist
            sampled_frames = rng.sample(
                range(start_frame + 1, end_frame), self.config.training.num_views - 2
            )
        else:
            # Random sampling (original behavior)
            frame_dist = random.randint(min_frame_dist, max_frame_dist)
            if len(frames) <= frame_dist:
                return None
            start_frame = random.randint(0, len(frames) - frame_dist - 1)
            end_frame = start_frame + frame_dist
            sampled_frames = random.sample(
                range(start_frame + 1, end_frame), self.config.training.num_views - 2
            )

        image_indices = [start_frame, end_frame] + sampled_frames
        return image_indices

    def __getitem__(self, idx):
        # try:
        # Handle tuple format: (metadata_path, split_name) or just metadata_path
        scene_entry = self.all_scene_paths[idx]
        if isinstance(scene_entry, tuple):
            scene_path, split_name = scene_entry
        else:
            scene_path = scene_entry.strip()
            split_name = None

        data_json = json.load(open(scene_path, "r"))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]
        dataset_name = scene_name.split("_context_")[0] if "_context_" in scene_name else scene_name
        extras = None
        context_indices_tensor = None
        target_indices_tensor = None
        context_transient_paths = None
        target_transient_paths = None
        context_gt_paths = None
        target_gt_paths = None
        context_gt_images = None

        # In inference mode, use split_name if available, otherwise fall back to scene_name
        # Also load GT images for target views during inference
        scene_dir = os.path.dirname(scene_path)  # .../metadata/

        if self.inference:
            lookup_name = split_name if split_name is not None else scene_name
            if lookup_name in self.view_idx_list:
                current_view_idx = self.view_idx_list[lookup_name]
                input_indices = current_view_idx["context"]
                target_indices = current_view_idx["target"]

                # Check if we should use GT images for input as well (via config flag)
                # Check both inference.use_gt_images_for_inference and root level
                use_gt_for_input = self.config.get("use_gt_images_for_inference", False)
                if not use_gt_for_input and hasattr(self.config, "inference"):
                    use_gt_for_input = self.config.inference.get(
                        "use_gt_images_for_inference", False
                    )

                dataset_name = (
                    scene_name.split("_context_")[0] if "_context_" in scene_name else scene_name
                )

                has_gt_path = "gt_image_path" in frames[0]

                # Build canonical path lists for context and target views
                context_transient_paths = [
                    os.path.normpath(os.path.join(scene_dir, frames[ic]["image_path"]))
                    for ic in input_indices
                ]
                context_gt_paths = (
                    [
                        os.path.normpath(os.path.join(scene_dir, frames[ic]["gt_image_path"]))
                        for ic in input_indices
                    ]
                    if has_gt_path
                    else []
                )

                target_gt_paths = (
                    [
                        os.path.normpath(os.path.join(scene_dir, frames[ic]["gt_image_path"]))
                        for ic in target_indices
                    ]
                    if has_gt_path
                    else [
                        os.path.normpath(os.path.join(scene_dir, frames[ic]["image_path"]))
                        for ic in target_indices
                    ]
                )
                target_transient_paths = [
                    os.path.normpath(os.path.join(scene_dir, frames[ic]["image_path"]))
                    for ic in target_indices
                ]

                # Determine which images are fed into the model for context views
                if use_gt_for_input and has_gt_path:
                    input_image_paths = context_gt_paths
                    if idx == 0:
                        print(
                            "[Dataset] Inference mode: Loading GT images for INPUT views "
                            "(use_gt_images_for_inference=True)"
                        )
                else:
                    input_image_paths = context_transient_paths
                    if idx == 0 and use_gt_for_input and not has_gt_path:
                        print(
                            "[Dataset] Warning: use_gt_images_for_inference=True but no "
                            "gt_image_path found; using transient images for input"
                        )
                    elif idx == 0:
                        print(
                            "[Dataset] Inference mode: Loading transient images for INPUT views "
                            "(default)"
                        )

                input_frames_chosen = [frames[ic] for ic in input_indices]
                input_images_used, input_intrinsics, input_c2ws = self.preprocess_frames(
                    input_frames_chosen, input_image_paths
                )

                # Load additional context representations for evaluation bookkeeping
                if has_gt_path and len(context_gt_paths) == len(input_indices):
                    context_gt_images, _, _ = self.preprocess_frames(
                        input_frames_chosen, context_gt_paths
                    )
                else:
                    context_gt_images = None

                if use_gt_for_input and has_gt_path:
                    context_transient_images, _, _ = self.preprocess_frames(
                        input_frames_chosen, context_transient_paths
                    )
                else:
                    context_transient_images = input_images_used.clone()

                # Load target GT images (always GT when available)
                target_frames_chosen = [frames[ic] for ic in target_indices]
                target_images_gt, target_intrinsics, target_c2ws = self.preprocess_frames(
                    target_frames_chosen, target_gt_paths
                )

                if has_gt_path:
                    target_transient_images, _, _ = self.preprocess_frames(
                        target_frames_chosen, target_transient_paths
                    )
                else:
                    target_transient_images = target_images_gt.clone()

                # Combine context and target tensors for downstream processing
                all_images = torch.cat([input_images_used, target_images_gt], dim=0)
                all_intrinsics = torch.cat([input_intrinsics, target_intrinsics], dim=0)
                all_c2ws = torch.cat([input_c2ws, target_c2ws], dim=0)

                context_transient_images = context_transient_images.clone()
                target_transient_images = target_transient_images.clone()
                if context_gt_images is not None:
                    context_gt_images = context_gt_images.clone()

                transient_images_all = torch.cat(
                    [context_transient_images, target_transient_images], dim=0
                )

                # Load GT motion masks if requested
                gt_motion_masks_all = None
                if self.config.inference.get("use_gt_motion_masks", False):
                    # Extract base scene name (remove _context_X_split_Y suffix)
                    scene_name_for_masks = (
                        scene_name.split("_context_")[0]
                        if "_context_" in scene_name
                        else scene_name
                    )

                    mask_root_cfg = self.config.inference.get(
                        "motion_mask_root",
                        "data/dynamic_re10k/test/binary_masks",
                    )
                    if isinstance(mask_root_cfg, dict):
                        # Pick root based on dataset prefix (fallback to generic)
                        dataset_prefix = (
                            "wildrayzer" if "wildrayzer" in scene_name_for_masks else "dre10k"
                        )
                        mask_root = mask_root_cfg.get(dataset_prefix, mask_root_cfg.get("default"))
                        if mask_root is None:
                            raise ValueError(
                                f"motion_mask_root dict missing entry for dataset '{dataset_prefix}'"
                            )
                    else:
                        mask_root = mask_root_cfg

                    preserve_suffix = self.config.inference.get("motion_mask_preserve_suffix", False)
                    scene_key = scene_name if preserve_suffix else scene_name_for_masks
                    mask_dir = os.path.join(mask_root, scene_key)

                    # FAIL HARD if mask directory doesn't exist
                    if not os.path.exists(mask_dir):
                        raise FileNotFoundError(
                            f"GT mask directory not found: {mask_dir}\n"
                            f"Scene: {scene_name}\n"
                            f"use_gt_motion_masks=True requires GT masks for all scenes!"
                        )

                    # Detect if masks are 0-indexed or 1-indexed by checking first file
                    # Some scenes (kitchen, painting) are 1-indexed, most are 0-indexed
                    is_one_indexed = os.path.exists(
                        os.path.join(mask_dir, "00001.npy")
                    ) and not os.path.exists(os.path.join(mask_dir, "00000.npy"))
                    offset = 1 if is_one_indexed else 0
                    indexing_type = "1-indexed" if is_one_indexed else "0-indexed"

                    if idx == 0:
                        print(f"[Dataset] {scene_name_for_masks} masks are {indexing_type}")

                    # Load input view masks - FAIL HARD if any missing
                    input_masks = []
                    for frame_idx in input_indices:
                        mask_filename = f"{frame_idx + offset:05d}.npy"
                        mask_path = os.path.join(mask_dir, mask_filename)
                        if not os.path.exists(mask_path):
                            raise FileNotFoundError(
                                f"GT mask not found: {mask_path}\n"
                                f"Scene: {scene_name}, Frame index: {frame_idx}, "
                                f"Indexing: {indexing_type} (offset={offset})\n"
                                f"use_gt_motion_masks=True requires GT masks for all frames!"
                            )
                        mask = np.load(mask_path)  # (256, 256) uint8, values [0, 1]
                        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, 256, 256)
                        input_masks.append(mask)

                    # (num_input, 1, H_mask, W_mask)
                    gt_motion_masks_input = torch.stack(input_masks)
                    target_size = input_images_used.shape[-2:]
                    gt_motion_masks_input = F.interpolate(
                        gt_motion_masks_input,
                        size=target_size,
                        mode="nearest",
                    )
                    if idx == 0:
                        print(f"[Dataset] Loaded GT motion masks for INPUT views from {mask_dir}")

                    # Load target view masks - FAIL HARD if any missing
                    # Use same offset as input masks (already detected above)
                    target_masks = []
                    for frame_idx in target_indices:
                        mask_filename = f"{frame_idx + offset:05d}.npy"
                        mask_path = os.path.join(mask_dir, mask_filename)
                        if not os.path.exists(mask_path):
                            raise FileNotFoundError(
                                f"GT mask not found: {mask_path}\n"
                                f"Scene: {scene_name}, Frame index: {frame_idx}, "
                                f"Indexing: {indexing_type} (offset={offset})\n"
                                f"use_gt_motion_masks=True requires GT masks for all frames!"
                            )
                        mask = np.load(mask_path)  # (256, 256) uint8
                        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, 256, 256)
                        target_masks.append(mask)

                    # (num_target, 1, H_mask, W_mask)
                    gt_motion_masks_target = torch.stack(target_masks)
                    gt_motion_masks_target = F.interpolate(
                        gt_motion_masks_target,
                        size=target_size,
                        mode="nearest",
                    )
                    if idx == 0:
                        print(f"[Dataset] Loaded GT motion masks for TARGET views from {mask_dir}")

                    gt_motion_masks_all = torch.cat(
                        [gt_motion_masks_input, gt_motion_masks_target], dim=0
                    ).contiguous()

                image_indices = input_indices + target_indices
                frames_chosen = input_frames_chosen + target_frames_chosen

                context_indices_tensor = torch.arange(len(input_indices), dtype=torch.long)
                target_indices_tensor = torch.arange(len(target_indices), dtype=torch.long)
                target_indices_tensor = target_indices_tensor + len(input_indices)

                extras = {}
                if context_gt_images is not None:
                    extras["context_gt_images"] = context_gt_images
                if context_gt_paths:
                    extras["context_gt_image_paths"] = context_gt_paths
                extras["context_source_indices"] = input_indices
                extras["target_source_indices"] = target_indices

                input_images = all_images
                input_intrinsics = all_intrinsics
                input_c2ws = all_c2ws
            else:
                # Fallback to random sampling (use transient images for all)
                image_indices = self.view_selector(frames, scene_idx=idx, scene_path=scene_path)
                if image_indices is None:
                    return self.__getitem__(random.randint(0, len(self) - 1))

                image_paths_chosen = [
                    os.path.normpath(os.path.join(scene_dir, frames[ic]["image_path"]))
                    for ic in image_indices
                ]
                frames_chosen = [frames[ic] for ic in image_indices]
                input_images, input_intrinsics, input_c2ws = self.preprocess_frames(
                    frames_chosen, image_paths_chosen
                )
        else:
            # Training mode: sample input and target views using transient images for all
            image_indices = self.view_selector(frames, scene_idx=idx, scene_path=scene_path)
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            image_paths_chosen = [
                os.path.normpath(os.path.join(scene_dir, frames[ic]["image_path"]))
                for ic in image_indices
            ]
            frames_chosen = [frames[ic] for ic in image_indices]
            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(
                frames_chosen, image_paths_chosen
            )

        # except:
        #     traceback.print_exc()
        #     print(f"error loading")
        #     print(image_indices)
        #     print(image_paths_chosen)
        #     return self.__getitem__(random.randint(0, len(self) - 1))

        # Skip pose preprocessing for visualization - keep poses in original coordinate system
        # scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        # input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        # Extract original filenames for each selected frame (for inference mapping)
        original_filenames = [frames[ic]["image_path"] for ic in image_indices]

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        batch_dict = {
            "image": input_images,
            "c2w": input_c2ws,
            "fxfycxcy": input_intrinsics,
            "index": indices,
            "scene_name": scene_name,
            "original_filenames": original_filenames,  # list of str (original image paths)
        }

        if extras is not None:
            batch_dict["context_indices"] = context_indices_tensor
            batch_dict["target_indices"] = target_indices_tensor
            batch_dict["context_original_filenames"] = context_transient_paths
            batch_dict["target_original_filenames"] = target_transient_paths
            if context_gt_paths:
                batch_dict["context_gt_original_filenames"] = context_gt_paths
            if target_gt_paths:
                batch_dict["target_gt_original_filenames"] = target_gt_paths
            batch_dict["context_source_indices"] = torch.tensor(
                extras.get("context_source_indices", []), dtype=torch.long
            )
            batch_dict["target_source_indices"] = torch.tensor(
                extras.get("target_source_indices", []), dtype=torch.long
            )
            batch_dict["extras"] = extras
        batch_dict["dataset_name"] = dataset_name

        # Add GT motion masks if loaded (inference only)
        if (
            self.inference
            and self.config.inference.get("use_gt_motion_masks", False)
            and "gt_motion_masks_all" in locals()
            and gt_motion_masks_all is not None
        ):
            batch_dict["gt_motion_masks"] = gt_motion_masks_all

        # Apply copy-paste augmentation (training only, on both input and target views)
        if self.use_copy_paste:
            original_images = batch_dict["image"].clone()
            paste_prob = self.copy_paste_config.get("paste_probability", 0.5)
            copy_paste_mask = None

            if random.random() < paste_prob:
                try:
                    cfg = self.copy_paste_config

                    per_view_prob = cfg.get("per_view_objects_prob", None)
                    if per_view_prob is not None:
                        per_view_choice = random.random() < per_view_prob
                    else:
                        per_view_choice = cfg.get("per_view_objects", False)

                    generate_masks = cfg.get("generate_overlay_masks", False)

                    augmented_images, overlay_masks = apply_copy_paste_to_views(
                        batch_dict["image"],
                        self.copy_paste_extractor,
                        num_animals_range=tuple(cfg.get("num_animals_per_scene", [1, 2])),
                        scale_range=tuple(cfg.get("scale_range", [0.1, 0.3])),
                        position_margin=cfg.get("position_margin", 0.1),
                        blend=cfg.get("blend", True),
                        sigma=cfg.get("sigma", 3),
                        per_view_objects=per_view_choice,
                        generate_overlay_masks=generate_masks,
                    )
                    batch_dict["image"] = augmented_images

                    if overlay_masks is not None:
                        copy_paste_mask = overlay_masks
                except Exception as e:
                    # Silently skip augmentation on error to not interrupt training
                    print(f"⚠️  Copy-paste augmentation failed for sequence {idx}: {e}")

            if copy_paste_mask is None:
                images_tensor = batch_dict["image"]
                v, _, h, w = images_tensor.shape
                copy_paste_mask = images_tensor.new_zeros((v, 1, h, w))

            batch_dict["copy_paste_mask"] = copy_paste_mask
            batch_dict["image_clean"] = original_images
        else:
            batch_dict.pop("copy_paste_mask", None)
            batch_dict.pop("image_clean", None)

        return batch_dict

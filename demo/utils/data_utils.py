# Copyright (c) 2025 Hanwen Jiang, Xuweiyi Chen. Adapted for WildRayZer from the RayZer project.

import random
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
import imageio
import math


def create_video_from_frames(frames, output_video_file, framerate=30):
    """
    Creates a video from a sequence of frames.

    Parameters:
        frames (numpy.ndarray): Array of image frames (shape: N x H x W x C).
        output_video_file (str): Path to save the output video file.
        framerate (int, optional): Frames per second for the video. Default is 30.
    """
    frames = np.asarray(frames)

    # Normalize frames if values are in [0,1] range
    if frames.max() <= 1:
        frames = (frames * 255).astype(np.uint8)

    imageio.mimsave(output_video_file, frames, fps=framerate, quality=8)


# used in lvsm repo, which is slightly different from rayzer's view sampling setting
class ProcessData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        """
        Args:
            c2w (torch.tensor): [b, v, 4, 4]
            fxfycxcy (torch.tensor): [b, v, 4]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:, :, 0], fxfycxcy[:, :, 1], fxfycxcy[:, :, 2], fxfycxcy[:, :, 3]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d

    def fetch_views(self, data_batch, has_target_image=False, target_has_input=True):
        """
        Splits the input data batch into input and target sets.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w], optional for some target views
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            target_has_input (bool): If True, target includes input views.

        Returns:
            tuple: (input_dict, target_dict), both as EasyDict objects.

        """
        # randomize input views if dynamic_input_view_num is True and not in inference mode
        if self.config.training.get("dynamic_input_view_num", False) and (
            not self.config.inference.get("if_inference", False)
        ):
            self.config.training.num_input_views = np.random.randint(2, 5)

        input_dict, target_dict = {}, {}
        # index = [] save for future use if we want to select specific views
        # Handle different data formats from adapters
        if "input" in data_batch and "target" in data_batch:
            # Handle nested format from simple_stereo4d_adapter
            # The DataLoader batches the data, so we get [B, V, C, H, W] format
            input_images = data_batch["input"]["image"]  # [B, num_input, 3, H, W]
            target_images = data_batch["target"]["image"]  # [B, num_target, 3, H, W]

            # Concatenate along the view dimension (dim=1)
            all_images = torch.cat([input_images, target_images], dim=1)  # [B, V, 3, H, W]

            # Create flat structure that rest of fetch_views expects
            data_batch_flat = {
                "image": all_images,  # [B, V, 3, H, W] - already has batch dim
                "scene_name": data_batch.get("scene_name", ""),
                "fps": data_batch.get("fps", 0.0),
                "frame_count": data_batch.get("frame_count", 0),
            }

            # Add time if available
            if "time" in data_batch["input"] and "time" in data_batch["target"]:
                input_time = data_batch["input"]["time"]  # [B, num_input, 1]
                target_time = data_batch["target"]["time"]  # [B, num_target, 1]
                all_time = torch.cat([input_time, target_time], dim=1)  # [B, V, 1]
                data_batch_flat["time"] = all_time  # [B, V, 1]

            # Add index field for visualization (required by metric_utils.py)
            # Create index from time information and batch indices
            B = all_images.shape[0]
            V = all_images.shape[1]

            # Use time values to create pseudo frame indices
            # Convert normalized time [-1, 1] to frame-like indices [0, 1000]
            if "time" in data_batch_flat:
                time_tensor = data_batch_flat["time"]  # [B, V, 1]
                pseudo_frame_indices = (
                    (time_tensor.squeeze(-1) + 1.0) * 500
                ).long()  # [B, V] in [0, 1000]
            else:
                # Fallback: use view indices as frame indices
                pseudo_frame_indices = (
                    torch.arange(V, device=all_images.device).unsqueeze(0).expand(B, -1)
                )  # [B, V]

            # Create scene indices (batch indices repeated for each view)
            scene_indices = (
                torch.arange(B, device=all_images.device).unsqueeze(1).expand(B, V)
            )  # [B, V]

            # Combine frame and scene indices
            index_field = torch.stack([pseudo_frame_indices, scene_indices], dim=-1)  # [B, V, 2]
            data_batch_flat["index"] = index_field

            data_batch = data_batch_flat
            num_target_views, num_views, bs = (
                self.config.training.num_target_views,
                data_batch["image"].size(1),
                data_batch["image"].size(0),
            )
        elif "c2w" in data_batch:
            num_target_views, num_views, bs = (
                self.config.training.num_target_views,
                data_batch["c2w"].size(1),
                data_batch["image"].size(0),
            )
        else:
            # For pose-free datasets, get dimensions from image tensor
            num_target_views, num_views, bs = (
                self.config.training.num_target_views,
                data_batch["image"].size(1),
                data_batch["image"].size(0),
            )
        assert (
            num_target_views < num_views
        ), f"We have {num_views} views, but we want to select {num_target_views} target views. This is more than the total number of views we have."

        K = int(self.config.training.num_input_views)
        T = int(num_target_views)
        # Decide the target view indices
        if target_has_input:
            # Target views are the remaining views after inputs (maintaining temporal order)
            # Since your dataset puts inputs first, targets are views K to V-1
            target_indices = list(range(K, num_views))  # Views after the first K input views
            index = torch.tensor(
                [
                    target_indices[
                        :num_target_views
                    ]  # Take the first num_target_views from remaining
                    for _ in range(bs)
                ],
                dtype=torch.long,
                device=data_batch["image"].device,
            )  # [b, num_target_views]
        else:
            # assert (
            #     self.config.training.num_input_views + num_target_views <= self.config.training.num_views
            # ), f"We have {self.config.training.num_views} views in total, but we want to select {self.config.training.num_input_views} input views and {num_target_views} target views. This is more than the total number of views we have."

            # index = torch.tensor([
            #     [self.config.training.num_views - 1 - j for j in range(num_target_views)]
            #     for _ in range(bs)
            # ], dtype=torch.long, device=data_batch["image"].device)
            # index = torch.sort(index, dim=1).values # [b, num_target_views]
            assert (
                K + T <= num_views
            ), f"Need K (inputs) + T (targets) <= num_views, got {K}+{T}>{num_views}"
            base = torch.arange(K, K + T, device=data_batch["image"].device)  # [T]
            index = base.unsqueeze(0).expand(bs, -1).to(torch.long)  # [b, T]

        skip_keys = {
            "scene_name",
            "fps",
            "frame_count",
            "context_indices",
            "target_indices",
            "context_source_indices",
            "target_source_indices",
            "context_original_filenames",
            "target_original_filenames",
            "context_gt_original_filenames",
            "target_gt_original_filenames",
        }

        for key, value in data_batch.items():
            if key in skip_keys:
                # Preserve metadata tensors/lists without slicing by view dimension
                input_dict[key] = value
                target_dict[key] = value
                continue

            # Handle tensors with view dimensions
            if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                input_dict[key] = value[:, : self.config.training.num_input_views, ...]
            else:
                # Handle other data types or unexpected shapes
                input_dict[key] = value
                target_dict[key] = value
                continue

            to_expand_dim = value.shape[2:]  # [b, v, ...] -> [...]
            expanded_index = index.view(
                index.shape[0], index.shape[1], *(1,) * len(to_expand_dim)
            ).expand(-1, -1, *to_expand_dim)

            # Don't have target image supervision
            if key == "image" and not has_target_image:
                continue
            else:
                target_dict[key] = torch.gather(value, dim=1, index=expanded_index)

        height, width = data_batch["image"].shape[3], data_batch["image"].shape[4]
        input_dict["image_h_w"] = (height, width)
        target_dict["image_h_w"] = (height, width)

        input_dict, target_dict = edict(input_dict), edict(target_dict)

        return input_dict, target_dict

    @torch.no_grad()
    def forward(self, data_batch, has_target_image=True, target_has_input=True, compute_rays=True):
        """
        Preprocesses the input data batch and (optionally) computes ray_o and ray_d.

        Args:
            data_batch (dict): Contains input tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
            has_target_image (bool): If True, target views have image supervision.
            target_has_input (bool): If True, target views can be sampled from input views.
            compute_rays (bool): If True, compute ray_o and ray_d.

        Returns:
            Input and Target data_batch (dict): Contains processed tensors with the following keys:
                - 'image' (torch.Tensor): Shape [b, v, c, h, w]
                - 'fxfycxcy' (torch.Tensor): Shape [b, v, 4]
                - 'c2w' (torch.Tensor): Shape [b, v, 4, 4]
                - 'ray_o' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'ray_d' (torch.Tensor): Shape [b, v, 3, h, w]
                - 'image_h_w' (tuple): (height, width)
        """
        input_dict, target_dict = self.fetch_views(
            data_batch, has_target_image=has_target_image, target_has_input=target_has_input
        )

        if compute_rays:
            for dict in [input_dict, target_dict]:
                c2w = dict["c2w"]
                fxfycxcy = dict["fxfycxcy"]
                image_height, image_width = dict["image_h_w"]

                ray_o, ray_d = self.compute_rays(
                    c2w, fxfycxcy, image_height, image_width, device=data_batch["image"].device
                )
                dict["ray_o"], dict["ray_d"] = ray_o, ray_d

        return input_dict, target_dict


class SplitData(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Basic check: we want num_input_views + num_target_views = num_views
        assert (
            self.config.training.num_views
            == self.config.training.num_input_views + self.config.training.num_target_views
        ), "num_input_views + num_target_views must equal num_views"

        # Precompute input and target indices (no overlap, evenly spaced)
        self.input_pattern, self.target_pattern = self._build_indices(
            total_views=self.config.training.num_views,
            num_input_views=self.config.training.num_input_views,
            num_target_views=self.config.training.num_target_views,
        )

        print(
            "When not using random index, input and target indices are:",
            self.input_pattern,
            self.target_pattern,
        )
        # tmp1, tmp2 = self.input_pattern[-1].clone(), self.target_pattern[-1].clone()
        # self.target_pattern[-1] = tmp1
        # self.input_pattern[-1] = tmp2

    @torch.no_grad()
    def forward(self, data_batch, random_index=True):
        """
        Each tensor in data_batch has shape [B, V, ...].
        We'll slice along dimension 1 (the 'view' dimension).
        """
        input_dict, target_dict = {}, {}
        B, V = data_batch["image"].shape[:2]
        batch_idx = torch.arange(B).unsqueeze(1).to(data_batch["image"].device)

        # Check if using dataset-provided indices (evaluation mode)
        use_dataset_indices = "context_indices" in data_batch and "target_indices" in data_batch

        if use_dataset_indices:
            # use loaded view indices, for evaluation
            input_pattern = data_batch["context_indices"]
            target_pattern = data_batch["target_indices"]
        else:
            # for training
            if random_index:
                input_pattern, target_pattern = self.get_random_index(B, V)
            else:
                input_pattern, target_pattern = self.input_pattern.unsqueeze(0).repeat(
                    B, 1
                ), self.target_pattern.unsqueeze(0).repeat(B, 1)

        for key, value in data_batch.items():
            if key in set(
                [
                    "scene_name",
                    "context_indices",
                    "target_indices",
                    "dataset_sources",
                    "dataset_source",
                ]
            ):
                continue
            # value shape: [B, V, ...]
            B, V = value.shape[:2]
            # Only validate view count for training (not for evaluation with dataset-provided indices)
            if not use_dataset_indices:
                expected_views = self.config.training.num_views
                if V != expected_views:
                    raise ValueError(f"Expected {key} to have {expected_views} views, got {V}.")

            input_dict[key] = value[batch_idx, input_pattern, ...]
            target_dict[key] = value[batch_idx, target_pattern, ...]

        # Add scene_name to both dicts (needed for evaluation)
        # scene_name should be indexed by batch when it's a list
        if "scene_name" in data_batch:
            scene_names = data_batch["scene_name"]
            # If it's a list (batched), keep as list; if single string, keep as string
            input_dict["scene_name"] = scene_names
            target_dict["scene_name"] = scene_names

        return edict(input_dict), edict(target_dict), input_pattern, target_pattern

    def _build_indices(self, total_views, num_input_views, num_target_views):
        """
        Build two arrays of indices for input and target such that
        they don't overlap and cover all views evenly.
        E.g. total_views=24, num_input_views=16, num_target_views=8
        => input might be [0,1,3,4,6,7,9,10,...], target [2,5,8,11,...]
        """
        # Simple approach: gcd-based grouping
        g = math.gcd(num_input_views, num_target_views)
        group_size = total_views // g  # number of consecutive indices per group
        in_per_group = num_input_views // g
        tar_per_group = num_target_views // g

        input_indices = []
        target_indices = []

        for group_idx in range(g):
            start = group_idx * group_size
            block = list(range(start, start + group_size))
            # first part goes to inputs
            input_indices.extend(block[:in_per_group])
            # next part goes to targets
            target_indices.extend(block[in_per_group : in_per_group + tar_per_group])

        # Convert to torch.LongTensor
        input_indices = torch.tensor(input_indices, dtype=torch.long)
        target_indices = torch.tensor(target_indices, dtype=torch.long)
        input_indices, _ = torch.sort(input_indices)
        target_indices, _ = torch.sort(target_indices)

        return input_indices, target_indices

    def get_random_index(self, b, v):
        total_views = self.config.training.num_views
        num_input_views = self.config.training.num_input_views
        num_target_views = self.config.training.num_target_views
        random_shuffle = self.config.training.view_selector.get("shuffle", False)

        assert (
            num_input_views + num_target_views == total_views
        ), "Mismatch in total views allocation."

        rand_vals = torch.rand(b, v)  # shape [B, V]
        perms = rand_vals.argsort(dim=1)  # shape [B, V]

        # Ensure at least one index in input is smaller than all in target, and one index in input is larger than all in target
        idx_part1 = torch.zeros((b, num_input_views), dtype=torch.long, device=perms.device)
        idx_part2 = torch.zeros((b, num_target_views), dtype=torch.long, device=perms.device)

        for i in range(b):
            # Ensure the first index in input is always 0 and the last index is always v-1
            idx_part1[i, 0] = 0
            idx_part1[i, -1] = v - 1

            # Remaining indices to choose from
            remaining_indices = torch.arange(1, v - 1, device=perms.device)  # Exclude 0 and v-1

            # Randomly sample (num_input_views - 2) indices from remaining
            middle_size = num_input_views - 2
            middle_indices = remaining_indices[torch.randperm(len(remaining_indices))[:middle_size]]
            middle_indices, _ = middle_indices.sort()  # Ensure sorted order
            # Assign middle indices to idx_part1
            idx_part1[i, 1:-1] = middle_indices

            # Target indices are the remaining ones
            idx_part2_indeices = torch.tensor(
                [x for x in range(v) if x not in idx_part1[i]], device=perms.device
            )
            idx_part2_indeices, _ = idx_part2_indeices.sort()  # Ensure sorted order for target
            idx_part2[i] = idx_part2_indeices

            if random_shuffle:
                idx_part1[i] = idx_part1[i][torch.randperm(num_input_views)]
                idx_part2[i] = idx_part2[i][torch.randperm(num_target_views)]

        return idx_part1, idx_part2

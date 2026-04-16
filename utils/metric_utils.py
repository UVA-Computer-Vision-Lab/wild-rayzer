# Copyright (c) 2025 Hanwen Jiang, Xuweiyi Chen. Adapted for WildRayZer from the RayZer project.

import torch
from torch import Tensor
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
from utils import data_utils
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print
import torch.nn.functional as F
from utils.masked_metrics import compute_all_masked_metrics

import warnings

# Suppress warnings for LPIPS loss loading
warnings.filterwarnings(
    "ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Arguments other than a weight enum.*"
)


def _tensor_to_uint8_image(tensor: Tensor) -> np.ndarray:
    tensor = tensor.detach().to(torch.float32)
    array = tensor.cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0
    return np.clip(array.round(), 0, 255).astype(np.uint8)


def _save_rgb_image(tensor: Tensor, path: str) -> None:
    Image.fromarray(_tensor_to_uint8_image(tensor)).save(path)


def _save_mask_image(tensor: Tensor, path: str) -> None:
    mask = tensor.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    mask = np.clip(mask, 0.0, 1.0)
    mask_img = (mask * 255.0).round().astype(np.uint8)
    Image.fromarray(mask_img).save(path)


def _save_mask_heatmap(tensor: Tensor, path: str) -> None:
    mask = tensor.detach().float()
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.clamp(0.0, 1.0).cpu().numpy()

    try:
        import matplotlib.cm as cm
    except ImportError:
        # Fallback to grayscale if matplotlib is unavailable
        mask_img = (mask * 255.0).round().astype(np.uint8)
        Image.fromarray(mask_img).save(path)
        return

    colormap = cm.get_cmap("magma")
    colored = colormap(mask)[..., :3]
    heatmap = (colored * 255.0).round().astype(np.uint8)
    Image.fromarray(heatmap).save(path)


def _save_mask_overlay(mask: Tensor, image: Tensor, path: str, alpha: float = 0.45) -> None:
    mask = mask.detach().float()
    if mask.ndim == 3:
        mask = mask[0]
    mask = mask.clamp(0.0, 1.0)

    image_np = image.detach().to(torch.float32).cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()

    try:
        import matplotlib.cm as cm

        colored = cm.get_cmap("magma")(mask.cpu().numpy())[..., :3]
    except ImportError:
        colored = np.stack([mask.cpu().numpy()] * 3, axis=-1)

    overlay = (1.0 - alpha * mask.cpu().numpy()[..., None]) * image_np + alpha * colored
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_img = (overlay * 255.0).round().astype(np.uint8)
    Image.fromarray(overlay_img).save(path)


def _safe_mean(values):
    arr = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not arr:
        return None
    return float(sum(arr) / len(arr))


def _get_batch_value(batch_field, idx, default=None):
    if batch_field is None:
        return default
    if isinstance(batch_field, list):
        return batch_field[idx] if idx < len(batch_field) else default
    if isinstance(batch_field, torch.Tensor):
        return batch_field[idx]
    return batch_field


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, list):
        return list(value)
    return [value]


def _get_from_extras(extras: dict, key: str, idx: int, default=None):
    if extras is None or key not in extras:
        return default
    value = extras[key]
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return value[idx]
    if isinstance(value, list):
        return value[idx]
    return default


def _save_plucker_visualizations(
    plucker_tensor,
    out_dir,
    view_ids=None,
    step=16,
    include_quiver=False,
):
    """Persist per-view Plücker diagnostics to ``out_dir``."""

    if plucker_tensor is None or plucker_tensor.numel() == 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PluckerVis] matplotlib unavailable: {exc}")
        return

    plucker_cpu = plucker_tensor.detach().float().cpu()
    num_views = plucker_cpu.shape[0]
    stats_lines = []

    if view_ids is None:
        view_ids = [f"view{idx:02d}" for idx in range(num_views)]
    else:
        view_ids = [f"view{int(idx):02d}" for idx in view_ids]

    for view_idx in range(num_views):
        view_tag = view_ids[view_idx]
        base_name = os.path.join(out_dir, view_tag)

        m = plucker_cpu[view_idx, 0:3]
        d = plucker_cpu[view_idx, 3:6]

        # Direction RGB (mapped from [-1, 1] to [0, 1])
        d_img = (d.clamp(-1.0, 1.0) + 1.0) / 2.0
        d_img_np = d_img.permute(1, 2, 0).numpy()
        plt.imsave(f"{base_name}_direction.png", d_img_np)

        # Moment norm heatmap
        m_norm = torch.linalg.norm(m, dim=0).numpy()
        fig, ax = plt.subplots()
        im = ax.imshow(m_norm)
        ax.set_title("‖m‖ (distance to origin)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(f"{base_name}_moment_norm.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Klein residual heatmap |d · m|
        resid = torch.abs((m * d).sum(0)).numpy()
        fig, ax = plt.subplots()
        im = ax.imshow(resid)
        ax.set_title("|d·m| (Klein residual)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(f"{base_name}_klein_residual.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        stats_lines.append(
            f"{view_tag}: mean={resid.mean():.3e}, "
            f"p95={np.percentile(resid, 95):.3e}, max={resid.max():.3e}"
        )

        if include_quiver and step > 0:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            ii = torch.arange(0, m.shape[1], step)
            jj = torch.arange(0, m.shape[2], step)
            if ii.numel() == 0 or jj.numel() == 0:
                continue

            D = d[:, ii][:, :, jj].reshape(3, -1).T
            M = m[:, ii][:, :, jj].reshape(3, -1).T

            # Project to tangent plane for numerical stability
            dot = (D * M).sum(dim=1, keepdim=True)
            Mtan = M - dot * D

            Dn = D.numpy()
            Mn = Mtan.numpy()

            mean_norm = np.linalg.norm(Mn, axis=1).mean()
            scale = 0.25 / (mean_norm + 1e-9)
            U = Mn * scale

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 20)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.5)

            ax.quiver(
                Dn[:, 0],
                Dn[:, 1],
                Dn[:, 2],
                U[:, 0],
                U[:, 1],
                U[:, 2],
                length=1.0,
                normalize=False,
                linewidth=0.6,
            )
            ax.set_title(f"{view_tag} directions and tangent moments")
            ax.set_box_aspect([1, 1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            fig.savefig(f"{base_name}_quiver.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    if stats_lines:
        stats_path = os.path.join(out_dir, "klein_residual_stats.txt")
        with open(stats_path, "w") as fh:
            fh.write("\n".join(stats_lines))


@torch.no_grad()
def compute_psnr(ground_truth: Tensor, predicted: Tensor, eps: float = 1e-10) -> Tensor:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.

    - Accepts images in [0,1] or [-1,1]; automatically maps to [0,1].
    - Returns per-image PSNR in dB for the batch.
    """

    def to_01(x: Tensor) -> Tensor:
        return (x + 1.0) * 0.5 if x.min() < 0.0 else x

    gt = torch.clamp(to_01(ground_truth), 0.0, 1.0)
    pd = torch.clamp(to_01(predicted), 0.0, 1.0)
    mse = reduce((gt - pd) ** 2, "b c h w -> b", "mean").clamp_min(eps)
    return -10.0 * torch.log10(mse)


@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS

    return LPIPS(net=net_type).to(device)


@torch.no_grad()
def compute_lpips(ground_truth: Tensor, predicted: Tensor, normalize: bool = True) -> Tensor:
    """
    Compute Learned Perceptual Image Patch Similarity between images.

    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()


@torch.no_grad()
def compute_ssim(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """
    Compute Structural Similarity Index between images.

    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]

    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values = []

    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)

    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)


@torch.no_grad()
def export_results(
    result: edict, batch: dict, out_dir: str, compute_metrics: bool = False, config=None
):
    """Save per-scene evaluation bundles including images, masks, metadata, and metrics."""

    os.makedirs(out_dir, exist_ok=True)

    input_data, target_data = result.input, result.target
    batch_size = input_data.image.size(0)

    extras_batch = batch.get("extras", None)
    dataset_names_batch = batch.get("dataset_name", None)
    context_indices_batch = batch.get("context_indices", None)
    target_indices_batch = batch.get("target_indices", None)
    context_source_indices_batch = batch.get("context_source_indices", None)
    target_source_indices_batch = batch.get("target_source_indices", None)
    context_orig_batch = batch.get("context_original_filenames", None)
    target_orig_batch = batch.get("target_original_filenames", None)
    context_gt_orig_batch = batch.get("context_gt_original_filenames", None)
    target_gt_orig_batch = batch.get("target_gt_original_filenames", None)

    for batch_idx in range(batch_size):
        # Scene naming -----------------------------------------------------
        if isinstance(input_data.scene_name, list):
            scene_name = input_data.scene_name[batch_idx]
        else:
            scene_name = input_data.scene_name

        dataset_name = None
        if dataset_names_batch is not None:
            dataset_name = _get_batch_value(dataset_names_batch, batch_idx)
        if dataset_name is None:
            dataset_name = (
                scene_name.split("_context_")[0] if "_context_" in scene_name else scene_name
            )

        render_input_tensor = getattr(result, "render_input", None)

        sample_dir = os.path.join(out_dir, scene_name)
        input_dir = os.path.join(sample_dir, "input")
        gt_dir = os.path.join(sample_dir, "gt")
        rendered_dir = os.path.join(sample_dir, "rendered")
        mask_dir = os.path.join(sample_dir, "mask")
        vis_dir = os.path.join(sample_dir, "visualizations")
        rendered_context_dir = os.path.join(sample_dir, "rendered_context")

        directories = [sample_dir, input_dir, gt_dir, rendered_dir, mask_dir, vis_dir]
        if isinstance(render_input_tensor, torch.Tensor):
            directories.append(rendered_context_dir)

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Tensors (on device) ---------------------------------------------
        context_input_tensor = input_data.image[batch_idx]
        target_gt_tensor = target_data.image[batch_idx]
        rendered_tensor = result.render[batch_idx]
        rendered_context_tensor = (
            render_input_tensor[batch_idx]
            if isinstance(render_input_tensor, torch.Tensor)
            else None
        )

        context_masks_tensor = (
            input_data.gt_motion_masks[batch_idx]
            if hasattr(input_data, "gt_motion_masks") and input_data.gt_motion_masks is not None
            else None
        )
        target_masks_tensor = (
            target_data.gt_motion_masks[batch_idx]
            if hasattr(target_data, "gt_motion_masks") and target_data.gt_motion_masks is not None
            else None
        )
        predicted_input_masks_tensor = None
        predicted_target_masks_tensor = None

        v_context = context_input_tensor.shape[0]
        v_target = target_gt_tensor.shape[0]

        if (
            hasattr(result, "predicted_input_masks")
            and result.predicted_input_masks is not None
            and v_context > 0
        ):
            pred_all_input = result.predicted_input_masks
            if (
                isinstance(pred_all_input, torch.Tensor)
                and pred_all_input.shape[0] >= (batch_idx + 1) * v_context
            ):
                start = batch_idx * v_context
                end = start + v_context
                predicted_input_masks_tensor = torch.sigmoid(pred_all_input[start:end])

        if (
            hasattr(result, "predicted_target_masks")
            and result.predicted_target_masks is not None
            and v_target > 0
        ):
            pred_all_target = result.predicted_target_masks
            if (
                isinstance(pred_all_target, torch.Tensor)
                and pred_all_target.shape[0] >= (batch_idx + 1) * v_target
            ):
                start = batch_idx * v_target
                end = start + v_target
                predicted_target_masks_tensor = torch.sigmoid(pred_all_target[start:end])

        extras = extras_batch if isinstance(extras_batch, dict) else None
        context_gt_tensor = _get_from_extras(extras, "context_gt_images", batch_idx)
        context_transient_paths = _get_from_extras(
            extras, "context_transient_image_paths", batch_idx, default=[]
        )
        target_transient_paths = _get_from_extras(
            extras, "target_transient_image_paths", batch_idx, default=[]
        )
        context_gt_paths_extra = _get_from_extras(
            extras, "context_gt_image_paths", batch_idx, default=[]
        )
        target_gt_paths_extra = _get_from_extras(
            extras, "target_gt_image_paths", batch_idx, default=[]
        )

        # Save RGB images --------------------------------------------------
        context_input_cpu = context_input_tensor.detach().cpu()
        for view_idx, img in enumerate(context_input_cpu):
            _save_rgb_image(img, os.path.join(input_dir, f"context_view_{view_idx:02d}.png"))

        context_gt_cpu = None
        if isinstance(context_gt_tensor, torch.Tensor):
            context_gt_cpu = context_gt_tensor.detach().cpu()
            for view_idx, img in enumerate(context_gt_cpu):
                _save_rgb_image(img, os.path.join(gt_dir, f"context_view_{view_idx:02d}.png"))

        target_gt_cpu = target_gt_tensor.detach().cpu()
        rendered_cpu = rendered_tensor.detach().cpu()
        for view_idx, img in enumerate(target_gt_cpu):
            _save_rgb_image(img, os.path.join(gt_dir, f"target_view_{view_idx:02d}.png"))
        for view_idx, img in enumerate(rendered_cpu):
            _save_rgb_image(img, os.path.join(rendered_dir, f"target_view_{view_idx:02d}.png"))

        rendered_context_cpu = None
        if isinstance(rendered_context_tensor, torch.Tensor):
            rendered_context_cpu = rendered_context_tensor.detach().cpu()
            for view_idx, img in enumerate(rendered_context_cpu):
                _save_rgb_image(
                    img, os.path.join(rendered_context_dir, f"context_view_{view_idx:02d}.png")
                )

        # Save GT masks ----------------------------------------------------
        if context_masks_tensor is not None:
            context_masks_cpu = context_masks_tensor.detach().cpu()
            for view_idx, mask in enumerate(context_masks_cpu):
                _save_mask_image(mask, os.path.join(mask_dir, f"context_view_{view_idx:02d}.png"))
                np.save(
                    os.path.join(mask_dir, f"context_view_{view_idx:02d}.npy"),
                    mask.numpy(),
                )

        if target_masks_tensor is not None:
            target_masks_cpu = target_masks_tensor.detach().cpu()
            for view_idx, mask in enumerate(target_masks_cpu):
                _save_mask_image(mask, os.path.join(mask_dir, f"target_view_{view_idx:02d}.png"))
                np.save(
                    os.path.join(mask_dir, f"target_view_{view_idx:02d}.npy"),
                    mask.numpy(),
                )

        if predicted_input_masks_tensor is not None:
            for view_idx, mask in enumerate(predicted_input_masks_tensor):
                _save_mask_heatmap(
                    mask, os.path.join(mask_dir, f"pred_context_view_{view_idx:02d}.png")
                )
                _save_mask_overlay(
                    mask,
                    context_input_tensor[view_idx],
                    os.path.join(mask_dir, f"pred_context_view_{view_idx:02d}_overlay.png"),
                )
                np.save(
                    os.path.join(mask_dir, f"pred_context_view_{view_idx:02d}.npy"),
                    mask.detach().to(torch.float32).cpu().numpy(),
                )

        if predicted_target_masks_tensor is not None:
            for view_idx, mask in enumerate(predicted_target_masks_tensor):
                _save_mask_heatmap(
                    mask, os.path.join(mask_dir, f"pred_target_view_{view_idx:02d}.png")
                )
                _save_mask_overlay(
                    mask,
                    target_gt_tensor[view_idx],
                    os.path.join(mask_dir, f"pred_target_view_{view_idx:02d}_overlay.png"),
                )
                np.save(
                    os.path.join(mask_dir, f"pred_target_view_{view_idx:02d}.npy"),
                    mask.detach().to(torch.float32).cpu().numpy(),
                )

        # Save Plücker visualizations --------------------------------------
        if (
            config is not None
            and config.inference.get("save_plucker_vis", False)
            and hasattr(result, "plucker_target")
            and result.plucker_target is not None
        ):
            plucker_target_tensor = result.plucker_target[batch_idx]
            plucker_dir = os.path.join(sample_dir, "plucker_vis")

            # Get target view indices for naming
            target_view_indices = None
            if target_indices_batch is not None:
                target_view_indices = _get_batch_value(target_indices_batch, batch_idx)
                if target_view_indices is not None:
                    target_view_indices = _to_list(target_view_indices)

            stride = int(config.inference.get("plucker_vis_stride", 16))
            include_quiver = config.inference.get("plucker_vis_include_quiver", False)

            _save_plucker_visualizations(
                plucker_target_tensor,
                plucker_dir,
                view_ids=target_view_indices,
                step=max(stride, 1),
                include_quiver=bool(include_quiver),
            )

        # Visualizations ---------------------------------------------------
        _save_images(result, batch_idx, vis_dir)

        metrics_payload = {}
        summary_payload = {}

        context_psnr: list[float] = []
        context_ssim: list[float] = []
        context_lpips: list[float] = []
        target_psnr: list[float] = []
        target_ssim: list[float] = []
        target_lpips: list[float] = []
        context_masked: dict[str, list[float]] = {}
        target_masked: dict[str, list[float]] = {}

        if compute_metrics:
            device = context_input_tensor.device
            lpips_model = get_lpips_model(device=device)

            target_gt_eval = target_gt_tensor.to(dtype=torch.float32)
            rendered_eval = rendered_tensor.to(dtype=torch.float32)

            psnr_target_tensor = compute_psnr(target_gt_eval, rendered_eval)
            lpips_target_tensor = compute_lpips(target_gt_eval, rendered_eval)
            ssim_target_tensor = compute_ssim(target_gt_eval, rendered_eval)

            target_psnr = [float(v) for v in psnr_target_tensor.detach().cpu().tolist()]
            target_lpips = [float(v) for v in lpips_target_tensor.detach().cpu().tolist()]
            target_ssim = [float(v) for v in ssim_target_tensor.detach().cpu().tolist()]

            use_gt_masks = target_masks_tensor is not None

            if context_gt_tensor is not None:
                context_gt_eval = context_gt_tensor.to(device=device, dtype=torch.float32)
                context_input_eval = context_input_tensor.to(dtype=torch.float32)

                psnr_context_tensor = compute_psnr(context_gt_eval, context_input_eval)
                lpips_context_tensor = compute_lpips(context_gt_eval, context_input_eval)
                ssim_context_tensor = compute_ssim(context_gt_eval, context_input_eval)

                context_psnr = [float(v) for v in psnr_context_tensor.detach().cpu().tolist()]
                context_lpips = [float(v) for v in lpips_context_tensor.detach().cpu().tolist()]
                context_ssim = [float(v) for v in ssim_context_tensor.detach().cpu().tolist()]

                if use_gt_masks and context_masks_tensor is not None:
                    masked_ctx = compute_all_masked_metrics(
                        context_gt_eval,
                        context_input_eval,
                        context_masks_tensor.to(device=device, dtype=torch.float32),
                        lpips_model,
                        return_per_image=True,
                    )
                    context_masked = {
                        key: [float(v) for v in value.detach().cpu().tolist()]
                        for key, value in masked_ctx.items()
                    }

            if use_gt_masks:
                masked_tgt = compute_all_masked_metrics(
                    target_gt_eval,
                    rendered_eval,
                    target_masks_tensor.to(device=device, dtype=torch.float32),
                    lpips_model,
                    return_per_image=True,
                )
                target_masked = {
                    key: [float(v) for v in value.detach().cpu().tolist()]
                    for key, value in masked_tgt.items()
                }

        # Metadata assembly -----------------------------------------------
        context_positions = _to_list(_get_batch_value(context_indices_batch, batch_idx, []))
        target_positions = _to_list(_get_batch_value(target_indices_batch, batch_idx, []))

        context_indices_list = _to_list(
            _get_batch_value(context_source_indices_batch, batch_idx, context_positions)
        )
        target_indices_list = _to_list(
            _get_batch_value(target_source_indices_batch, batch_idx, target_positions)
        )
        colmap_indices = context_indices_list + target_indices_list

        context_original = _get_batch_value(context_orig_batch, batch_idx, context_transient_paths)
        target_original = _get_batch_value(target_orig_batch, batch_idx, target_transient_paths)

        context_gt_paths = _get_batch_value(
            context_gt_orig_batch, batch_idx, context_gt_paths_extra
        )
        target_gt_paths = _get_batch_value(target_gt_orig_batch, batch_idx, target_gt_paths_extra)

        num_context = context_input_tensor.shape[0]
        num_target = target_gt_tensor.shape[0]

        context_input_files = [f"input/context_view_{i:02d}.png" for i in range(num_context)]
        context_gt_files = (
            [f"gt/context_view_{i:02d}.png" for i in range(num_context)]
            if context_gt_cpu is not None
            else []
        )
        target_gt_files = [f"gt/target_view_{i:02d}.png" for i in range(num_target)]
        rendered_files = [f"rendered/target_view_{i:02d}.png" for i in range(num_target)]
        rendered_context_files = (
            [f"rendered_context/context_view_{i:02d}.png" for i in range(num_context)]
            if rendered_context_cpu is not None
            else []
        )
        mask_context_files = (
            [f"mask/context_view_{i:02d}.png" for i in range(num_context)]
            if context_masks_tensor is not None
            else []
        )
        mask_target_files = (
            [f"mask/target_view_{i:02d}.png" for i in range(num_target)]
            if target_masks_tensor is not None
            else []
        )

        context_results = {
            "indices": context_indices_list,
            "psnr": context_psnr,
            "ssim": context_ssim,
            "lpips": context_lpips,
        }
        if context_masked:
            context_results["masked"] = context_masked

        target_results = {
            "indices": target_indices_list,
            "psnr": target_psnr,
            "ssim": target_ssim,
            "lpips": target_lpips,
        }
        if target_masked:
            target_results["masked"] = target_masked

        all_psnr = context_psnr + target_psnr
        all_ssim = context_ssim + target_ssim
        all_lpips = context_lpips + target_lpips

        all_results = {
            "indices": context_indices_list + target_indices_list,
            "psnr": all_psnr,
            "ssim": all_ssim,
            "lpips": all_lpips,
        }

        context_avg = {
            "psnr": _safe_mean(context_psnr),
            "ssim": _safe_mean(context_ssim),
            "lpips": _safe_mean(context_lpips),
        }
        target_avg = {
            "psnr": _safe_mean(target_psnr),
            "ssim": _safe_mean(target_ssim),
            "lpips": _safe_mean(target_lpips),
        }
        all_avg = {
            "psnr": _safe_mean(all_psnr),
            "ssim": _safe_mean(all_ssim),
            "lpips": _safe_mean(all_lpips),
        }

        context_masked_avg = {key: _safe_mean(values) for key, values in context_masked.items()}
        target_masked_avg = {key: _safe_mean(values) for key, values in target_masked.items()}

        metadata = {
            "dataset": dataset_name,
            "scene_name": scene_name,
            "num_input_views": int(num_context),
            "num_target_views": int(num_target),
            "config": {
                "train_indices": context_indices_list,
                "test_indices": target_indices_list,
                "colmap_indices": colmap_indices,
            },
            "paths": {
                "context_original": [str(p) for p in _to_list(context_original)],
                "context_gt": [str(p) for p in _to_list(context_gt_paths)],
                "target_original": [str(p) for p in _to_list(target_original)],
                "target_gt": [str(p) for p in _to_list(target_gt_paths)],
            },
            "files": {
                "input": context_input_files,
                "gt_context": context_gt_files,
                "gt_target": target_gt_files,
                "rendered": rendered_files,
                "rendered_context": rendered_context_files,
                "mask_context": mask_context_files,
                "mask_target": mask_target_files,
            },
            "context_results": context_results,
            "target_results": target_results,
            "all_results": all_results,
            "context_avg": {k: float(v) for k, v in context_avg.items() if v is not None},
            "target_avg": {k: float(v) for k, v in target_avg.items() if v is not None},
            "all_avg": {k: float(v) for k, v in all_avg.items() if v is not None},
        }

        if context_masked:
            metadata["context_masked_avg"] = {
                k: float(v) for k, v in context_masked_avg.items() if v is not None
            }
            metadata["context_results"]["masked"] = {
                k: [float(val) for val in values] for k, values in context_masked.items()
            }

        if target_masked:
            metadata["target_masked_avg"] = {
                k: float(v) for k, v in target_masked_avg.items() if v is not None
            }
            metadata["target_results"]["masked"] = {
                k: [float(val) for val in values] for k, values in target_masked.items()
            }

        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        if compute_metrics:
            summary = {}
            if all_avg["psnr"] is not None:
                summary["psnr"] = float(all_avg["psnr"])
            if all_avg["lpips"] is not None:
                summary["lpips"] = float(all_avg["lpips"])
            if all_avg["ssim"] is not None:
                summary["ssim"] = float(all_avg["ssim"])
            if context_avg["psnr"] is not None:
                summary["psnr_context"] = float(context_avg["psnr"])
            if target_avg["psnr"] is not None:
                summary["psnr_target"] = float(target_avg["psnr"])

            metrics_payload = {
                "summary": summary,
                "context_avg": metadata["context_avg"],
                "target_avg": metadata["target_avg"],
                "all_avg": metadata["all_avg"],
            }

            if "context_masked_avg" in metadata:
                metrics_payload["context_masked_avg"] = metadata["context_masked_avg"]
            if "target_masked_avg" in metadata:
                metrics_payload["target_masked_avg"] = metadata["target_masked_avg"]

            with open(os.path.join(sample_dir, "metrics.json"), "w") as f:
                json.dump(metrics_payload, f, indent=2)

        # Save video if available ----------------------------------------
        if hasattr(result, "video_rendering") and result.video_rendering is not None:
            video_path = os.path.join(sample_dir, "input_traj.mp4")
            _save_video(result.video_rendering[batch_idx], video_path)


def export_results_with_original_filenames(
    result: edict, batch: dict, out_dir: str, save_metadata: bool = True
):
    """
    Export inference results with original filenames for motion evaluation.

    Saves:
    - Rendered images with original filenames
    - GT images with original filenames
    - Mapping JSON for each scene

    Args:
        result: EasyDict containing input, target, and rendered images
        batch: Original batch dict containing scene_name and original_filenames
        out_dir: Root directory to save results
        save_metadata: Whether to save mapping JSON

    Directory structure:
        out_dir/
        ├── scene_name_1/
        │   ├── rendered/
        │   │   ├── 00042.png  # original filename
        │   │   ├── 00087.png
        │   ├── gt/
        │   │   ├── 00042.png
        │   │   ├── 00087.png
        │   ├── mapping.json
        └── ...
    """
    import json
    from pathlib import Path

    os.makedirs(out_dir, exist_ok=True)

    input_data, target_data = result.input, result.target
    batch_size = input_data.image.size(0)

    for b in range(batch_size):
        # Get scene name and original filenames
        scene_name = batch["scene_name"][b]
        original_filenames = batch["original_filenames"][b]  # list of filenames for all views

        # Get number of input and target views
        v_input = input_data.image.shape[1]
        v_target = target_data.image.shape[1] if hasattr(target_data, "image") else 0
        v_all = v_input + v_target

        # Create scene directory
        scene_dir = os.path.join(out_dir, scene_name)
        rendered_dir = os.path.join(scene_dir, "rendered")
        gt_dir = os.path.join(scene_dir, "gt")
        os.makedirs(rendered_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        # Mapping: frame index -> original filename
        mapping = {}

        # Save target view images (rendered + GT)
        for view_idx in range(v_target):
            # Get original filename (target views come after input views)
            original_path = original_filenames[v_input + view_idx]
            # Extract just the filename (e.g., "00042.png" from "../images/scene/00042.png")
            original_filename = Path(original_path).name

            # Save rendered image
            rendered_img = result.render[b, view_idx].detach().cpu().float()  # [3, H, W]
            rendered_img_np = (
                (rendered_img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            )
            rendered_path = os.path.join(rendered_dir, original_filename)
            Image.fromarray(rendered_img_np).save(rendered_path)

            # Save GT image
            gt_img = target_data.image[b, view_idx].detach().cpu().float()  # [3, H, W]
            gt_img_np = (gt_img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            gt_path = os.path.join(gt_dir, original_filename)
            Image.fromarray(gt_img_np).save(gt_path)

            # Update mapping
            mapping[view_idx] = original_filename

        # Save mapping JSON
        if save_metadata:
            mapping_data = {
                "scene_name": scene_name,
                "num_input_views": v_input,
                "num_target_views": v_target,
                "frame_mapping": mapping,  # {view_idx: original_filename}
            }
            mapping_path = os.path.join(scene_dir, "mapping.json")
            with open(mapping_path, "w") as f:
                json.dump(mapping_data, f, indent=2)

        print(
            f"✓ Exported scene '{scene_name}': "
            f"{v_target} target views saved with original filenames"
        )


def visualize_intermediate_results(out_dir, result, save_all=False):
    os.makedirs(out_dir, exist_ok=True)

    input, target = result.input, result.target

    # if result.render is not None:
    # target_image = target.image
    # rendered_image = result.render
    # b, v, _, h, w = rendered_image.size()
    # rendered_image = rendered_image.reshape(b * v, -1, h, w)
    # target_image = target_image.reshape(b * v, -1, h, w)
    # visualized_image = torch.cat((target_image, rendered_image), dim=3).detach().cpu()
    # visualized_image = rearrange(
    #     visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2
    # )
    # visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

    # uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]

    # uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"
    # Image.fromarray(visualized_image).save(
    #     os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg")
    # )
    # with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
    #     uids = "_".join([f"{uid:08}" for uid in uids])
    #     f.write(uids)

    # Save the input image grid
    input_uids = [input.index[b, 0, -1].item() for b in range(input.index.size(0))]
    input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"
    b, v, c, h, w = input.image.size()
    input_images = input.image.reshape(b * v, c, h, w).detach().cpu()
    input_grid = rearrange(input_images, "(b v) c h w -> (b h) (v w) c", v=v)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_grid).save(os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg"))

    # Save input, rendered interpolated video, and motion masks for ALL samples
    for b in range(input.image.size(0)):
        uid = input.index[b, 0, -1].item()

        # Create sample-specific subdirectory
        sample_dir = os.path.join(out_dir, f"sample_{uid:08d}")
        os.makedirs(sample_dir, exist_ok=True)

        # vis input
        vis_image = rearrange(input.image[b], "v c h w -> h (v w) c")
        vis_image = (vis_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        Image.fromarray(vis_image).save(os.path.join(sample_dir, f"input_{uid:08d}.png"))

        if ("video_rendering" in result.keys()) and torch.is_tensor(result.video_rendering):
            video_rendering = result.video_rendering[b]  # [v, c, h, w]
            _save_video(video_rendering, os.path.join(sample_dir, f"render_video_{uid:08d}.mp4"))

        # NEW: Save 4-column motion mask visualization (GT | Soft Mask | Rendered | Binary Mask)
        if hasattr(result, "predicted_target_masks") and result.predicted_target_masks is not None:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # Get all views (input + target) for this batch sample
            v_input = input.image.shape[1]  # Number of input views
            v_target = target.image.shape[1]  # Number of target views
            v_all = v_input + v_target

            all_imgs = (
                torch.cat([input.image[b], target.image[b]], dim=0).detach().cpu()
            )  # [v_all, 3, h, w]
            rendered_imgs = result.render[b].detach().cpu()  # [v_target, 3, h, w]

            # Get motion masks for input and target views separately
            # predicted_input_masks: [b*v_input, 1, 256, 256]
            # predicted_target_masks: [b*v_target, 1, 256, 256]
            if (
                hasattr(result, "predicted_input_masks")
                and result.predicted_input_masks is not None
            ):
                if result.predicted_input_masks.ndim == 5:
                    pred_masks_input = (
                        result.predicted_input_masks[b].detach().cpu()
                    )  # [v_input, 1, 256, 256]
                else:
                    pred_masks_input = (
                        result.predicted_input_masks[b * v_input : (b + 1) * v_input].detach().cpu()
                    )  # [v_input, 1, 256, 256]
            else:
                pred_masks_input = None

            # Handle predicted_target_masks dimension
            if result.predicted_target_masks.ndim == 5:
                pred_masks_target = (
                    result.predicted_target_masks[b].detach().cpu()
                )  # [v_target, 1, 256, 256]
            elif result.predicted_target_masks.ndim == 4:
                pred_masks_target = (
                    result.predicted_target_masks[b * v_target : (b + 1) * v_target].detach().cpu()
                )  # [v_target, 1, 256, 256]
            else:
                num_dims = result.predicted_target_masks.ndim
                raise ValueError("Unexpected predicted_target_masks dimensions: " f"{num_dims}D")

            # Concatenate input and target masks
            if pred_masks_input is not None:
                # Ensure both have the same dimensions
                if pred_masks_input.ndim != pred_masks_target.ndim:
                    raise ValueError(
                        f"Dimension mismatch: pred_masks_input is {pred_masks_input.ndim}D "
                        f"but pred_masks_target is {pred_masks_target.ndim}D"
                    )
                pred_masks_all = torch.cat(
                    [pred_masks_input, pred_masks_target], dim=0
                )  # [v_all, 1, 256, 256]
            else:
                # If no input masks, create zeros with same device as target
                pred_masks_all = torch.cat(
                    [
                        torch.zeros(v_input, 1, 256, 256, device=pred_masks_target.device),
                        pred_masks_target,
                    ],
                    dim=0,
                )  # [v_all, 1, 256, 256]

            # Upsample if needed and apply sigmoid
            if pred_masks_all.shape[-1] != 256:
                pred_masks_up = torch.nn.functional.interpolate(
                    pred_masks_all, size=(256, 256), mode="bilinear", align_corners=False
                )
            else:
                pred_masks_up = pred_masks_all

            pred_masks_prob = torch.sigmoid(pred_masks_up)  # [v_all, 1, 256, 256]

            # Create binary masks with threshold 0.1
            pred_masks_binary = (pred_masks_prob > 0.1).float()  # [v_all, 1, 256, 256]

            # Check if we have input patch masks (RE10K dropout / token dropping)
            has_patch_mask = (
                hasattr(result, "input_patch_mask") and result.input_patch_mask is not None
            )
            input_patch_mask_cov = None
            if hasattr(result, "input_patch_mask_coverage"):
                cov_tensor = result.input_patch_mask_coverage
                if cov_tensor is not None:
                    if cov_tensor.ndim == 3:  # [batch, ?, ?] unexpected
                        cov_tensor = cov_tensor.view(cov_tensor.shape[0], -1)
                    if cov_tensor.ndim == 2 and b < cov_tensor.shape[0]:
                        input_patch_mask_cov = cov_tensor[b].detach().cpu().numpy()  # [v_input]

            # Compute PSNR per target view
            psnrs = compute_psnr(target.image[b], result.render[b])  # [v_target]

            # Create figure: 4 columns for all views (input + target)
            fig, axes = plt.subplots(v_all, 4, figsize=(16, 4 * v_all))
            if v_all == 1:
                axes = axes.reshape(1, -1)

            cmap = cm.get_cmap("hot")

            for view_idx in range(v_all):
                is_input_view = view_idx < v_input
                target_view_idx = view_idx - v_input if not is_input_view else None

                # Get image and mask
                img_np = all_imgs[view_idx].permute(1, 2, 0).float().numpy()
                mask_soft_np = pred_masks_prob[view_idx, 0].float().numpy()
                mask_binary_np = pred_masks_binary[view_idx, 0].float().numpy()

                # Column 1: GT Image
                axes[view_idx, 0].imshow(img_np)
                if is_input_view:
                    axes[view_idx, 0].set_title(f"Input View {view_idx}", fontsize=12)
                else:
                    axes[view_idx, 0].set_title(f"Target View {target_view_idx}", fontsize=12)
                axes[view_idx, 0].axis("off")

                # Column 2: Soft Motion Mask Overlay
                axes[view_idx, 1].imshow(img_np)
                mask_colored = cmap(mask_soft_np)[:, :, :3]
                axes[view_idx, 1].imshow(mask_colored, alpha=0.5)
                axes[view_idx, 1].set_title("Soft Motion Mask", fontsize=12)
                axes[view_idx, 1].axis("off")

                # Column 3: Rendered (only for target views)
                if not is_input_view:
                    rendered_np = rendered_imgs[target_view_idx].permute(1, 2, 0).float().numpy()
                    psnr_val = psnrs[target_view_idx].item()
                    axes[view_idx, 2].imshow(rendered_np)
                    axes[view_idx, 2].set_title(
                        f"Rendered (PSNR: {psnr_val:.2f})",
                        fontsize=12,
                    )
                    axes[view_idx, 2].axis("off")
                else:
                    # Input view - show input image
                    axes[view_idx, 2].imshow(img_np)
                    axes[view_idx, 2].set_title("Input (No Render)", fontsize=12)
                    axes[view_idx, 2].axis("off")

                # Column 4: Input Dropout (for input views) OR Binary Motion Mask (for target views)
                if is_input_view and has_patch_mask:
                    # Show MAE dropout mask for input views
                    input_patch_mask_flat = (
                        result.input_patch_mask[b * v_input + view_idx].detach().cpu()
                    )  # [1, H_patch, W_patch]

                    # Upsample to full resolution with nearest neighbor
                    h, w = img_np.shape[:2]
                    patch_mask_up = torch.nn.functional.interpolate(
                        input_patch_mask_flat.unsqueeze(0), size=(h, w), mode="nearest"
                    ).squeeze(
                        0
                    )  # [1, h, w]

                    # Create red overlay for dropped patches
                    patch_mask_np = patch_mask_up[0].numpy()

                    red = np.zeros_like(img_np)
                    red[:, :, 0] = 1.0
                    alpha = 0.5
                    dropout_overlay = img_np * (1.0 - alpha * patch_mask_np[:, :, None]) + red * (
                        alpha * patch_mask_np[:, :, None]
                    )
                    dropout_overlay = np.clip(dropout_overlay, 0, 1)

                    axes[view_idx, 3].imshow(dropout_overlay)
                    if input_patch_mask_cov is not None:
                        coverage_val = float(input_patch_mask_cov[view_idx])
                        axes[view_idx, 3].set_title(
                            f"Token Drop (mean={coverage_val:.3f})", fontsize=12
                        )
                    else:
                        axes[view_idx, 3].set_title("Token Drop", fontsize=12)
                    axes[view_idx, 3].axis("off")
                else:
                    # Show binary motion mask (threshold=0.1) for target views
                    red = np.zeros_like(img_np)
                    red[:, :, 0] = 1.0
                    alpha = 0.5
                    binary_overlay = img_np * (1.0 - alpha * mask_binary_np[:, :, None]) + red * (
                        alpha * mask_binary_np[:, :, None]
                    )
                    binary_overlay = np.clip(binary_overlay, 0, 1)

                    axes[view_idx, 3].imshow(binary_overlay)
                    if is_input_view:
                        axes[view_idx, 3].set_title("Binary Mask (thresh=0.1)", fontsize=12)
                    else:
                        axes[view_idx, 3].set_title("Binary Loss Gate (thresh=0.1)", fontsize=12)
                    axes[view_idx, 3].axis("off")

            plt.tight_layout()
            save_path = os.path.join(sample_dir, "motion_mask_visualization.png")
            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            print(f"✓ Saved motion mask visualization: {save_path}")

            # Save motion mask as .npy for later analysis
            motion_mask_np = (
                pred_masks_prob.float().numpy()
            )  # [v_target, 1, 256, 256] - convert BFloat16 to Float32
            np.save(os.path.join(sample_dir, "motion_mask_soft.npy"), motion_mask_np)

            # Save individual GT and rendered images for each target view
            for view_idx in range(v_target):
                # Save GT image
                gt_img = target.image[b, view_idx].detach().cpu().float()  # [3, H, W] in [0, 1]
                gt_img_np = (gt_img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                gt_path = os.path.join(sample_dir, f"gt_view_{view_idx:02d}.png")
                Image.fromarray(gt_img_np).save(gt_path)

                # Save rendered image
                rendered_img = (
                    result.render[b, view_idx].detach().cpu().float()
                )  # [3, H, W] in [0, 1]
                rendered_img_np = (
                    (rendered_img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                )
                rendered_path = os.path.join(sample_dir, f"rendered_view_{view_idx:02d}.png")
                Image.fromarray(rendered_img_np).save(rendered_path)

                print(
                    f"✓ Saved {v_target} pairs of GT and rendered images: "
                    f"{sample_dir}/{{gt,rendered}}_view_{view_idx:02d}.png"
                )

        # Save random dropout mask for RE10K if available
        if hasattr(result, "input_patch_mask") and result.input_patch_mask is not None:
            v_input = input.image.shape[1]
            input_patch_mask = (
                result.input_patch_mask[b * v_input : (b + 1) * v_input].detach().cpu().numpy()
            )  # [v_input, 1, H_patch, W_patch]
            np.save(os.path.join(sample_dir, "input_patch_mask.npy"), input_patch_mask)
            print(f"✓ Saved input patch mask (MAE dropout): {sample_dir}/input_patch_mask.npy")

        # Save interpolated video if available (already rendered by model.render_images_video)
        # if hasattr(result, "video_rendering") and result.video_rendering is not None:
        #     video_frames = result.video_rendering[b].detach().cpu()  # [num_frames, 3, h, w]
        #     video_frames_np = (
        #         video_frames.permute(0, 2, 3, 1).float().numpy()
        #     )  # [num_frames, h, w, 3]
        #     video_frames_np = (video_frames_np * 255.0).clip(0, 255).astype(np.uint8)

        #     video_path = os.path.join(sample_dir, "interpolated_video.mp4")
        #     from utils import data_utils

        #     data_utils.create_video_from_frames(video_frames_np, video_path, framerate=30)
        #     print(f"✓ Saved interpolated video: {video_path}")


def _save_images(result, batch_idx, out_dir):
    """Save basic visualization composites to the visualization directory."""

    os.makedirs(out_dir, exist_ok=True)

    input_tensor = result.input.image[batch_idx]
    input_grid = rearrange(input_tensor, "v c h w -> h (v w) c").detach().cpu().numpy()
    input_grid = (input_grid * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_grid).save(os.path.join(out_dir, "input_grid.png"))

    comparison = (
        torch.cat((result.target.image[batch_idx], result.render[batch_idx]), dim=2).detach().cpu()
    )
    comparison = rearrange(comparison, "v c h w -> h (v w) c").numpy()
    comparison = (comparison * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(comparison).save(os.path.join(out_dir, "gt_vs_rendered.png"))


def _save_metrics(target, prediction, view_indices, out_dir, scene_name):
    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)

    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean()),
        },
        "per_view": [],
    }

    for i, view_idx in enumerate(view_indices):
        metrics["per_view"].append(
            {
                "view": int(view_idx),
                "psnr": float(psnr_values[i]),
                "lpips": float(lpips_values[i]),
                "ssim": float(ssim_values[i]),
            }
        )

    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _extract_predicted_masks(result, batch_idx, v_input, v_target):
    pred_input = None
    pred_target = None

    if hasattr(result, "predicted_target_masks") and result.predicted_target_masks is not None:
        masks = result.predicted_target_masks
        if masks.ndim == 5:
            pred_target = masks[batch_idx].detach().cpu().float()
        elif masks.ndim == 4 and v_target > 0:
            start = batch_idx * v_target
            pred_target = masks[start : start + v_target].detach().cpu().float()

    if hasattr(result, "predicted_input_masks") and result.predicted_input_masks is not None:
        masks_in = result.predicted_input_masks
        if masks_in.ndim == 5:
            pred_input = masks_in[batch_idx].detach().cpu().float()
        elif masks_in.ndim == 4 and v_input > 0:
            start = batch_idx * v_input
            pred_input = masks_in[start : start + v_input].detach().cpu().float()

    return pred_input, pred_target


def _save_video(frames, output_path):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = np.ascontiguousarray(np.array(frames.to(torch.float32).detach().cpu()))
    frames = rearrange(frames, "v c h w -> v h w c")
    data_utils.create_video_from_frames(frames, output_path, framerate=30)


def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: (
            not os.path.basename(x).isdigit(),  # False for numeric (sort first), True for strings
            int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x),
        ),
    )

    metrics = {}
    valid_subfolders = []

    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue

        valid_subfolders.append(subfolder)

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")

        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")

        f.write("\n")

        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")

    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")

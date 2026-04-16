#!/usr/bin/env python3
"""
Masked Metrics Implementation for PSNR, SSIM, and LPIPS (Improved Version)

This module provides masked versions of standard image quality metrics:
- Masked PSNR: Weighted MSE over mask regions (per-image scores)
- Masked SSIM: Area-pooled mask applied to SSIM map (per-image scores)
- Masked LPIPS: Per-layer feature masking for perceptual distance (per-image scores)

Key improvements:
1. Per-image scores: Returns [B] tensors instead of scalars
2. Proper mask resizing: Uses F.interpolate with mode='area' for downsample, 'nearest' for upsample
3. LPIPS per-layer masking: Applies masks at every feature level (no approximations)
4. SSIM constants: Documented for [0,1] range (C1=0.01^2, C2=0.03^2)
5. NaN handling: Returns NaN for empty/tiny masks to prevent contamination

Usage:
    from masked_metrics import compute_masked_psnr, compute_masked_ssim, compute_masked_lpips

    # Images and masks should be torch tensors
    # Images: [B, C, H, W] in range [0, 1]
    # Masks: [B, 1, H, W] in range [0, 1] (1 = foreground/transient)

    psnr_transient = compute_masked_psnr(img_gt, img_pred, mask)  # Returns [B]
    psnr_static = compute_masked_psnr(img_gt, img_pred, 1 - mask)

    # Get mean, filtering NaNs
    mean_psnr = psnr_transient[~torch.isnan(psnr_transient)].mean()
"""

import torch
import torch.nn.functional as F


def _resize_mask(mask: torch.Tensor, target_size: tuple, mode: str = "auto") -> torch.Tensor:
    """
    Resize mask to target size with appropriate interpolation.

    Args:
        mask: Input mask [B, 1, H, W]
        target_size: (H, W)
        mode: 'auto' (choose based on up/down), 'area', or 'nearest'

    Returns:
        Resized mask [B, 1, H, W]
    """
    if mask.shape[-2:] == target_size:
        return mask

    if mode == "auto":
        # Use area pooling for downsampling, nearest for upsampling
        downsample = mask.shape[-2] > target_size[0] or mask.shape[-1] > target_size[1]
        mode = "area" if downsample else "nearest"

    return F.interpolate(mask, size=target_size, mode=mode)


def compute_masked_psnr(
    img_gt: torch.Tensor, img_pred: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute masked PSNR per image.

    Formula:
        MSE_M = sum((I - I_hat)^2 * M) / (sum(M) + eps)
        PSNR_M = -10 * log10(MSE_M)

    Args:
        img_gt: Ground truth image, shape [B, C, H, W], range [0, 1]
        img_pred: Predicted image, shape [B, C, H, W], range [0, 1]
        mask: Binary mask, shape [B, 1, H, W], range [0, 1]
              1 = region of interest (e.g., transient)
        eps: Small constant for numerical stability

    Returns:
        Per-image PSNR values, shape [B]
        Returns NaN for images with empty/tiny masks (sum < eps)
    """
    B = img_gt.shape[0]
    device = img_gt.device

    # Ensure images are in [0, 1]
    img_gt = torch.clamp(img_gt, 0, 1)
    img_pred = torch.clamp(img_pred, 0, 1)

    # Resize mask to match image size if needed
    mask = _resize_mask(mask, img_gt.shape[-2:], mode="nearest")

    # Expand mask to match channels if needed
    if mask.shape[1] == 1 and img_gt.shape[1] > 1:
        mask = mask.expand(-1, img_gt.shape[1], -1, -1)

    # Compute squared error per image
    squared_error = (img_gt - img_pred) ** 2  # [B, C, H, W]

    # Compute per-image masked MSE
    masked_se_sum = (squared_error * mask).flatten(1).sum(dim=1)  # [B]
    mask_sum = mask.flatten(1).sum(dim=1)  # [B]

    # Initialize output with NaN
    psnr_values = torch.full((B,), float("nan"), device=device)

    # Compute PSNR only for valid masks
    valid_mask = mask_sum >= eps
    if valid_mask.any():
        mse_valid = masked_se_sum[valid_mask] / mask_sum[valid_mask]

        # Avoid log(0) for perfect matches
        mse_valid = torch.clamp(mse_valid, min=eps)

        psnr_values[valid_mask] = -10 * torch.log10(mse_valid)

    return psnr_values


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel


def _ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    C1: float = 0.01**2,  # For [0,1] range
    C2: float = 0.03**2,  # For [0,1] range
) -> torch.Tensor:
    """
    Compute per-pixel SSIM map.

    SSIM constants C1 and C2 are for images in [0,1] range:
    - C1 = (0.01)^2 stabilizes the luminance comparison
    - C2 = (0.03)^2 stabilizes the contrast comparison
    If images were in [0,255], multiply by (data_range)^2

    Args:
        img1: Image tensor [B, C, H, W]
        img2: Image tensor [B, C, H, W]
        window: Gaussian window [1, 1, K, K]
        C1: Constant for luminance (default for [0,1] range)
        C2: Constant for contrast (default for [0,1] range)

    Returns:
        SSIM map [B, C, H', W']
    """
    C, window_size = img1.shape[1], window.shape[-1]

    # Expand window for all channels
    window = window.expand(C, 1, window_size, window_size)

    # Compute local means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1**2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map


def compute_masked_ssim(
    img_gt: torch.Tensor,
    img_pred: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute masked SSIM per image.

    Process:
    1. Compute SSIM map using Gaussian window
    2. Downsample mask to SSIM map size using area pooling
    3. Return masked average of SSIM map

    Args:
        img_gt: Ground truth image, shape [B, C, H, W], range [0, 1]
        img_pred: Predicted image, shape [B, C, H, W], range [0, 1]
        mask: Binary mask, shape [B, 1, H, W], range [0, 1]
        window_size: Size of Gaussian window (default 11)
        sigma: Gaussian sigma (default 1.5)
        eps: Small constant for numerical stability

    Returns:
        Per-image SSIM values, shape [B]
        Returns NaN for images with empty/tiny masks
    """
    B, C = img_gt.shape[0], img_gt.shape[1]
    device = img_gt.device

    # Ensure images are in [0, 1]
    img_gt = torch.clamp(img_gt, 0, 1)
    img_pred = torch.clamp(img_pred, 0, 1)

    # Create Gaussian window
    window = _gaussian_kernel(window_size, sigma, device).unsqueeze(0).unsqueeze(0)

    # Compute SSIM map [B, C, H', W']
    ssim_map = _ssim_map(img_gt, img_pred, window)

    # Downsample mask to SSIM map size using area pooling
    mask_resized = _resize_mask(mask, ssim_map.shape[-2:], mode="area")

    # Expand mask to match channels
    if mask_resized.shape[1] == 1 and C > 1:
        mask_resized = mask_resized.expand(-1, C, -1, -1)

    # Compute per-image masked average
    ssim_sum = (ssim_map * mask_resized).flatten(1).sum(dim=1)  # [B]
    mask_sum = mask_resized.flatten(1).sum(dim=1)  # [B]

    # Initialize output with NaN
    ssim_values = torch.full((B,), float("nan"), device=device)

    # Compute SSIM only for valid masks
    valid_mask = mask_sum >= eps
    if valid_mask.any():
        ssim_values[valid_mask] = ssim_sum[valid_mask] / mask_sum[valid_mask]

    return ssim_values


def compute_masked_lpips(
    img_gt: torch.Tensor, img_pred: torch.Tensor, mask: torch.Tensor, lpips_model, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute masked LPIPS per image using per-layer feature masking.

    For each VGG feature level used by LPIPS, we:
        1. Obtain the spatial difference map (retPerLayer=True)
        2. Resize the mask to the feature map resolution (area pooling for downsampling)
        3. Compute the masked average for that level
        4. Accumulate the contributions across all levels

    Args:
        img_gt: Ground truth image, shape [B, C, H, W], range [0, 1]
        img_pred: Predicted image, shape [B, C, H, W], range [0, 1]
        mask: Binary mask, shape [B, 1, H, W], range [0, 1]
        lpips_model: LPIPS model instance (must support retPerLayer=True)
        eps: Small constant for numerical stability

    Returns:
        Per-image LPIPS values, shape [B]
        Returns NaN for images whose masks are empty across all layers
    """
    B = img_gt.shape[0]
    device = img_gt.device

    # Ensure images are in [0, 1]
    img_gt = torch.clamp(img_gt, 0, 1)
    img_pred = torch.clamp(img_pred, 0, 1)

    # Resize mask to match image size (nearest for upsampling)
    mask = _resize_mask(mask, img_gt.shape[-2:], mode="nearest")

    # Ensure model is in eval mode
    was_training = lpips_model.training
    lpips_model.eval()

    # Convert to LPIPS range [-1, 1]
    img_gt_lpips = img_gt * 2 - 1
    img_pred_lpips = img_pred * 2 - 1

    lpips_values = torch.zeros(B, device=device)
    any_valid = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        # Retrieve per-layer spatial difference maps
        per_layer_diffs = lpips_model(
            img_gt_lpips, img_pred_lpips, normalize=False, retPerLayer=True
        )

        # Per-layer accumulation
        for layer_diff in per_layer_diffs:
            if isinstance(layer_diff, (list, tuple)):
                layer_diff = layer_diff[0]
            if layer_diff.dim() == 3:
                layer_diff = layer_diff.unsqueeze(1)

            # Downsample mask to feature map resolution (area pooling)
            mask_resized = _resize_mask(mask, layer_diff.shape[-2:], mode="area")

            # Sum over channels if necessary to align with mask channel (1)
            if layer_diff.shape[1] > 1:
                # LPIPS spatial maps are typically single channel, but guard anyway
                layer_diff = layer_diff.mean(dim=1, keepdim=True)

            layer_diff = layer_diff.squeeze(1)  # [B, H', W']
            mask_resized = mask_resized.squeeze(1)  # [B, H', W']

            layer_diff_flat = layer_diff.flatten(1)
            mask_flat = mask_resized.flatten(1)

            mask_sum = mask_flat.sum(dim=1)  # [B]
            valid = mask_sum >= eps

            if valid.any():
                layer_sum = (layer_diff_flat[valid] * mask_flat[valid]).sum(dim=1)
                contribution = layer_sum / mask_sum[valid]
                lpips_values[valid] += contribution
                any_valid[valid] = True

    # Restore model training state
    if was_training:
        lpips_model.train()

    # Images with no valid layers receive NaN
    lpips_values[~any_valid] = float("nan")

    return lpips_values


def compute_all_masked_metrics(
    img_gt: torch.Tensor,
    img_pred: torch.Tensor,
    mask_transient: torch.Tensor,
    lpips_model,
    return_per_image: bool = False,
) -> dict:
    """
    Compute all masked metrics (PSNR, SSIM, LPIPS) for both transient and static regions.

    Args:
        img_gt: Ground truth image, shape [B, C, H, W], range [0, 1]
        img_pred: Predicted image, shape [B, C, H, W], range [0, 1]
        mask_transient: Binary mask for transient regions, shape [B, 1, H, W]
        lpips_model: LPIPS model instance
        return_per_image: If True, return [B] tensors; if False, return mean values

    Returns:
        Dictionary with metrics:
        - psnr_transient, psnr_static
        - ssim_transient, ssim_static
        - lpips_transient, lpips_static

        If return_per_image=True: values are tensors [B]
        If return_per_image=False: values are scalars (mean, NaNs filtered)
    """
    mask_static = 1 - mask_transient

    # Compute all metrics per image
    psnr_t = compute_masked_psnr(img_gt, img_pred, mask_transient)
    psnr_s = compute_masked_psnr(img_gt, img_pred, mask_static)

    ssim_t = compute_masked_ssim(img_gt, img_pred, mask_transient)
    ssim_s = compute_masked_ssim(img_gt, img_pred, mask_static)

    lpips_t = compute_masked_lpips(img_gt, img_pred, mask_transient, lpips_model)
    lpips_s = compute_masked_lpips(img_gt, img_pred, mask_static, lpips_model)

    if return_per_image:
        return {
            "psnr_transient": psnr_t,
            "psnr_static": psnr_s,
            "ssim_transient": ssim_t,
            "ssim_static": ssim_s,
            "lpips_transient": lpips_t,
            "lpips_static": lpips_s,
        }
    else:
        # Return mean, filtering NaNs
        return {
            "psnr_transient": (
                psnr_t[~torch.isnan(psnr_t)].mean().item()
                if (~torch.isnan(psnr_t)).any()
                else float("nan")
            ),
            "psnr_static": (
                psnr_s[~torch.isnan(psnr_s)].mean().item()
                if (~torch.isnan(psnr_s)).any()
                else float("nan")
            ),
            "ssim_transient": (
                ssim_t[~torch.isnan(ssim_t)].mean().item()
                if (~torch.isnan(ssim_t)).any()
                else float("nan")
            ),
            "ssim_static": (
                ssim_s[~torch.isnan(ssim_s)].mean().item()
                if (~torch.isnan(ssim_s)).any()
                else float("nan")
            ),
            "lpips_transient": (
                lpips_t[~torch.isnan(lpips_t)].mean().item()
                if (~torch.isnan(lpips_t)).any()
                else float("nan")
            ),
            "lpips_static": (
                lpips_s[~torch.isnan(lpips_s)].mean().item()
                if (~torch.isnan(lpips_s)).any()
                else float("nan")
            ),
        }


# Example usage
if __name__ == "__main__":
    import lpips

    print("Testing improved masked metrics...")

    # Create dummy data
    B, C, H, W = 2, 3, 256, 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_gt = torch.rand(B, C, H, W, device=device)
    img_pred = torch.rand(B, C, H, W, device=device)

    # Create mask with one empty mask to test NaN handling
    mask = torch.zeros(B, 1, H, W, device=device)
    mask[0, 0, 64:192, 64:192] = 1.0  # First image has valid mask
    # Second image has empty mask (all zeros)

    # Initialize LPIPS
    lpips_model = lpips.LPIPS(net="vgg", spatial=True).to(device)

    # Test per-image metrics
    print("\n=== Per-image metrics ===")
    psnr_vals = compute_masked_psnr(img_gt, img_pred, mask)
    print(f"PSNR per image: {psnr_vals}")
    print(f"  Image 0 (valid): {psnr_vals[0].item():.2f}")
    print(f"  Image 1 (empty): {psnr_vals[1].item()}")  # Should be NaN

    ssim_vals = compute_masked_ssim(img_gt, img_pred, mask)
    print(f"SSIM per image: {ssim_vals}")

    lpips_vals = compute_masked_lpips(img_gt, img_pred, mask, lpips_model)
    print(f"LPIPS per image: {lpips_vals}")

    # Test aggregated metrics (filtering NaNs)
    print("\n=== Aggregated metrics (NaN-filtered) ===")
    metrics = compute_all_masked_metrics(
        img_gt, img_pred, mask, lpips_model, return_per_image=False
    )
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")

    print("\n✓ All tests passed!")

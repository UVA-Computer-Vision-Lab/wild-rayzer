# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

from einops import rearrange
import lpips
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision.models import vgg19
import scipy.io
import os
from pathlib import Path


# the perception loss code is modified from https://github.com/zhengqili/Crowdsampling-the-Plenoptic-Function/blob/f5216f312cf82d77f8d20454b5eeb3930324630a/models/networks.py#L1478
# and some parts are based on https://github.com/arthurhero/Long-LRM/blob/main/model/loss.py
class PerceptualLoss(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.vgg = self._build_vgg()
        self._load_weights()
        self._setup_feature_blocks()

    def _build_vgg(self):
        """Create VGG model with average pooling instead of max pooling."""
        model = vgg19()
        # Replace max pooling with average pooling
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.MaxPool2d):
                model.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)

        return model.to(self.device).eval()

    def _load_weights(self):
        """Load pre-trained VGG weights."""
        weight_file = Path("./metric_checkpoint/imagenet-vgg-verydeep-19.mat")
        weight_file.parent.mkdir(exist_ok=True, parents=True)

        # Check if distributed training is initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                # Download weights if needed
                if not weight_file.exists():
                    os.system(
                        f"wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}"
                    )
            torch.distributed.barrier()
        else:
            # Single GPU mode - just download if needed
            if not weight_file.exists():
                os.system(
                    f"wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -O {weight_file}"
                )

        # Load MatConvNet weights
        vgg_data = scipy.io.loadmat(weight_file)
        vgg_layers = vgg_data["layers"][0]

        # Layer indices and filter sizes
        layer_indices = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        filter_sizes = [
            64,
            64,
            128,
            128,
            256,
            256,
            256,
            256,
            512,
            512,
            512,
            512,
            512,
            512,
            512,
            512,
        ]

        # Transfer weights to PyTorch model
        with torch.no_grad():
            for i, layer_idx in enumerate(layer_indices):
                # Set weights
                weights = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][0]).permute(3, 2, 0, 1)
                self.vgg.features[layer_idx].weight = nn.Parameter(weights, requires_grad=False)

                # Set biases
                biases = torch.from_numpy(vgg_layers[layer_idx][0][0][2][0][1]).view(
                    filter_sizes[i]
                )
                self.vgg.features[layer_idx].bias = nn.Parameter(biases, requires_grad=False)

    def _setup_feature_blocks(self):
        """Create feature extraction blocks at different network depths."""
        output_indices = [0, 4, 9, 14, 23, 32]
        self.blocks = nn.ModuleList()

        # Create sequential blocks
        for i in range(len(output_indices) - 1):
            block = nn.Sequential(
                *list(self.vgg.features[output_indices[i] : output_indices[i + 1]])
            )
            self.blocks.append(block.to(self.device).eval())

        # Freeze all parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def _extract_features(self, x):
        """Extract features from each block."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def _preprocess_images(self, images):
        """Convert images to VGG input format."""
        # VGG mean values for ImageNet
        mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1).to(images.device)
        return images * 255.0 - mean

    @staticmethod
    def _compute_error(real, fake):
        return torch.mean(torch.abs(real - fake))

    @staticmethod
    def _compute_error_with_mask(real, fake, mask):
        return torch.mean(torch.abs(real - fake) * mask)

    def forward(self, pred_img, target_img, mask=None):
        """Compute perceptual loss between prediction and target."""
        # Preprocess images
        target_img_p = self._preprocess_images(target_img)
        pred_img_p = self._preprocess_images(pred_img)

        # Extract features
        target_features = self._extract_features(target_img_p)
        pred_features = self._extract_features(pred_img_p)

        # Pixel-level error
        if mask is not None:

            def m_at(feat):
                if mask.shape[-2:] != feat.shape[-2:]:
                    # Use nearest to preserve binary mask (avoid soft edges from bilinear)
                    return F.interpolate(mask, size=feat.shape[-2:], mode="nearest")
                return mask

            e0 = self._compute_error_with_mask(target_img_p, pred_img_p, m_at(target_img_p))
            e1 = (
                self._compute_error_with_mask(
                    target_features[0], pred_features[0], m_at(target_features[0])
                )
                / 2.6
            )
            e2 = (
                self._compute_error_with_mask(
                    target_features[1], pred_features[1], m_at(target_features[1])
                )
                / 4.8
            )
            e3 = (
                self._compute_error_with_mask(
                    target_features[2], pred_features[2], m_at(target_features[2])
                )
                / 3.7
            )
            e4 = (
                self._compute_error_with_mask(
                    target_features[3], pred_features[3], m_at(target_features[3])
                )
                / 5.6
            )
            e5 = (
                self._compute_error_with_mask(
                    target_features[4], pred_features[4], m_at(target_features[4])
                )
                / 1.5
                * 10
            )
        else:
            e0 = self._compute_error(target_img_p, pred_img_p)
            e1 = self._compute_error(target_features[0], pred_features[0]) / 2.6
            e2 = self._compute_error(target_features[1], pred_features[1]) / 4.8
            e3 = self._compute_error(target_features[2], pred_features[2]) / 3.7
            e4 = self._compute_error(target_features[3], pred_features[3]) / 5.6
            e5 = self._compute_error(target_features[4], pred_features[4]) * 10 / 1.5
        # Feature-level errors with scaling factors

        # Combine all errors and normalize
        total_loss = (e0 + e1 + e2 + e3 + e4 + e5) / 255.0

        return total_loss


class LossComputer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # PSNR filtering: skip samples with PSNR below threshold
        self.min_psnr_for_loss = self.config.training.get("min_psnr_for_loss", 0.0)
        if self.min_psnr_for_loss > 0:
            print(
                f"[Loss] PSNR filtering enabled: Skip samples with PSNR < {self.min_psnr_for_loss} dB"
            )

        if self.config.training.lpips_loss_weight > 0.0:
            # avoid multiple GPUs from downloading the same LPIPS model multiple times
            if torch.distributed.get_rank() == 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0:
                self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
        if self.config.training.perceptual_loss_weight > 0.0:
            self.perceptual_loss_module = self._init_frozen_module(PerceptualLoss())

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def forward(
        self,
        rendering,
        target,
        input=None,
        create_visual=False,
        motion_mask=None,
        predicted_masks_all=None,
        pseudolabels_all=None,
        current_step=0,
        dataset_sources=None,
    ):
        """
        Calculate rendering losses and optional motion mask distillation loss.

        Args:
            rendering: [b, v, 3, h, w], value range [0, 1]
            target: [b, v, 3, h, w], value range [0, 1]
            input: unused, kept for compatibility
            create_visual: unused, kept for compatibility
            motion_mask: unused (rendering losses are disabled during motion-mask-only training)
            predicted_masks_all: [b*v, 1, H, W] predicted motion mask logits (for distillation)
            pseudolabels_all: [b*v, 1, H, W] DINOv3 pseudo-label masks (for distillation)
            current_step: unused, kept for future extensions
            dataset_sources: list of dataset sources (e.g., ["re10k", "dre10k", ...]) for per-sample handling

        Returns:
            Dictionary of loss metrics
        """
        b, v, _, h, w = rendering.size()
        device = rendering.device

        # PSNR filtering: skip samples with catastrophically bad renderings (except RE10K)
        if self.min_psnr_for_loss > 0:
            rendering_flat = rendering.reshape(b * v, -1, h, w)
            target_flat = target.reshape(b * v, -1, h, w)

            # Compute PSNR per sample
            with torch.no_grad():
                mse_per_sample = ((rendering_flat - target_flat) ** 2).mean(dim=[1, 2, 3])  # [b*v]
                psnr_per_sample = -10.0 * torch.log10(mse_per_sample.clamp(min=1e-10))

                # Create mask for RE10K samples (always include in loss)
                if dataset_sources is not None:
                    is_re10k = torch.tensor(
                        [dataset_sources[i // v] == "re10k" for i in range(b * v)],
                        device=device,
                        dtype=torch.float32,
                    )  # [b*v]
                else:
                    is_re10k = torch.zeros(b * v, device=device, dtype=torch.float32)

                # # Visualization: Save some samples to debug poor rendering quality
                # if not hasattr(self, '_viz_saved'):
                #     self._viz_saved = True
                #     import os
                #     from PIL import Image
                #     import numpy as np

                #     viz_dir = "./experiments/evaluation"
                #     os.makedirs(viz_dir, exist_ok=True)

                #     # Save first 8 samples (or fewer if batch is smaller)
                #     num_viz = min(8, rendering_flat.shape[0])
                #     for i in range(num_viz):
                #         # Get rendering and target for this sample
                #         render_img = rendering_flat[i].cpu().float().numpy()  # [3, h, w]
                #         target_img = target_flat[i].cpu().float().numpy()  # [3, h, w]
                #         psnr_val = psnr_per_sample[i].item()

                #         # Convert to [h, w, 3] and clip to [0, 1]
                #         render_img = np.transpose(render_img, (1, 2, 0)).clip(0, 1)
                #         target_img = np.transpose(target_img, (1, 2, 0)).clip(0, 1)

                #         # Convert to uint8
                #         render_img = (render_img * 255).astype(np.uint8)
                #         target_img = (target_img * 255).astype(np.uint8)

                #         # Create side-by-side comparison
                #         h, w = render_img.shape[:2]
                #         comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
                #         comparison[:, :w] = target_img
                #         comparison[:, w:] = render_img

                #         # Save
                #         img_path = os.path.join(viz_dir, f"debug_sample_{i:02d}_psnr_{psnr_val:.2f}.png")
                #         Image.fromarray(comparison).save(img_path)
                #         print(f"[DEBUG] Saved comparison to {img_path} (PSNR: {psnr_val:.2f} dB)")

                #     print(f"[DEBUG] PSNR stats: min={psnr_per_sample.min().item():.2f}, "
                #           f"max={psnr_per_sample.max().item():.2f}, "
                #           f"mean={psnr_per_sample.mean().item():.2f} dB")
                # Filter: keep samples with PSNR >= threshold OR RE10K samples (always keep)
                psnr_valid = (psnr_per_sample >= self.min_psnr_for_loss).float()  # [b*v]
                valid_mask = torch.clamp(
                    psnr_valid + is_re10k, 0.0, 1.0
                )  # Union: PSNR valid OR RE10K
                num_valid = int(valid_mask.sum().item())

                # If all samples are invalid, return zero loss (skip this batch)
                # Create a leaf tensor with requires_grad for proper gradient flow
                if num_valid == 0:
                    zero_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
                    return edict(
                        loss=zero_loss,
                        l2_loss=torch.tensor(0.0, device=device),
                        psnr=torch.tensor(0.0, device=device),
                        lpips_loss=torch.tensor(0.0, device=device),
                        perceptual_loss=torch.tensor(0.0, device=device),
                        psnr_filtered_samples=torch.tensor(0, device=device),
                        psnr_total_samples=torch.tensor(b * v, device=device),
                    )

            # Reshape valid_mask for indexing: [b*v] -> [b, v]
            valid_mask_bv = valid_mask.reshape(b, v)

            # Gather valid samples
            valid_indices = torch.nonzero(valid_mask_bv, as_tuple=False)  # [num_valid, 2]
            rendering = rendering[valid_indices[:, 0], valid_indices[:, 1]].unsqueeze(
                0
            )  # [1, num_valid, 3, h, w]
            target = target[valid_indices[:, 0], valid_indices[:, 1]].unsqueeze(0)

            # Also filter predicted_masks_all and pseudolabels_all if provided
            if predicted_masks_all is not None:
                # valid_mask: [b*v] -> filter [b*v, 1, H, W]
                predicted_masks_all = predicted_masks_all[valid_mask.bool()]  # [num_valid, 1, H, W]
            if pseudolabels_all is not None:
                pseudolabels_all = pseudolabels_all[valid_mask.bool()]

            # Update batch size for subsequent calculations
            b, v = 1, num_valid
        else:
            num_valid = b * v

        # Compute rendering losses (will be 0 if weights are 0)
        rendering_flat = rendering.reshape(b * v, -1, h, w)
        target_flat = target.reshape(b * v, -1, h, w)

        # Handle alpha channel if present
        if target_flat.size(1) == 4:
            target_flat, _ = target_flat.split([3, 1], dim=1)

        # Create binary static mask from predicted motion masks (1=static, 0=motion)
        static_mask = None
        if predicted_masks_all is not None:
            threshold = self.config.model.get("motion_mask_threshold", 0.1)
            motion_prob = torch.sigmoid(predicted_masks_all)  # [b*v, 1, H, W]
            static_mask = (motion_prob <= threshold).float()  # Inverse: 1=static, 0=motion

            # Upsample to match rendering resolution if needed
            assert static_mask.shape[-2:] == (h, w)
        l2_loss = torch.tensor(1e-8).to(device)
        if self.config.training.l2_loss_weight > 0.0:
            if static_mask is not None:
                # Masked MSE: only compute on static regions
                diff_sq = (rendering_flat - target_flat) ** 2 * static_mask
                l2_loss = diff_sq.sum() / static_mask.sum().clamp(min=1.0)
            else:
                l2_loss = F.mse_loss(rendering_flat, target_flat)

        psnr = -10.0 * torch.log10(l2_loss.clamp(min=1e-10))

        lpips_loss = torch.tensor(0.0).to(device)
        if self.config.training.lpips_loss_weight > 0.0:
            # Scale from [0,1] to [-1,1] as required by LPIPS
            lpips_loss = self.lpips_loss_module(
                rendering_flat * 2.0 - 1.0, target_flat * 2.0 - 1.0
            ).mean()

        perceptual_loss = torch.tensor(0.0).to(device)
        if self.config.training.perceptual_loss_weight > 0.0:
            perceptual_loss = self.perceptual_loss_module(
                rendering_flat, target_flat, mask=static_mask
            )

        loss = (
            self.config.training.l2_loss_weight * l2_loss
            + self.config.training.lpips_loss_weight * lpips_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
        )

        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            norm_perceptual_loss=perceptual_loss / l2_loss.clamp(min=1e-10),
            norm_lpips_loss=lpips_loss / l2_loss.clamp(min=1e-10),
        )

        # Add PSNR filtering stats if enabled
        if self.min_psnr_for_loss > 0:
            loss_metrics.psnr_filtered_samples = torch.tensor(num_valid, device=device)
            loss_metrics.psnr_total_samples = torch.tensor(
                rendering.shape[0] * rendering.shape[1], device=device
            )  # original b*v

        # Motion mask distillation loss (simple BCE)
        if predicted_masks_all is not None and pseudolabels_all is not None:
            # Upsample predicted masks to match pseudo-label resolution
            # predicted_masks_all: [num_valid, 1, H_pred, W_pred] -> [num_valid, 1, H_target, W_target]
            target_size = pseudolabels_all.shape[-2:]  # e.g., (256, 256)
            if predicted_masks_all.shape[-2:] != target_size:
                predicted_masks_upsampled = F.interpolate(
                    predicted_masks_all, size=target_size, mode="bilinear", align_corners=False
                )
            else:
                predicted_masks_upsampled = predicted_masks_all

            # BCE loss between predicted logits and DINOv3 pseudo-labels
            mask_bce_loss = F.binary_cross_entropy_with_logits(
                predicted_masks_upsampled, pseudolabels_all
            )

            # Get distillation weight from config
            mask_distill_weight = self.config.training.get("mask_distill_loss_weight", 1.0)

            # Add to total loss
            loss_metrics.loss = loss_metrics.loss + mask_distill_weight * mask_bce_loss
            loss_metrics.mask_distill_loss = mask_bce_loss
            loss_metrics.mask_distill_weight = torch.tensor(mask_distill_weight, device=device)

        return loss_metrics


# ==============================================================================
# Loss Computer for Official RAYZAR Model
# ==============================================================================


class LossComputer_official(nn.Module):
    """
    Loss computer for the official RAYZAR model (Images2Latent4D).
    Supports L2, LPIPS, Perceptual, and SSIM losses.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # loss weight for input views
        self.weight_l2 = self.config.training.get("l2_loss_weight", 1.0)
        self.weight_lpips = self.config.training.get("lpips_loss_weight", 0.0)
        self.weight_perceptual = self.config.training.get("perceptual_loss_weight", 0.0)
        self.weight_ssim = self.config.training.get("ssim_loss_weight", 0.0)
        self.weight_clip = self.config.training.get("clip_loss_weight", 0.0)

        # Motion mask distillation weight
        self.weight_mask_distill = self.config.training.get("mask_distill_loss_weight", 0.0)

        # PSNR filtering: skip samples with PSNR below threshold for mask distillation
        self.psnr_filter_threshold = self.config.training.get("psnr_filter_threshold", 0.0)

        # Use predicted motion masks to mask out dynamic regions from reconstruction loss
        self.use_masked_reconstruction_loss = self.config.training.get(
            "use_masked_reconstruction_loss", False
        )

        # Motion mask threshold for reconstruction loss masking
        self.motion_mask_threshold = self.config.training.get("motion_mask_threshold", 0.1)

        # loss weight for interpolation views
        self.weight_l2_interpolate = self.config.training.get("l2_loss_weight_interpolate", 0.0)
        self.weight_lpips_interpolate = self.config.training.get(
            "lpips_loss_weight_interpolate", 0.0
        )
        self.weight_perceptual_interpolate = self.config.training.get(
            "perceptual_loss_weight_interpolate", 0.0
        )
        self.weight_ssim_interpolate = self.config.training.get("ssim_loss_weight_interpolate", 0.0)
        self.weight_clip_interpolate = self.config.training.get("clip_loss_weight_interpolate", 0.0)

        # loss weight for code swapping (contrastive loss)
        self.weight_contrastive = self.config.training.get("contrastive_loss_swapping_weight", 0.0)

        if self.weight_lpips > 0.0 or self.weight_lpips_interpolate > 0.0:
            self.lpips_loss_module = lpips.LPIPS(net="vgg")
            self.lpips_loss_module.eval()
            # freeze the lpips loss module
            for param in self.lpips_loss_module.parameters():
                param.requires_grad = False

        if self.weight_perceptual > 0.0 or self.weight_perceptual_interpolate > 0.0:
            self.perceptual_loss_module = PerceptualLoss()
            self.perceptual_loss_module.eval()
            # freeze the perceptual loss module
            for param in self.perceptual_loss_module.parameters():
                param.requires_grad = False

        self.supervise_interpolate = config.training.get("supervise_interpolate", False)
        self.render_interpolate = config.training.get("render_interpolate", False)
        if self.supervise_interpolate:
            assert self.render_interpolate, "Must render interpolated images for supervising them"

    def calculate_loss(
        self, rendering, target, create_visual=True, is_interpolate=False, static_mask=None
    ):
        """
        rendering: [b, v, 3, h, w]; in range (0, 1)
        target: [b, v, 3, h, w]; in range (0, 1)
        static_mask: [b*v, 1, h, w]; binary mask where 1=static (compute loss), 0=dynamic (ignore)
        """
        b, v, _, h, w = rendering.size()
        rendering = rendering.reshape(b * v, -1, h, w)  # [b*v,c,h,w]
        target = target.reshape(b * v, -1, h, w)

        if is_interpolate:
            weight_l2 = self.weight_l2_interpolate
            weight_lpips = self.weight_lpips_interpolate
            weight_perceptual = self.weight_perceptual_interpolate
            weight_ssim = self.weight_ssim_interpolate
            weight_clip = self.weight_clip_interpolate
        else:
            weight_l2 = self.weight_l2
            weight_lpips = self.weight_lpips
            weight_perceptual = self.weight_perceptual
            weight_ssim = self.weight_ssim
            weight_clip = self.weight_clip

        mask = None
        if target.size(1) == 4:
            target, mask = target.split([3, 1], dim=1)

        # Compute masked or unmasked L2 loss
        l2_loss = torch.tensor(1e-8).to(rendering.device)
        with torch.set_grad_enabled(weight_l2 > 0.0):
            if static_mask is not None:
                # Upsample mask to match rendering resolution if needed (use nearest to preserve binary)
                if static_mask.shape[-2:] != (h, w):
                    static_mask_resized = F.interpolate(static_mask, size=(h, w), mode="nearest")
                else:
                    static_mask_resized = static_mask

                # Check if we have enough static pixels (avoid divide-by-tiny instability)
                num_static_pixels = static_mask_resized.sum()
                min_static_pixels = (
                    0.1 * b * v * h * w
                )  # At least 10% of total pixels should be static

                if num_static_pixels < min_static_pixels:
                    # Too few static pixels - fall back to unmasked loss to avoid instability
                    l2_loss = F.mse_loss(rendering, target)
                else:
                    # Masked L2 loss: only compute on static regions
                    mse_per_pixel = (rendering - target) ** 2  # [b*v, 3, h, w]
                    masked_mse = mse_per_pixel * static_mask_resized  # [b*v, 3, h, w]
                    l2_loss = masked_mse.sum() / (num_static_pixels * rendering.size(1))
            else:
                l2_loss = F.mse_loss(rendering, target)

        psnr = -10.0 * torch.log10(l2_loss)

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if weight_lpips > 0.0:
            # Note: LPIPS doesn't have native mask support, so we apply mask after
            if static_mask is not None:
                if static_mask.shape[-2:] != (h, w):
                    static_mask_resized = F.interpolate(static_mask, size=(h, w), mode="nearest")
                else:
                    static_mask_resized = static_mask

                # Check if we have enough static pixels
                num_static_pixels = static_mask_resized.sum()
                min_static_pixels = 0.1 * b * v * h * w

                if num_static_pixels < min_static_pixels:
                    # Too few static pixels - fall back to unmasked loss
                    lpips_loss = self.lpips_loss_module(
                        rendering * 2.0 - 1.0, target * 2.0 - 1.0
                    ).mean()
                else:
                    # Compute LPIPS per-pixel, then mask
                    lpips_per_pixel = self.lpips_loss_module(
                        rendering * 2.0 - 1.0, target * 2.0 - 1.0, normalize=False
                    )  # [b*v, 1, h', w']

                    # Upsample mask to match LPIPS output resolution (use nearest to preserve binary)
                    if static_mask_resized.shape[-2:] != lpips_per_pixel.shape[-2:]:
                        static_mask_lpips = F.interpolate(
                            static_mask_resized, size=lpips_per_pixel.shape[-2:], mode="nearest"
                        )
                    else:
                        static_mask_lpips = static_mask_resized

                    num_static_lpips = static_mask_lpips.sum()
                    masked_lpips = lpips_per_pixel * static_mask_lpips
                    lpips_loss = masked_lpips.sum() / num_static_lpips.clamp(min=1.0)
            else:
                lpips_loss = self.lpips_loss_module(
                    rendering * 2.0 - 1.0, target * 2.0 - 1.0
                ).mean()

        perceptual_loss = torch.tensor(0.0).to(l2_loss.device)
        if weight_perceptual > 0.0:
            # PerceptualLoss already supports masking
            if static_mask is not None:
                if static_mask.shape[-2:] != (h, w):
                    static_mask_resized = F.interpolate(static_mask, size=(h, w), mode="nearest")
                else:
                    static_mask_resized = static_mask

                # Check if we have enough static pixels
                num_static_pixels = static_mask_resized.sum()
                min_static_pixels = 0.1 * b * v * h * w

                if num_static_pixels < min_static_pixels:
                    # Too few static pixels - fall back to unmasked loss
                    perceptual_loss = self.perceptual_loss_module(rendering, target)
                else:
                    perceptual_loss = self.perceptual_loss_module(
                        rendering, target, mask=static_mask_resized
                    )
            else:
                perceptual_loss = self.perceptual_loss_module(rendering, target)

        ssim_loss = torch.tensor(0.0).to(l2_loss.device)
        if weight_ssim > 0.0:
            # Note: SSIM doesn't have native mask support in this implementation
            # For now, compute without mask (could implement masked SSIM if needed)
            ssim_loss = self.ssim_loss_module(rendering, target)

        loss = (
            weight_l2 * l2_loss
            + weight_lpips * lpips_loss
            + weight_perceptual * perceptual_loss
            + weight_ssim * ssim_loss
        )

        visual = None
        if create_visual:
            visual = torch.cat((target, rendering), dim=3).detach().cpu()  # [b*v, c, h, w * 2]
            visual = rearrange(visual, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
            visual = (visual.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)

        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            ssim_loss=ssim_loss,
            visual=visual,
        )
        return loss_metrics

    def forward(
        self,
        rendering,
        target,
        input,
        predicted_target_masks=None,
        dinov3_target_masks=None,
        predicted_input_masks=None,
        copy_paste_input_masks=None,
        create_visual=False,
        dataset_sources=None,
    ):
        """
        rendering: [b, v, 3, h, w]; in range (0, 1)
        target: [b, v, 3, h, w]; in range (0, 1)
        predicted_target_masks: [b*v, 1, H, W] predicted motion mask logits (optional)
        dinov3_target_masks: [b*v, 1, H, W] DINOv3 pseudo-label masks (optional)
        predicted_input_masks: [b*v_input, 1, H, W] predicted motion mask logits for input views
        copy_paste_input_masks: [b*v_input, 1, H, W] binary GT masks for input views (copy-paste)
        dataset_sources: list of dataset sources (e.g., ["re10k", "dre10k", ...]) for per-sample handling
        """

        # Create static mask from predicted motion masks (invert so 1=static, 0=dynamic)
        # IMPORTANT: Only apply masking to DRE10K (dynamic scenes), not RE10K (static scenes)
        static_mask = None
        if self.use_masked_reconstruction_loss and predicted_target_masks is not None:
            b, v = rendering.shape[0], rendering.shape[1]

            # 1) Convert logits -> prob -> hard binary motion -> invert to static
            motion_mask_prob = torch.sigmoid(predicted_target_masks)  # [b*v, 1, H, W]
            # Threshold to get binary mask (>threshold means motion)
            motion_mask_binary = (
                motion_mask_prob > self.motion_mask_threshold
            ).float()  # [b*v, 1, H, W]
            # Invert to get static mask (1=static, 0=dynamic)
            # CRITICAL: Detach to prevent gradient flow to mask predictor (frozen in Stage 3)
            static_mask_raw = (1.0 - motion_mask_binary).detach()  # [b*v, 1, H, W]

            # Only apply masking to DRE10K samples (we KNOW RE10K is static)
            if dataset_sources is not None:
                # Safety check: ensure dataset_sources matches batch size
                assert (
                    len(dataset_sources) == b
                ), f"dataset_sources length {len(dataset_sources)} != batch size {b}"

                # 2) Normalize and allow variants like 'dre10k_aug', 'dynamic'
                def is_dre10k(tag):
                    """Check if dataset is dynamic (DRE10K or variants)."""
                    t = str(tag).lower()
                    return ("dre10k" in t) or (t == "dynamic")

                # Create per-sample mask enable flag: True for DRE10K, False for RE10K
                apply_mask = torch.tensor(
                    [is_dre10k(ds) for ds in dataset_sources],
                    device=rendering.device,
                    dtype=torch.bool,  # Use bool dtype for semantic clarity
                )  # [b]

                # Expand to per-view: [b] -> [b*v, 1, 1, 1]
                apply_mask_expanded = apply_mask.repeat_interleave(v).view(-1, 1, 1, 1)

                # For RE10K: use all 1.0 (full image), for DRE10K: use predicted mask
                static_mask = torch.where(
                    apply_mask_expanded,
                    static_mask_raw,  # DRE10K: use predicted mask
                    torch.ones_like(static_mask_raw),  # RE10K: all static (1.0 = full image)
                )
            else:
                # Fallback: apply to all samples (backward compatibility)
                static_mask = static_mask_raw

        # calculate loss for input views with static mask
        loss_metrics_input = self.calculate_loss(
            rendering,
            target,
            create_visual=create_visual,
            is_interpolate=False,
            static_mask=static_mask,
        )

        # Motion mask distillation loss (BCE between predicted masks and DINOv3 pseudo-labels)
        mask_bce_loss = torch.tensor(0.0, device=rendering.device)
        input_mask_bce_loss = torch.tensor(0.0, device=rendering.device)

        if (
            self.weight_mask_distill > 0.0
            and predicted_input_masks is not None
            and copy_paste_input_masks is not None
        ):
            target_size_input = copy_paste_input_masks.shape[-2:]
            if predicted_input_masks.shape[-2:] != target_size_input:
                predicted_input_masks_upsampled = F.interpolate(
                    predicted_input_masks,
                    size=target_size_input,
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                predicted_input_masks_upsampled = predicted_input_masks

            input_mask_bce_loss = F.binary_cross_entropy_with_logits(
                predicted_input_masks_upsampled,
                copy_paste_input_masks.to(
                    predicted_input_masks_upsampled.device, dtype=torch.float32
                ),
            )

        if (
            self.weight_mask_distill > 0.0
            and predicted_target_masks is not None
            and dinov3_target_masks is not None
        ):
            # Apply PSNR filtering: only use samples with good rendering quality
            # IMPORTANT: Use the SAME masked MSE as the reconstruction loss for consistency
            if self.psnr_filter_threshold > 0.0:
                # Compute per-view PSNR using the SAME static mask as reconstruction loss
                b, v, _, h, w = rendering.size()
                rendering_flat = rendering.reshape(b * v, -1, h, w)
                target_flat = target.reshape(b * v, -1, h, w)

                if static_mask is not None:
                    # Upsample mask to rendering resolution (use nearest to preserve binary)
                    if static_mask.shape[-2:] != (h, w):
                        static_mask_resized = F.interpolate(
                            static_mask, size=(h, w), mode="nearest"
                        )
                    else:
                        static_mask_resized = static_mask

                    # Compute masked MSE per view (same as L2 loss)
                    mse_per_pixel = (rendering_flat - target_flat) ** 2  # [b*v, 3, h, w]
                    masked_mse = mse_per_pixel * static_mask_resized  # [b*v, 3, h, w]

                    # Average over pixels and channels for each view
                    mse_per_view = masked_mse.sum(dim=[1, 2, 3]) / (
                        static_mask_resized.sum(dim=[1, 2, 3]) * rendering_flat.size(1) + 1e-8
                    )  # [b*v]
                else:
                    # No mask - compute regular MSE
                    mse_per_view = F.mse_loss(rendering_flat, target_flat, reduction="none")
                    mse_per_view = mse_per_view.mean(dim=[1, 2, 3])  # [b*v]

                psnr_per_view = -10.0 * torch.log10(mse_per_view.clamp(min=1e-10))

                # Only keep samples with PSNR above threshold
                valid_mask = psnr_per_view >= self.psnr_filter_threshold  # [b*v]

                if valid_mask.sum() > 0:
                    predicted_masks_filtered = predicted_target_masks[valid_mask]
                    dinov3_masks_filtered = dinov3_target_masks[valid_mask]
                else:
                    # No valid samples, skip mask loss
                    predicted_masks_filtered = None
                    dinov3_masks_filtered = None
            else:
                predicted_masks_filtered = predicted_target_masks
                dinov3_masks_filtered = dinov3_target_masks

            # Compute BCE loss if we have valid samples
            if predicted_masks_filtered is not None and dinov3_masks_filtered is not None:
                # Upsample predicted masks to match pseudo-label resolution if needed
                target_size = dinov3_masks_filtered.shape[-2:]
                if predicted_masks_filtered.shape[-2:] != target_size:
                    predicted_masks_upsampled = F.interpolate(
                        predicted_masks_filtered,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    predicted_masks_upsampled = predicted_masks_filtered

                # BCE loss between predicted logits and DINOv3 pseudo-labels
                mask_bce_loss = F.binary_cross_entropy_with_logits(
                    predicted_masks_upsampled, dinov3_masks_filtered
                )

        # Total loss
        total_loss = loss_metrics_input.loss + self.weight_mask_distill * (
            mask_bce_loss + input_mask_bce_loss
        )

        # merge dict
        loss_metrics_all = edict(
            loss=total_loss,
            mask_bce_loss=mask_bce_loss,
            input_mask_bce_loss=input_mask_bce_loss,
            **{k: v for k, v in loss_metrics_input.items() if k != "loss"},
        )

        return loss_metrics_all

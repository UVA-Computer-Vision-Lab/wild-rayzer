# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).
import copy
from datetime import datetime
import os
import math
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from skimage.metrics import structural_similarity
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
import torch.distributed as dist
from transformers import AutoImageProcessor, AutoModel
import cv2
from utils import camera_utils, data_utils
from .transformer import (
    QK_Norm_TransformerBlock,
    init_weights,
    _init_weights,
    _init_weights_layerwise,
)
from .loss import LossComputer_official
import torch.nn.functional as F
from utils.pe_utils_official import get_1d_sincos_pos_emb_from_grid, get_2d_sincos_pos_embed
from utils.pose_utils import rot6d2mat, quat2mat
from utils.metric_utils import compute_psnr, compute_lpips, compute_ssim
from utils.data_utils import create_video_from_frames
from utils.training_utils import bilinear_resize
from PIL import Image


def _gn_groups(out_ch: int, prefer: int = 32) -> int:
    g = math.gcd(out_ch, prefer)
    return g if g > 0 else 1


class Up2x(nn.Module):
    def __init__(self, in_ch, out_ch, gn_groups=32):
        super().__init__()
        g = _gn_groups(out_ch, gn_groups)

        # 2x upsample (bilinear) then change channels in_ch -> out_ch
        self.conv_up = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        # refinement at target channels
        self.refine = nn.Sequential(
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        # single upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # project channels to out_ch
        x = self.conv_up(x)
        # refine
        x = self.refine(x)
        return x


class MotionMaskPredictor(nn.Module):
    """
    Three-modality motion mask predictor: DINOv3 + Image Tokens + Camera Poses.

    Architecture:
        1. Feature Fusion: Combines semantic (DINO), appearance (tokens), and geometric (pose) cues
        2. Motion Reasoning Transformer: Processes fused features to understand motion patterns
        3. Improved DPT Decoder: Multi-scale upsampling with skip connections for crisp masks

    This is the M-step of the EM training loop: learns to predict motion masks from GT images.
    """

    def __init__(self, config, dinov3_backbone, dinov3_processor):
        super().__init__()
        self.config = config
        # Store DINOv3 as non-persistent reference (moves to device but not saved in state_dict)
        # We use object.__setattr__ to bypass nn.Module's __setattr__ which would save it
        object.__setattr__(self, "_dinov3_backbone_ref", dinov3_backbone)
        object.__setattr__(self, "_dinov3_processor_ref", dinov3_processor)
        self.use_grad_checkpoint = config.training.get("grad_checkpoint_every", 1) > 0

        # Manually register DINOv3's device placement hook (moves to device without saving)
        # This ensures it follows the parent model's device
        self._register_dinov3_device_hook()

        # Dimensions
        d_main = config.model.transformer.d  # Main transformer dim (e.g., 768)
        d_dino = dinov3_backbone.config.hidden_size  # DINOv3 dim (e.g., 1024 for ViT-L)
        d_fused = d_main  # Output fusion dimension

        # Store DINOv3 config
        self.patch_size_dino = getattr(dinov3_backbone.config, "patch_size", 16)
        self.num_register_tokens = getattr(dinov3_backbone.config, "num_register_tokens", 4)

        # Normalization buffers for DINOv3 (from processor)
        mean = torch.tensor(dinov3_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(dinov3_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("dino_norm_mean", mean)
        self.register_buffer("dino_norm_std", std)

        # 1. Feature Fusion Module
        # Project each modality to common dimension, then fuse
        self.dino_proj = nn.Sequential(
            nn.LayerNorm(d_dino, bias=False),
            nn.Linear(d_dino, d_fused, bias=True),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(d_main, bias=False),
            nn.Linear(d_main, d_fused, bias=True),
        )
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(d_main, bias=False),  # Plücker embeddings already in d_main
            nn.Linear(d_main, d_fused, bias=True),
        )

        # Fusion via concatenation + MLP (3*d_fused -> d_fused)
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(d_fused * 3, bias=False),
            nn.Linear(d_fused * 3, d_fused, bias=True),
            nn.GELU(),
            nn.Linear(d_fused, d_fused, bias=True),
        )

        use_qk_norm = config.model.transformer.get("use_qk_norm", False)
        d_head = d_fused // 12  # 12 heads for 768-dim
        self.motion_transformer = nn.ModuleList(
            [QK_Norm_TransformerBlock(d_fused, d_head, use_qk_norm=use_qk_norm) for _ in range(4)]
        )

        # 3. Improved DPT Head with Multi-Scale Features
        # Following original DPT: extract features at multiple scales and fuse
        self.dpt_readout = nn.Sequential(
            nn.LayerNorm(d_fused, bias=False),
            nn.Linear(d_fused, d_fused, bias=True),
        )

        # Multi-scale feature extraction (3 scales)
        # Scale 1: 16x16 (patch level)
        self.scale1_refine = nn.Sequential(
            nn.Conv2d(d_fused, d_fused // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=d_fused // 2),
            nn.GELU(),
            nn.Conv2d(d_fused // 2, d_fused // 2, kernel_size=3, padding=1),
        )
        self.up2 = Up2x(d_fused // 2, d_fused // 4)  # 16 -> 32
        self.up3 = Up2x(d_fused // 4, d_fused // 8)  # 32 -> 64
        self.up4 = Up2x(d_fused // 8, d_fused // 16)  # 64 -> 128
        self.up5 = Up2x(d_fused // 16, d_fused // 32)  # 128 -> 256

        self.scale2_refine = nn.Sequential(
            nn.Conv2d(d_fused // 4, d_fused // 4, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=d_fused // 4),
            nn.GELU(),
        )
        self.scale3_refine = nn.Sequential(
            nn.Conv2d(d_fused // 8, d_fused // 8, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=d_fused // 8),
            nn.GELU(),
        )
        self.scale4_refine = nn.Sequential(
            nn.Conv2d(d_fused // 16, d_fused // 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=d_fused // 16),
            nn.GELU(),
        )
        self.scale5_refine = nn.Sequential(
            nn.Conv2d(d_fused // 32, d_fused // 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_fused // 32),
            nn.GELU(),
        )

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(d_fused // 32, d_fused // 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_fused // 32, 1, kernel_size=1, bias=True),
        )

        # Initialize weights
        for m in [
            self.dino_proj,
            self.token_proj,
            self.pose_proj,
            self.fusion_mlp,
            self.motion_transformer,
            self.scale1_refine,
            self.scale2_refine,
            self.scale3_refine,
            self.scale4_refine,
            self.scale5_refine,
            self.head,
        ]:
            m.apply(_init_weights)
        self.up2.apply(_init_weights)
        self.up3.apply(_init_weights)
        self.up4.apply(_init_weights)
        self.up5.apply(_init_weights)

    def _register_dinov3_device_hook(self):
        """Register a hook to move DINOv3 backbone when parent model moves to a device."""

        def move_dinov3_to_device(module, inputs):
            if hasattr(module, "_dinov3_backbone_ref"):
                backbone = object.__getattribute__(module, "_dinov3_backbone_ref")
                if backbone is not None:
                    device = next(module.parameters()).device
                    backbone.to(device)

        # Register the hook
        self.register_forward_pre_hook(move_dinov3_to_device)

    @torch.no_grad()
    def _extract_dino_features(self, images_224):
        """
        Args:
            images_224: [B, 3, 224, 224] in [0,1], already resized to 224
        Returns:
            [B, 16*16, D_dino] patch tokens
        """
        img_norm = (images_224 - self.dino_norm_mean) / self.dino_norm_std
        outputs = self._dinov3_backbone_ref(img_norm, output_hidden_states=False)
        skip = 1 + self.num_register_tokens
        return outputs.last_hidden_state[:, skip:, :]  # [B, 256, D_dino] => 16×16

    def forward(self, target_images, target_image_tokens, target_plucker_emb):
        """
        Predict motion masks from three modalities.

        Args:
            target_images: [B*V, 3, H, W] GT target images in [0, 1]
            target_image_tokens: [B*V, N_main, D_main] encoded image tokens from main transformer
            target_plucker_emb: [B*V, N_main, D_main] tokenized Plücker embeddings (camera poses)

        Returns:
            mask_logits: [B*V, 1, H_out, W_out] motion mask logits (H_out=256, W_out=256)
        """
        BV, _, H, W = target_images.shape
        N_main = target_image_tokens.shape[1]
        Hm = Wm = int(N_main**0.5)  # e.g., 16x16
        assert Hm * Wm == N_main, f"Hm * Wm ({Hm}*{Wm}) != N_main ({N_main})"

        # Use full resolution (256x256) for DINOv3 to match main model's grid
        dino_in = target_images  # Already 256x256, no resize needed
        dino_features = self._extract_dino_features(dino_in)  # [BV, 256, D_dino]

        Nd = dino_features.shape[1]
        Hd = Wd = int(Nd**0.5)

        # 2. Align spatial dimensions (interpolate DINO to main grid size)
        if (Hd != Hm) or (Wd != Wm):
            dino_spatial = rearrange(dino_features, "bv (h w) d -> bv d h w", h=Hd, w=Wd)
            dino_spatial = bilinear_resize(dino_spatial, size=(Hm, Wm))
            dino_features = rearrange(dino_spatial, "bv d h w -> bv (h w) d")

        dino_proj = self.dino_proj(dino_features)
        token_proj = self.token_proj(target_image_tokens)
        pose_proj = self.pose_proj(target_plucker_emb)
        fused = torch.cat([dino_proj, token_proj, pose_proj], dim=-1)
        fused = self.fusion_mlp(fused)

        # 5. Motion reasoning via transformer (with gradient checkpointing)
        x = fused
        if self.use_grad_checkpoint and self.training:
            for layer in self.motion_transformer:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.motion_transformer:
                x = layer(x)

        # readout -> spatial 16×16
        x = self.dpt_readout(x)  # [BV, 256, D_fused]
        x = rearrange(x, "bv (h w) d -> bv d h w", h=Hm, w=Wm)  # [BV, D_fused, 16, 16]

        # decoder 16→32→64→128→256
        x1 = self.scale1_refine(x)  # 16×16, C/2
        x2 = self.up2(x1)
        x2 = self.scale2_refine(x2)
        x3 = self.up3(x2)
        x3 = self.scale3_refine(x3)
        x4 = self.up4(x3)
        x4 = self.scale4_refine(x4)
        x5 = self.up5(x4)
        x5 = self.scale5_refine(x5)

        mask_logits = self.head(x5)
        return mask_logits


class DinoV3UncertaintyPseudoLabelMaker(nn.Module):
    """
    DINOv3-based pseudo-label generator for motion/uncertainty masks.
    Uses min(DINO cosine dissimilarity, SSIM dissimilarity) with percentile-based soft masking.

    This is the improved version from our Jupyter notebook analysis:
    - Combines DINOv3 features + SSIM for robustness
    - Generates soft masks (continuous probabilities) not binary
    - Uses percentile-based normalization (P75, P85, P90)
    - NO gradient computation (frozen DINOv3)
    - DDP-safe: device placement handled automatically by parent module
    """

    def __init__(
        self,
        dinov3_backbone,
        dinov3_processor,
        percentile_threshold=75,
        use_ssim=True,
        target_size=256,
        use_coseg_binary=False,
        coseg_k=64,
        coseg_consistency_min_frames=4,
        coseg_frame_saliency_quantile=0.75,
        coseg_cluster_top_percent=0.05,
        coseg_morph_kernel=3,
        coseg_morph_iters=1,
        coseg_min_component_area_ratio=0.0025,
        coseg_grabcut_kernel=7,
        w_dino=0.5,
        w_ssim=0.5,
    ):
        super().__init__()
        self.percentile_threshold = percentile_threshold
        self.use_ssim = use_ssim
        self.target_size = target_size
        self.use_coseg_binary = use_coseg_binary
        # Online co-seg params (lightweight)
        self.coseg_k = coseg_k
        self.coseg_consistency_min_frames = coseg_consistency_min_frames
        self.coseg_frame_saliency_quantile = coseg_frame_saliency_quantile
        self.coseg_cluster_top_percent = coseg_cluster_top_percent
        self.coseg_morph_kernel = coseg_morph_kernel
        self.coseg_morph_iters = coseg_morph_iters
        self.coseg_min_component_area_ratio = coseg_min_component_area_ratio
        self.coseg_grabcut_kernel = coseg_grabcut_kernel
        self.w_dino = w_dino
        self.w_ssim = w_ssim

        # Store DINOv3 backbone as non-persistent reference (moves to device but not saved in state_dict)
        object.__setattr__(self, "backbone", dinov3_backbone)
        object.__setattr__(self, "processor", dinov3_processor)

        # Manually register DINOv3's device placement hook
        self._register_dinov3_device_hook()

        # DINOv3 config - read actual patch_size from model
        self.patch_size = getattr(dinov3_backbone.config, "patch_size", 16)
        self.num_register_tokens = getattr(dinov3_backbone.config, "num_register_tokens", 4)
        self.embed_dim = dinov3_backbone.config.hidden_size

        # Normalization parameters (as buffers for automatic device placement)
        mean = torch.tensor(dinov3_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(dinov3_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("img_norm_mean", mean)
        self.register_buffer("img_norm_std", std)

    @property
    def device(self):
        """Get current device from registered buffers (DDP-safe)."""
        return self.img_norm_mean.device

    def _register_dinov3_device_hook(self):
        """Register a hook to move DINOv3 backbone when parent model moves to a device."""

        def move_dinov3_to_device(module, inputs):
            if hasattr(module, "backbone"):
                backbone = object.__getattribute__(module, "backbone")
                if backbone is not None:
                    device = module.device
                    backbone.to(device)

        # Register the hook
        self.register_forward_pre_hook(move_dinov3_to_device)

    @torch.no_grad()
    def _extract_patch_features(self, img_tensor):
        """
        Extract patch-level features from DINOv3.

        Args:
            img_tensor: [B, 3, H, W] in range [0, 1]

        Returns:
            features: [B, D, H_patches, W_patches] L2-normalized patch features
        """
        # Get original image dimensions
        B, _, H, W = img_tensor.shape
        
        # Ensure backbone is on the same device as input
        device = img_tensor.device
        if hasattr(self, "backbone"):
            backbone = object.__getattribute__(self, "backbone")
            if backbone is not None and next(backbone.parameters()).device != device:
                backbone.to(device)
        
        # Normalize using DINOv3's preprocessing
        img_normalized = (
            img_tensor - self.img_norm_mean.to(img_tensor.device)
        ) / self.img_norm_std.to(img_tensor.device)

        # Forward pass
        outputs = self.backbone(img_normalized, output_hidden_states=False)

        # Extract patch tokens (skip CLS + register tokens)
        skip_tokens = 1 + self.num_register_tokens  # CLS + 4 register tokens
        patch_tokens = outputs.last_hidden_state[:, skip_tokens:, :]  # [B, N_patches, D]

        B, N, D = patch_tokens.shape
        
        # Infer patch grid dimensions from N
        # The model might internally resize images, so we can't rely on input dimensions
        
        # First, try assuming square output
        sqrt_N = int(np.sqrt(N))
        if sqrt_N * sqrt_N == N:
            # Perfect square - use it
            H_patches = W_patches = sqrt_N
        else:
            # Not square - need to factorize N into H_patches × W_patches
            # Try to match the input image aspect ratio
            aspect_ratio = W / H
            
            # Find factors of N that are closest to the aspect ratio
            best_h = best_w = None
            min_diff = float('inf')
            
            for h in range(1, int(np.sqrt(N)) + 1):
                if N % h == 0:
                    w = N // h
                    ratio_diff = abs(w / h - aspect_ratio)
                    if ratio_diff < min_diff:
                        min_diff = ratio_diff
                        best_h, best_w = h, w
            
            if best_h is None:
                raise ValueError(f"Cannot factorize {N} patches (N must be composite or square)")
            
            H_patches, W_patches = best_h, best_w

        # Reshape to spatial grid and L2-normalize
        features = patch_tokens.transpose(1, 2).reshape(B, D, H_patches, W_patches)
        features = F.normalize(features, dim=1)  # L2 normalize along channel dim

        return features

    def _refine_with_grabcut(self, gt_img_rgb: np.ndarray, init_mask: np.ndarray) -> np.ndarray:
        """
        Refine binary mask using GrabCut (matches offline script functionality).

        Args:
            gt_img_rgb: [H, W, 3] uint8 RGB image
            init_mask: [H, W] uint8 {0, 1} initial mask

        Returns:
            refined_mask: [H, W] uint8 {0, 1} GrabCut-refined mask
        """
        if self.coseg_grabcut_kernel <= 0:
            return init_mask

        # Erode to get confident FG/BG seeds
        kernel = np.ones((self.coseg_grabcut_kernel, self.coseg_grabcut_kernel), np.uint8)
        fg_seed = cv2.erode(np.uint8(init_mask), kernel)
        bg_seed = cv2.erode(np.uint8(1 - init_mask), kernel)

        # Build GrabCut mask
        full_mask = np.ones((gt_img_rgb.shape[0], gt_img_rgb.shape[1]), np.uint8) * cv2.GC_PR_FGD
        full_mask[bg_seed == 1] = cv2.GC_BGD
        full_mask[fg_seed == 1] = cv2.GC_FGD

        # Run GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(gt_img_rgb, full_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            grabcut_mask = np.where((full_mask == 2) | (full_mask == 0), 0, 1).astype("uint8")
        except Exception:
            # Fallback to init mask if GrabCut fails
            grabcut_mask = init_mask.astype("uint8")

        return grabcut_mask

    @torch.no_grad()
    def generate_binary_coseg_masks(
        self, gt_img, pred_img, save_dir=None, frame_indices=None, return_intermediates=False
    ):
        """
        Online co-segmentation: KMeans over GT patch features + fused saliency voting.
        GPU-optimized with vectorized PyTorch ops.
        Returns [B,1,target_size,target_size] uint8 {0,1}.

        Args:
            gt_img: [B, 3, H, W] ground truth images
            pred_img: [B, 3, H, W] predicted images
            save_dir: Optional directory to save masks as PNG files
            frame_indices: Optional list of frame indices for filename
            return_intermediates: If True, return dict with intermediate results for visualization

        Returns:
            If return_intermediates=False: torch.Tensor [B,1,H,W] final masks
            If return_intermediates=True: dict with keys:
                - 'masks': final masks [B,1,H,W]
                - 'clusters': K-means cluster assignments [B,H,W]
                - 'fg_binary': binary mask after K-means selection, before refinement [B,1,H,W]
        """
        gt_features = self._extract_patch_features(gt_img)  # [B, D, Hp, Wp]
        pred_features = self._extract_patch_features(pred_img)
        B, D, Hp, Wp = gt_features.shape
        device = gt_img.device

        gt_norm = F.normalize(gt_features, dim=1)
        pred_norm = F.normalize(pred_features, dim=1)
        cos_sim = (gt_norm * pred_norm).sum(dim=1, keepdim=True)  # [B,1,Hp,Wp]
        cos_dissim = 1.0 - cos_sim

        if self.use_ssim:
            ssim_dissim = self._compute_ssim_dissimilarity(gt_img, pred_img)  # [B,1,T,T]
            ssim_down = F.interpolate(ssim_dissim, size=(Hp, Wp), mode="area")

            def z(x):
                mu = x.mean(dim=(2, 3), keepdim=True)
                sd = x.std(dim=(2, 3), keepdim=True) + 1e-8
                return (x - mu) / sd

            fused_patch = self.w_dino * z(cos_dissim) + self.w_ssim * z(ssim_down)
        else:
            fused_patch = cos_dissim

        # KMeans on GT descriptors (GPU-accelerated, deterministic)
        gt_desc = gt_features.permute(0, 2, 3, 1).reshape(B, Hp * Wp, D)  # [B, Hp*Wp, D]
        X_all = gt_desc.reshape(-1, D)  # [B*Hp*Wp, D]

        # GPU KMeans with k-means++ initialization (matches sklearn behavior)
        N = X_all.shape[0]
        K = self.coseg_k

        # k-means++ initialization (deterministic with fixed seed)
        torch.manual_seed(42)
        centroids = []
        # Pick first centroid randomly
        first_idx = torch.randint(0, N, (1,), device=device)
        centroids.append(X_all[first_idx])

        for _ in range(K - 1):
            # Compute distance to nearest centroid
            centroids_stack = torch.cat(centroids, dim=0)  # [num_centroids, D]
            dists = torch.cdist(X_all, centroids_stack)  # [N, num_centroids]
            min_dists = dists.min(dim=1)[0]  # [N]
            # Sample proportional to distance squared (k-means++)
            probs = min_dists**2
            probs = probs / (probs.sum() + 1e-10)
            next_idx = torch.multinomial(probs, 1)
            centroids.append(X_all[next_idx])

        centroids = torch.cat(centroids, dim=0)  # [K, D]

        # Lloyd's algorithm (20 iterations for convergence)
        for iteration in range(20):
            # Assignment step
            dists = torch.cdist(X_all, centroids)  # [N, K]
            labels_flat = dists.argmin(dim=1)  # [N]

            # Update step
            centroids_new = centroids.clone()
            for k in range(K):
                mask_k = labels_flat == k
                if mask_k.sum() > 0:
                    centroids_new[k] = X_all[mask_k].mean(dim=0)

            # Check convergence
            if iteration > 0 and torch.allclose(centroids, centroids_new, atol=1e-4):
                break
            centroids = centroids_new

        labels_per_image = labels_flat.reshape(B, Hp * Wp)  # [B, Hp*Wp]

        # Vectorized cluster scoring (all on GPU)
        fused_flat = fused_patch.squeeze(1).reshape(B, Hp * Wp)  # [B, Hp*Wp]
        Kc = self.coseg_k

        # Per-cluster global average saliency: [K]
        cluster_avg_sals = torch.zeros(Kc, device=device)
        for k in range(Kc):
            mask_k = labels_per_image == k  # [B, Hp*Wp]
            vals_k = fused_flat[mask_k]
            cluster_avg_sals[k] = vals_k.mean() if vals_k.numel() > 0 else 0.0

        # Per-cluster per-frame median saliency: [K, B]
        cluster_per_frame_medians = torch.zeros(Kc, B, device=device)
        for k in range(Kc):
            for f in range(B):
                mask_kf = labels_per_image[f] == k  # [Hp*Wp]
                vals_kf = fused_flat[f, mask_kf]
                cluster_per_frame_medians[k, f] = vals_kf.median() if vals_kf.numel() > 0 else 0.0

        # Thresholds
        rank_thr = torch.quantile(cluster_avg_sals, 1.0 - self.coseg_cluster_top_percent)
        per_frame_thr = torch.quantile(fused_flat, self.coseg_frame_saliency_quantile)

        # Select foreground clusters (vectorized)
        high_sal_mask = cluster_avg_sals >= rank_thr  # [K]
        consistent_mask = (cluster_per_frame_medians >= per_frame_thr).sum(
            dim=1
        ) >= self.coseg_consistency_min_frames  # [K]
        fg_cluster_mask = high_sal_mask & consistent_mask  # [K]
        fg_cluster_indices = torch.nonzero(fg_cluster_mask, as_tuple=True)[0]  # list of K indices

        # Build masks (vectorized per-frame)
        masks = []
        fg_binary_before_refine = []  # Store binary mask before GrabCut for visualization
        labels_grid = labels_per_image.reshape(B, Hp, Wp)  # [B, Hp, Wp]
        
        for f in range(B):
            # Binary mask: pixel is FG if its cluster is in fg_cluster_indices
            fg_mask = torch.zeros(Hp, Wp, dtype=torch.bool, device=device)
            for k_idx in fg_cluster_indices:
                fg_mask |= labels_grid[f] == k_idx

            fg_uint8 = fg_mask.to(torch.uint8)  # [Hp, Wp]
            # Upsample to target_size
            up = F.interpolate(
                fg_uint8[None, None].float(),
                size=(self.target_size, self.target_size),
                mode="nearest",
            )
            up = up.squeeze().cpu().numpy().astype(np.uint8)

            # Morphological ops + connected components (still on CPU with cv2)
            if self.coseg_morph_kernel > 0 and self.coseg_morph_iters > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.coseg_morph_kernel, self.coseg_morph_kernel)
                )
                up = cv2.morphologyEx(up, cv2.MORPH_CLOSE, k, iterations=self.coseg_morph_iters)
                up = cv2.morphologyEx(up, cv2.MORPH_OPEN, k, iterations=self.coseg_morph_iters)

            num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(up, connectivity=8)
            min_area = int(
                self.coseg_min_component_area_ratio * self.target_size * self.target_size
            )
            clean = np.zeros_like(up, dtype=np.uint8)
            for lab in range(1, num_labels):
                if stats[lab, cv2.CC_STAT_AREA] >= min_area:
                    clean[labels_cc == lab] = 1

            # Store binary mask before GrabCut refinement
            fg_binary_before_refine.append(clean)

            # Apply GrabCut refinement (matches offline script)
            gt_img_rgb = (gt_img[f].permute(1, 2, 0).cpu().numpy() * 255.0).astype(
                np.uint8
            )  # [H, W, 3]
            grabcut_mask = self._refine_with_grabcut(gt_img_rgb, clean)
            masks.append(grabcut_mask)

        masks_np = np.stack(masks, axis=0)[:, None, :, :]  # [B,1,H,W]
        masks_tensor = torch.from_numpy(masks_np).to(device)

        # Optionally save masks to disk
        if save_dir is not None:
            import os

            os.makedirs(save_dir, exist_ok=True)
            for b_idx in range(B):
                mask_uint8 = masks_np[b_idx, 0]  # [H, W] uint8 {0, 1}
                mask_uint8 = (mask_uint8 * 255).astype(np.uint8)  # Scale to {0, 255}

                # Generate filename
                if frame_indices is not None and b_idx < len(frame_indices):
                    frame_idx = frame_indices[b_idx]
                    filename = f"coseg_mask_frame_{frame_idx:06d}.png"
                else:
                    filename = f"coseg_mask_{b_idx:02d}.png"

                Image.fromarray(mask_uint8, mode="L").save(os.path.join(save_dir, filename))

        # Return intermediate results if requested
        if return_intermediates:
            # Upsample cluster labels to target_size for visualization
            clusters_upsampled = F.interpolate(
                labels_grid.unsqueeze(1).float(),
                size=(self.target_size, self.target_size),
                mode="nearest",
            )  # [B, 1, T, T]
            
            fg_binary_np = np.stack(fg_binary_before_refine, axis=0)[:, None, :, :]  # [B,1,H,W]
            fg_binary_tensor = torch.from_numpy(fg_binary_np).to(device).float()
            
            return {
                "masks": masks_tensor,
                "clusters": clusters_upsampled.squeeze(1),  # [B, H, W]
                "fg_binary": fg_binary_tensor,  # [B, 1, H, W]
            }
        
        return masks_tensor

    def _compute_cosine_dissimilarity(self, gt_img, pred_img):
        """
        Compute cosine dissimilarity (1 - cosine_similarity) between GT and predicted images.

        Args:
            gt_img: [B, 3, H, W] in range [0, 1]
            pred_img: [B, 3, H, W] in range [0, 1]

        Returns:
            cos_dissim: [B, 1, target_size, target_size] dissimilarity map
        """
        # Extract features
        gt_features = self._extract_patch_features(gt_img)  # [B, D, H_p, W_p]
        pred_features = self._extract_patch_features(pred_img)  # [B, D, H_p, W_p]

        # Cosine similarity (features already L2-normalized)
        gt_norm = F.normalize(gt_features, dim=1)
        pred_norm = F.normalize(pred_features, dim=1)
        cos_sim = (gt_norm * pred_norm).sum(dim=1, keepdim=True)  # [B, 1, H_p, W_p]

        # Dissimilarity = 1 - similarity
        cos_dissim = 1.0 - cos_sim

        # Upsample to target size
        cos_dissim_resized = F.interpolate(
            cos_dissim,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        return cos_dissim_resized

    def _compute_ssim_dissimilarity(self, gt_img, pred_img):
        """
        Compute SSIM dissimilarity (1 - SSIM) between GT and predicted images.

        Args:
            gt_img: [B, 3, H, W] in range [0, 1]
            pred_img: [B, 3, H, W] in range [0, 1]

        Returns:
            ssim_dissim: [B, 1, target_size, target_size] dissimilarity map
        """

        B = gt_img.shape[0]

        # Resize to target size first
        gt_resized = F.interpolate(
            gt_img, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False
        )
        pred_resized = F.interpolate(
            pred_img,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Compute SSIM for each image in the batch
        ssim_dissim_list = []
        for b in range(B):
            # Convert to numpy uint8 [0, 255]
            gt_np = (gt_resized[b].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            pred_np = (pred_resized[b].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

            # Compute SSIM map
            ssim_map = structural_similarity(
                gt_np, pred_np, win_size=11, channel_axis=2, data_range=255, full=True
            )[
                1
            ]  # Get the full SSIM map

            # Average across channels if needed
            ssim_map_gray = ssim_map.mean(axis=2) if ssim_map.ndim == 3 else ssim_map

            # Dissimilarity = 1 - SSIM
            ssim_dissim = 1.0 - ssim_map_gray
            ssim_dissim_list.append(ssim_dissim)

        # Stack and convert to tensor
        ssim_dissim = np.stack(ssim_dissim_list, axis=0)  # [B, H, W]
        ssim_dissim = torch.from_numpy(ssim_dissim).unsqueeze(1).to(gt_img.device)  # [B, 1, H, W]

        return ssim_dissim

    def _create_soft_mask(self, dissimilarity_map):
        """
        Create soft mask using percentile-based normalization.

        Args:
            dissimilarity_map: [B, 1, H, W] combined dissimilarity

        Returns:
            soft_mask: [B, 1, H, W] soft mask in range [0, 1]
        """
        B = dissimilarity_map.shape[0]
        soft_masks = []

        for b in range(B):
            dissim = dissimilarity_map[b, 0]  # [H, W]

            # Percentile-based normalization
            p_thresh = np.percentile(dissim.cpu().numpy(), self.percentile_threshold)
            p_max = dissim.max().item()

            # Normalize: [p_thresh, p_max] -> [0, 1]
            soft_mask = torch.clamp((dissim - p_thresh) / (p_max - p_thresh + 1e-8), 0.0, 1.0)
            soft_masks.append(soft_mask)

        soft_masks = torch.stack(soft_masks, dim=0).unsqueeze(1)  # [B, 1, H, W]
        return soft_masks

    @torch.no_grad()
    def forward(self, gt_img, pred_img, debug_save_path=None, use_comprehensive_viz=False):
        """
        Generate motion/uncertainty pseudo-labels (soft or binary based on use_coseg_binary flag).

        Args:
            gt_img: [B, 3, H, W] ground truth images in range [0, 1]
            pred_img: [B, 3, H, W] predicted images in range [0, 1]
            debug_save_path: Optional path to save debug visualization
            use_comprehensive_viz: If True, save comprehensive visualization showing full pipeline
                                  (GT, Rendering, MSE, SSIM, DINO, min, K-means, FG, Final)

        Returns:
            motion_mask: [B, 1, target_size, target_size] mask in range [0, 1]
                        If use_coseg_binary=True: binary {0,1} via KMeans+GrabCut
                        If use_coseg_binary=False: soft [0,1] via percentile normalization
        """
        # Compute DINOv3 cosine dissimilarity
        dino_dissim = self._compute_cosine_dissimilarity(gt_img, pred_img)

        # Optionally combine with SSIM
        if self.use_ssim:
            ssim_dissim = self._compute_ssim_dissimilarity(gt_img, pred_img)
            # Conservative: min dissimilarity (both must agree it's different)
            combined_dissim = torch.minimum(dino_dissim, ssim_dissim)
        else:
            ssim_dissim = None
            combined_dissim = dino_dissim
        
        # Initialize vars for comprehensive visualization
        kmeans_clusters = None
        fg_mask_binary = None
        mse_dissim = None
        
        if self.use_coseg_binary:
            # Produce binary motion masks (1=motion, 0=static) at target_size
            if use_comprehensive_viz and debug_save_path is not None:
                # Request intermediate results for visualization
                result = self.generate_binary_coseg_masks(gt_img, pred_img, return_intermediates=True)
                motion_mask = result["masks"].to(combined_dissim.device).to(torch.float32)
                kmeans_clusters = result["clusters"]
                fg_mask_binary = result["fg_binary"]
            else:
                bin_motion = (
                    self.generate_binary_coseg_masks(gt_img, pred_img)
                    .to(combined_dissim.device)
                    .to(torch.float32)
                )
                motion_mask = bin_motion
        else:
            # Create soft mask using percentile normalization (uncertainty-style)
            motion_mask = self._create_soft_mask(combined_dissim)

        # Debug visualization
        if debug_save_path is not None:
            if use_comprehensive_viz:
                # Compute MSE for comprehensive visualization
                mse_dissim = self._compute_mse_dissimilarity(gt_img, pred_img)
                
                # Save comprehensive visualization
                self._save_comprehensive_visualization(
                    gt_img,
                    pred_img,
                    mse_dissim,
                    ssim_dissim,
                    dino_dissim,
                    combined_dissim,
                    kmeans_clusters,
                    fg_mask_binary,
                    motion_mask,
                    debug_save_path,
                )
            else:
                # Save basic debug visualization
                self._save_debug_visualization(
                    gt_img,
                    pred_img,
                    dino_dissim,
                    ssim_dissim,
                    combined_dissim,
                    motion_mask,
                    debug_save_path,
                )

        return motion_mask

    def _compute_mse_dissimilarity(self, gt_img, pred_img):
        """
        Compute MSE dissimilarity between GT and predicted images.

        Args:
            gt_img: [B, 3, H, W] in range [0, 1]
            pred_img: [B, 3, H, W] in range [0, 1]

        Returns:
            mse_dissim: [B, 1, target_size, target_size] dissimilarity map
        """
        # Resize to target size
        gt_resized = F.interpolate(
            gt_img, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False
        )
        pred_resized = F.interpolate(
            pred_img,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Compute MSE per pixel (average over channels)
        mse = ((gt_resized - pred_resized) ** 2).mean(dim=1, keepdim=True)  # [B, 1, H, W]

        return mse

    def _save_comprehensive_visualization(
        self,
        gt_img,
        pred_img,
        mse_dissim,
        ssim_dissim,
        dino_dissim,
        combined_dissim,
        kmeans_clusters,
        fg_mask_binary,
        final_mask,
        save_path,
    ):
        """
        Save comprehensive visualization showing the full pipeline:
        GT, Rendering, MSE, SSIM, DINO, min(SSIM,DINO), K-means, FG selection, Final mask.

        Args:
            gt_img: [B, 3, H, W]
            pred_img: [B, 3, H, W]
            mse_dissim: [B, 1, H, W] MSE dissimilarity
            ssim_dissim: [B, 1, H, W] SSIM dissimilarity
            dino_dissim: [B, 1, H, W] DINO dissimilarity
            combined_dissim: [B, 1, H, W] min(SSIM, DINO)
            kmeans_clusters: [B, H, W] cluster assignments (for visualization)
            fg_mask_binary: [B, 1, H, W] binary mask after K-means selection
            final_mask: [B, 1, H, W] final mask after all post-processing
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        from pathlib import Path
        import matplotlib.colors as mcolors

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Take first sample from batch (convert bfloat16 to float32 for numpy)
        gt = gt_img[0].float().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        pred = pred_img[0].float().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        mse = mse_dissim[0, 0].float().cpu().numpy()  # [H, W]
        ssim = ssim_dissim[0, 0].float().cpu().numpy() if ssim_dissim is not None else None
        dino = dino_dissim[0, 0].float().cpu().numpy()  # [H, W]
        combined = combined_dissim[0, 0].float().cpu().numpy()  # [H, W]
        clusters = kmeans_clusters[0].float().cpu().numpy() if kmeans_clusters is not None else None
        fg_binary = (
            fg_mask_binary[0, 0].float().cpu().numpy() if fg_mask_binary is not None else None
        )
        final = final_mask[0, 0].float().cpu().numpy()  # [H, W]

        # Create figure with 2 rows x 5 columns
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))

        # Row 1: Full pipeline
        # 1. GT Image
        axes[0, 0].imshow(gt)
        axes[0, 0].set_title("GT Image", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")

        # 2. Rendered Image
        axes[0, 1].imshow(pred)
        axes[0, 1].set_title("Rendered Image", fontsize=12, fontweight="bold")
        axes[0, 1].axis("off")

        # 3. MSE Dissimilarity
        im_mse = axes[0, 2].imshow(mse, cmap="hot")
        axes[0, 2].set_title(
            f"MSE Dissim\n(mean={mse.mean():.3f}, max={mse.max():.3f})", fontsize=12, fontweight="bold"
        )
        axes[0, 2].axis("off")
        plt.colorbar(im_mse, ax=axes[0, 2], fraction=0.046)

        # 4. DINO Dissimilarity
        im_dino = axes[0, 3].imshow(dino, cmap="hot", vmin=0, vmax=1)
        axes[0, 3].set_title(
            f"DINO Cosine Dissim\n(mean={dino.mean():.3f}, max={dino.max():.3f})",
            fontsize=12,
            fontweight="bold",
        )
        axes[0, 3].axis("off")
        plt.colorbar(im_dino, ax=axes[0, 3], fraction=0.046)

        # 5. SSIM Dissimilarity
        if ssim is not None:
            im_ssim = axes[0, 4].imshow(ssim, cmap="hot", vmin=0, vmax=1)
            axes[0, 4].set_title(
                f"SSIM Dissim\n(mean={ssim.mean():.3f}, max={ssim.max():.3f})",
                fontsize=12,
                fontweight="bold",
            )
            axes[0, 4].axis("off")
            plt.colorbar(im_ssim, ax=axes[0, 4], fraction=0.046)
        else:
            axes[0, 4].axis("off")

        # Row 2: Clustering and final results
        # 1. min(SSIM, DINO)
        im_combined = axes[1, 0].imshow(combined, cmap="hot", vmin=0, vmax=1)
        axes[1, 0].set_title(
            f"min(DINO, SSIM)\n(mean={combined.mean():.3f}, max={combined.max():.3f})",
            fontsize=12,
            fontweight="bold",
        )
        axes[1, 0].axis("off")
        plt.colorbar(im_combined, ax=axes[1, 0], fraction=0.046)

        # 2. K-means Clusters (if available)
        if clusters is not None:
            # Create a colorful visualization of clusters
            n_clusters = int(clusters.max()) + 1
            im_clusters = axes[1, 1].imshow(clusters, cmap="tab20", interpolation="nearest")
            axes[1, 1].set_title(
                f"K-means Clusters\n(K={n_clusters})", fontsize=12, fontweight="bold"
            )
            axes[1, 1].axis("off")
        else:
            axes[1, 1].axis("off")

        # 3. Binary FG mask (after K-means selection, before refinement)
        if fg_binary is not None:
            im_fg = axes[1, 2].imshow(fg_binary, cmap="hot", vmin=0, vmax=1)
            axes[1, 2].set_title(
                f"FG Selection (binary)\n(coverage={fg_binary.mean():.3f})",
                fontsize=12,
                fontweight="bold",
            )
            axes[1, 2].axis("off")
            plt.colorbar(im_fg, ax=axes[1, 2], fraction=0.046)
        else:
            axes[1, 2].axis("off")

        # 4. Final mask (after all post-processing)
        im_final = axes[1, 3].imshow(final, cmap="hot", vmin=0, vmax=1)
        axes[1, 3].set_title(
            f"Final Mask\n(coverage={final.mean():.3f})", fontsize=12, fontweight="bold"
        )
        axes[1, 3].axis("off")
        plt.colorbar(im_final, ax=axes[1, 3], fraction=0.046)

        # 5. Final mask overlay on GT
        axes[1, 4].imshow(gt)
        axes[1, 4].imshow(final, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[1, 4].set_title("Final Mask on GT", fontsize=12, fontweight="bold")
        axes[1, 4].axis("off")

        # Title
        mode = "Binary Co-Seg" if self.use_coseg_binary else "Soft Percentile"
        param_str = f"mode={mode}, w_dino={self.w_dino:.2f}, w_ssim={self.w_ssim:.2f}"
        if self.use_coseg_binary:
            param_str += f", K={self.coseg_k}"
        else:
            param_str += f", P={self.percentile_threshold}"

        plt.suptitle(f"DINOv3 Motion Mask Pipeline - {param_str}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _save_debug_visualization(
        self, gt_img, pred_img, dino_dissim, ssim_dissim, combined_dissim, motion_mask, save_path
    ):
        """Save debug visualization showing all intermediate steps."""
        import matplotlib.pyplot as plt
        from pathlib import Path

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Take first sample from batch (convert bfloat16 to float32 for numpy)
        gt = gt_img[0].float().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        pred = pred_img[0].float().cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
        dino = dino_dissim[0, 0].float().cpu().numpy()  # [H, W]
        ssim = (
            ssim_dissim[0, 0].float().cpu().numpy() if ssim_dissim is not None else None
        )  # [H, W]
        combined = combined_dissim[0, 0].float().cpu().numpy()  # [H, W]
        mask = motion_mask[0, 0].float().cpu().numpy()  # [H, W]

        # Create figure
        n_cols = 6 if ssim is not None else 5
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8))

        # Row 1: Images and dissimilarity maps
        axes[0, 0].imshow(gt)
        axes[0, 0].set_title("GT Image", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(pred)
        axes[0, 1].set_title("Rendered Image", fontsize=14, fontweight="bold")
        axes[0, 1].axis("off")

        im2 = axes[0, 2].imshow(dino, cmap="hot", vmin=0, vmax=1)
        axes[0, 2].set_title(
            f"DINO Dissim\n(mean={dino.mean():.3f}, max={dino.max():.3f})",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, 2].axis("off")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

        if ssim is not None:
            im3 = axes[0, 3].imshow(ssim, cmap="hot", vmin=0, vmax=1)
            axes[0, 3].set_title(
                f"SSIM Dissim\n(mean={ssim.mean():.3f}, max={ssim.max():.3f})",
                fontsize=14,
                fontweight="bold",
            )
            axes[0, 3].axis("off")
            plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

            im4 = axes[0, 4].imshow(combined, cmap="hot", vmin=0, vmax=1)
            axes[0, 4].set_title(
                f"Combined (min)\n(mean={combined.mean():.3f}, max={combined.max():.3f})",
                fontsize=14,
                fontweight="bold",
            )
            axes[0, 4].axis("off")
            plt.colorbar(im4, ax=axes[0, 4], fraction=0.046)

            im5 = axes[0, 5].imshow(mask, cmap="hot", vmin=0, vmax=1)
            axes[0, 5].set_title(
                f"Final Mask (P{self.percentile_threshold})\n(mean={mask.mean():.3f}, max={mask.max():.3f})",
                fontsize=14,
                fontweight="bold",
            )
            axes[0, 5].axis("off")
            plt.colorbar(im5, ax=axes[0, 5], fraction=0.046)
        else:
            im4 = axes[0, 3].imshow(combined, cmap="hot", vmin=0, vmax=1)
            axes[0, 3].set_title(
                f"Combined\n(mean={combined.mean():.3f}, max={combined.max():.3f})",
                fontsize=14,
                fontweight="bold",
            )
            axes[0, 3].axis("off")
            plt.colorbar(im4, ax=axes[0, 3], fraction=0.046)

            im5 = axes[0, 4].imshow(mask, cmap="hot", vmin=0, vmax=1)
            axes[0, 4].set_title(
                f"Final Mask (P{self.percentile_threshold})\n(mean={mask.mean():.3f}, max={mask.max():.3f})",
                fontsize=14,
                fontweight="bold",
            )
            axes[0, 4].axis("off")
            plt.colorbar(im5, ax=axes[0, 4], fraction=0.046)

        # Row 2: Overlays on GT image
        axes[1, 0].imshow(gt)
        axes[1, 0].set_title("GT (reference)", fontsize=14)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred)
        axes[1, 1].set_title("Rendered (reference)", fontsize=14)
        axes[1, 1].axis("off")

        # DINO overlay
        axes[1, 2].imshow(gt)
        axes[1, 2].imshow(dino, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[1, 2].set_title("GT + DINO overlay", fontsize=14)
        axes[1, 2].axis("off")

        if ssim is not None:
            # SSIM overlay
            axes[1, 3].imshow(gt)
            axes[1, 3].imshow(ssim, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            axes[1, 3].set_title("GT + SSIM overlay", fontsize=14)
            axes[1, 3].axis("off")

            # Combined overlay
            axes[1, 4].imshow(gt)
            axes[1, 4].imshow(combined, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            axes[1, 4].set_title("GT + Combined overlay", fontsize=14)
            axes[1, 4].axis("off")

            # Final mask overlay
            axes[1, 5].imshow(gt)
            axes[1, 5].imshow(mask, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            axes[1, 5].set_title("GT + Final Mask overlay", fontsize=14)
            axes[1, 5].axis("off")
        else:
            # Combined overlay
            axes[1, 3].imshow(gt)
            axes[1, 3].imshow(combined, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            axes[1, 3].set_title("GT + Combined overlay", fontsize=14)
            axes[1, 3].axis("off")

            # Final mask overlay
            axes[1, 4].imshow(gt)
            axes[1, 4].imshow(mask, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            axes[1, 4].set_title("GT + Final Mask overlay", fontsize=14)
            axes[1, 4].axis("off")

        plt.suptitle(
            f"DINOv3 Pseudo-Label Debug (w_dino={self.w_dino:.2f}, w_ssim={self.w_ssim:.2f})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


class Images2Latent4D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.process_data = data_utils.ProcessData(config)
        # self.transform_input = TransformInput(config)

        # image tokenizer
        if not config.model.image_tokenizer.get("use_off_the_shelf", False):
            self.image_tokenizer = nn.Sequential(
                Rearrange(
                    "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                    ph=self.config.model.image_tokenizer.patch_size,
                    pw=self.config.model.image_tokenizer.patch_size,
                ),
                nn.Linear(
                    config.model.image_tokenizer.in_channels
                    * (config.model.image_tokenizer.patch_size**2),
                    config.model.transformer.d,
                    bias=False,
                ),
            )
            self.image_tokenizer.apply(_init_weights)
        else:
            # self.image_tokenizer = CustomImageTokenizer(config)
            pass

        # image positional embedding embedder
        self.use_pe_embedding_layer = config.model.get("input_with_pe", True)
        self.pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.pe_embedder.apply(_init_weights)

        # latent scene representation
        self.scene_code = nn.Parameter(
            torch.randn(
                config.model.scene_latent.length,
                config.model.transformer.d,
            )
        )
        nn.init.trunc_normal_(self.scene_code, std=0.02)

        # pose tokens
        self.cam_code = nn.Parameter(
            torch.randn(
                self.config.model.pose_latent.get("length", 1),
                config.model.transformer.d,
            )
        )
        nn.init.trunc_normal_(self.cam_code, std=0.02)

        # pose pe temporal embedder
        self.temporal_pe_embedder = (
            nn.Sequential(
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
                nn.SiLU(),
                nn.Linear(
                    config.model.transformer.d,
                    config.model.transformer.d,
                ),
            )
            if self.use_pe_embedding_layer
            else nn.Identity()
        )
        self.temporal_pe_embedder.apply(_init_weights)

        # qk norm settings
        use_qk_norm = config.model.transformer.get("use_qk_norm", False)

        # transformer encoder and init
        self.transformer_encoder = [
            QK_Norm_TransformerBlock(
                config.model.transformer.d,
                config.model.transformer.d_head,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(config.model.transformer.encoder_n_layer)
        ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.get("depth_init", False):
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_encoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            else:
                for idx in range(len(self.transformer_encoder)):
                    weight_init_std = 0.02 / (2 * config.model.transformer.encoder_n_layer) ** 0.5
                    self.transformer_encoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        else:
            self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
            self.transformer_encoder.apply(_init_weights)

        # transformer encoder2 and init
        self.transformer_encoder_geom = [
            QK_Norm_TransformerBlock(
                config.model.transformer.d,
                config.model.transformer.d_head,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(config.model.transformer.encoder_geom_n_layer)
        ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.get("depth_init", False):
                for idx in range(len(self.transformer_encoder_geom)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_encoder_geom[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            else:
                for idx in range(len(self.transformer_encoder_geom)):
                    weight_init_std = (
                        0.02 / (2 * config.model.transformer.encoder_geom_n_layer) ** 0.5
                    )
                    self.transformer_encoder_geom[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            self.transformer_encoder_geom = nn.ModuleList(self.transformer_encoder_geom)
        else:
            self.transformer_encoder_geom = nn.ModuleList(self.transformer_encoder_geom)
            self.transformer_encoder_geom.apply(_init_weights)

        # ln before decoder
        self.decoder_ln = nn.LayerNorm(config.model.transformer.d, bias=False)

        # transformer decoder and init
        self.transformer_decoder = [
            QK_Norm_TransformerBlock(
                config.model.transformer.d,
                config.model.transformer.d_head,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(config.model.transformer.decoder_n_layer)
        ]
        if config.model.transformer.get("special_init", False):
            if config.model.transformer.depth_init:
                for idx in range(len(self.transformer_decoder)):
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    self.transformer_decoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            else:
                for idx in range(len(self.transformer_decoder)):
                    weight_init_std = 0.02 / (2 * config.model.transformer.decoder_n_layer) ** 0.5
                    self.transformer_decoder[idx].apply(
                        lambda module: _init_weights_layerwise(module, weight_init_std)
                    )
            self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
        else:
            self.transformer_decoder = nn.ModuleList(self.transformer_decoder)
            self.transformer_decoder.apply(_init_weights)

        # pose predictor
        self.pose_predictor = PoseEstimator(config)

        # target pose tokenizer
        self.target_latent_h = (
            config.model.target_image.height // config.model.target_image.patch_size
        )
        self.target_latent_w = (
            config.model.target_image.width // config.model.target_image.patch_size
        )
        self.target_pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.target_pose_tokenizer.patch_size,
                pw=self.config.model.target_pose_tokenizer.patch_size,
            ),
            nn.Linear(
                config.model.target_pose_tokenizer.in_channels
                * (config.model.target_pose_tokenizer.patch_size**2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.target_pose_tokenizer.apply(_init_weights)

        self.target_pose_tokenizer2 = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.config.model.target_pose_tokenizer.patch_size,
                pw=self.config.model.target_pose_tokenizer.patch_size,
            ),
            nn.Linear(
                config.model.target_pose_tokenizer.in_channels
                * (config.model.target_pose_tokenizer.patch_size**2),
                config.model.transformer.d,
                bias=False,
            ),
        )
        self.target_pose_tokenizer2.apply(_init_weights)

        # fuse mlp
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d * 2, bias=False),
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                config.model.transformer.d,
                bias=True,
            ),
        )
        self.mlp_fuse.apply(_init_weights)

        # output regresser
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(config.model.transformer.d, bias=False),
            nn.Linear(
                config.model.transformer.d,
                (config.model.target_image.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.image_token_decoder.apply(_init_weights)

        # Motion mask system (two components):
        # 1. DINOv3 pseudo-label maker: generates GT masks from (rendered, GT) pairs
        # 2. MotionMaskPredictor: learns to predict masks from encoded tokens
        self.use_motion_mask = config.model.get("use_motion_mask", False)
        self.use_dinov3_pseudolabel = config.model.get("use_dinov3_pseudolabel", False)

        # Component 1: DINOv3 pseudo-label generator (frozen, for supervision)
        if self.use_dinov3_pseudolabel:
            dinov3_cfg = config.model.get("dinov3_pseudolabel", {})
            model_name = dinov3_cfg.get("model_name", "facebook/dinov3-vit7b16-pretrain-lvd1689m")

            # Load shared DINOv3 backbone
            dinov3_backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            dinov3_processor = AutoImageProcessor.from_pretrained(model_name)
            dinov3_backbone.eval()

            # Move to GPU (will be handled by .to(device) in __init__, but ensure dtype is set first)
            # Note: Don't call .cuda() here, let the parent model's .to(device) handle it

            # Freeze all DINOv3 parameters
            for p in dinov3_backbone.parameters():
                p.requires_grad = False

            self.dinov3_pseudolabel_maker = DinoV3UncertaintyPseudoLabelMaker(
                dinov3_backbone=dinov3_backbone,
                dinov3_processor=dinov3_processor,
                percentile_threshold=dinov3_cfg.get("percentile_threshold", 75),
                use_ssim=dinov3_cfg.get("use_ssim", True),
                target_size=dinov3_cfg.get("target_size", 256),
                use_coseg_binary=dinov3_cfg.get("use_coseg_binary", False),
                coseg_k=dinov3_cfg.get("coseg_k", 64),
                coseg_consistency_min_frames=dinov3_cfg.get("coseg_consistency_min_frames", 4),
                coseg_frame_saliency_quantile=dinov3_cfg.get("coseg_frame_saliency_quantile", 0.75),
                coseg_cluster_top_percent=dinov3_cfg.get("coseg_cluster_top_percent", 0.05),
                coseg_morph_kernel=dinov3_cfg.get("coseg_morph_kernel", 3),
                coseg_morph_iters=dinov3_cfg.get("coseg_morph_iters", 1),
                coseg_min_component_area_ratio=dinov3_cfg.get(
                    "coseg_min_component_area_ratio", 0.0025
                ),
                coseg_grabcut_kernel=dinov3_cfg.get("coseg_grabcut_kernel", 7),
                w_dino=dinov3_cfg.get("w_dino", 0.5),
                w_ssim=dinov3_cfg.get("w_ssim", 0.5),
            )

            percentile = dinov3_cfg.get("percentile_threshold", 75)
            use_ssim = dinov3_cfg.get("use_ssim", True)
            print(f"✓ DINOv3 pseudo-label maker initialized: P{percentile}, SSIM={use_ssim}")
        else:
            self.dinov3_pseudolabel_maker = None
            dinov3_backbone = None
            dinov3_processor = None

        # Component 2: Learned motion mask predictor (trainable)
        if self.use_motion_mask:
            if self.use_dinov3_pseudolabel and self.dinov3_pseudolabel_maker is not None:
                # Share the DINOv3 backbone from pseudo-label maker
                self.motion_mask_predictor = MotionMaskPredictor(
                    config,
                    self.dinov3_pseudolabel_maker.backbone,
                    self.dinov3_pseudolabel_maker.processor,
                )
                print("✓ Motion mask predictor initialized (sharing DINOv3 backbone)")
            else:
                # Load separate DINOv3 if pseudo-label maker not enabled
                model_name = config.model.get("dinov3_pseudolabel", {}).get(
                    "model_name", "facebook/dinov3-vit7b16-pretrain-lvd1689m"
                )
                dinov3_backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
                dinov3_processor = AutoImageProcessor.from_pretrained(model_name)
                dinov3_backbone.eval()
                for p in dinov3_backbone.parameters():
                    p.requires_grad = False

                self.motion_mask_predictor = MotionMaskPredictor(
                    config,
                    dinov3_backbone,
                    dinov3_processor,
                )
                print("✓ Motion mask predictor initialized (separate DINOv3 backbone)")
        else:
            self.motion_mask_predictor = None

        # Opportunistic caching for DINOv3 pseudo-labels (saves compute on frozen renderings)
        self.enable_mask_cache = config.model.dinov3_pseudolabel.get("enable_mask_cache", False)
        if self.enable_mask_cache:
            self.mask_cache_dir = config.model.dinov3_pseudolabel.get(
                "mask_cache_dir", "./motion_mask_cache"
            )
            print(f"✓ Opportunistic mask caching enabled: {self.mask_cache_dir}")
        else:
            self.mask_cache_dir = None

        # loss
        self.loss_computer = LossComputer_official(config)

        # config backup
        self.config_bk = copy.deepcopy(config)
        self.render_interpolate = config.training.get("render_interpolate", False)

        copy_paste_cfg = (
            getattr(config.training, "copy_paste", {}) if hasattr(config, "training") else {}
        )

        inference_cfg = getattr(config, "inference", None)
        if hasattr(inference_cfg, "get"):
            is_inference_mode = inference_cfg.get("if_inference", False)
        else:
            is_inference_mode = bool(inference_cfg)

        self.copy_paste_training_enabled = not is_inference_mode and copy_paste_cfg.get(
            "enabled", False
        )

        if config.model.transformer.get("fix_decoder", False):
            self.freeze_weights()
            if self.copy_paste_training_enabled:
                self._unfreeze_renderer_weights()

        # training settings
        if config.inference or config.get("evaluation", False):
            if config.training.get("random_inputs", False):
                self.random_index = True
            else:
                self.random_index = False
        else:
            self.random_index = config.training.get("random_split", False)

        print("Use random index:", self.random_index)

        self.motion_mask_threshold = config.model.get("motion_mask_threshold", 0.1)
        if config.model.get("use_mae_masking", False):
            self.noise_token = nn.Parameter(torch.randn(1, 1, self.config.model.transformer.d))
            nn.init.trunc_normal_(self.noise_token, std=0.02)

        # Stage 3: Freeze motion mask predictor if motion_mask_only_training is False
        # (This means we're training the rendering model, not the motion mask)
        motion_mask_only = config.model.get("motion_mask_only_training", False)
        if self.use_motion_mask and not motion_mask_only:
            if self.copy_paste_training_enabled:
                print("✓ Stage 3 + copy-paste detected: keeping motion mask predictor trainable")
            else:
                print(
                    "✓ Stage 3 detected: Freezing motion mask predictor (training rendering only)"
                )
                self._freeze_motion_mask_predictor()

    def freeze_weights(self):
        for param in self.target_pose_tokenizer.parameters():
            param.requires_grad = False

        for param in self.image_token_decoder.parameters():
            param.requires_grad = False

        for param in self.transformer_decoder.parameters():
            param.requires_grad = False

    def _unfreeze_renderer_weights(self):
        for module in [self.transformer_decoder, self.image_token_decoder]:
            for param in module.parameters():
                param.requires_grad = True
        print("✓ Copy-paste enabled: renderer weights unfrozen")

    def _freeze_motion_mask_predictor(self):
        """Freeze motion mask predictor for Stage 3 training (rendering only)."""
        if self.motion_mask_predictor is not None:
            for param in self.motion_mask_predictor.parameters():
                param.requires_grad = False
            print(
                f"  Frozen {sum(p.numel() for p in self.motion_mask_predictor.parameters())} motion mask parameters"
            )

    def train(self, mode=True):
        # override the train method to keep the fronzon modules in eval mode
        super().train(mode)
        self.loss_computer.eval()

    def get_overview(self):
        count_train_params = lambda model: sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        overview = edict(
            image_tokenizer=count_train_params(self.image_tokenizer),
            scene_latent=self.scene_code.data.numel(),
            transformer=count_train_params(self.transformer_decoder)
            + count_train_params(self.transformer_encoder),
            image_token_decoder=count_train_params(self.image_token_decoder),
        )
        return overview

    def _try_load_mask_from_cache(self, uid):
        """Try to load cached mask for a given scene UID."""
        if not self.enable_mask_cache or self.mask_cache_dir is None:
            return None

        cache_file = os.path.join(self.mask_cache_dir, f"mask_{uid:08d}.pt")
        if os.path.exists(cache_file):
            try:
                mask = torch.load(cache_file, map_location="cpu")
                return mask
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_file}: {e}")
                return None
        return None

    def _save_mask_to_cache(self, uid, mask):
        """Save computed mask to cache for future use."""
        if not self.enable_mask_cache or self.mask_cache_dir is None:
            return

        os.makedirs(self.mask_cache_dir, exist_ok=True)
        cache_file = os.path.join(self.mask_cache_dir, f"mask_{uid:08d}.pt")
        try:
            torch.save(mask.cpu(), cache_file)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_file}: {e}")

    def _apply_token_dropping(
        self,
        img_tokens_fused,  # [b, v_input*n, d] - AFTER fusion!
        predicted_masks,  # [b*v_input, 1, 256, 256] - motion mask logits OR binary probs
        b,
        v_input,
        n,
        d,
        dataset_sources=None,
        using_gt_masks=False,
        copy_paste_mask_input=None,
    ):
        """
        Apply token dropping to FUSED tokens using predicted motion masks.
        Zeros out dynamic tokens instead of replacing with learnable noise.

        For dynamic scenes (DRE10K): Use predicted motion masks
        For static scenes (RE10K): Apply random dropout (20% of tokens)

        Args:
            img_tokens_fused: [b, v_input*n, d] fused tokens (img + plucker)
            predicted_masks: [b*v_input, 1, 256, 256] motion masks (logits OR binary probs)
            b, v_input, n, d: batch size, num input views, num tokens, token dim
            dataset_sources: List[str] of length b, indicating dataset source per batch element
            using_gt_masks: bool, if True predicted_masks are already binary probs {0,1}

        Returns:
            img_tokens_dropped: [b, v_input*n, d] tokens with dynamic positions zeroed
            dynamic_mask: [b*v_input, 1, H_patch, W_patch] binary mask for visualization
        """
        if copy_paste_mask_input is None and (predicted_masks is None or not self.use_motion_mask):
            # No dropping
            return img_tokens_fused, None

        # Downsample masks from 256×256 to token grid (16×16)
        H_patch = W_patch = int(n**0.5)

        # Determine which masks to use for token dropping
        if copy_paste_mask_input is not None:
            mask_tokens_probs = copy_paste_mask_input.to(img_tokens_fused.device).float()
            # mask_tokens = F.adaptive_avg_pool2d(mask_tokens_probs, (H_patch, W_patch))
            # dynamic_mask = (mask_tokens > 0.5).float()
            use_copy_paste_masks = True
        else:
            use_copy_paste_masks = False
            if using_gt_masks:
                mask_tokens_probs = predicted_masks  # Already binary probabilities {0, 1}
            else:
                mask_tokens_probs = torch.sigmoid(predicted_masks)  # Convert logits to probs
        mask_tokens = F.adaptive_avg_pool2d(mask_tokens_probs, (H_patch, W_patch))
        dynamic_mask = (mask_tokens > self.motion_mask_threshold).float()

        # Apply random dropout based on dataset source
        if dataset_sources is not None and self.training and not use_copy_paste_masks:
            for batch_idx in range(b):
                dataset_source = dataset_sources[batch_idx]

                if dataset_source == "re10k" and self.config.model.get(
                    "re10k_random_dropout", False
                ):
                    # RE10K (static): Replace predicted masks with semantic box masks
                    dropout_ratio = self.config.model.get("re10k_dropout_ratio", 0.2)  # Default 20%
                    box_masks = torch.zeros(
                        v_input, 1, H_patch, W_patch, device=dynamic_mask.device
                    )

                    for v_idx in range(v_input):
                        # Number of boxes per view (1-3)
                        num_boxes = torch.randint(1, 4, (1,)).item()
                        target_area_per_box = (dropout_ratio * H_patch * W_patch) / num_boxes

                        for _ in range(num_boxes):
                            # Random box size (aspect ratio between 0.5 and 2.0)
                            aspect_ratio = 0.5 + 1.5 * torch.rand(1).item()  # [0.5, 2.0]
                            box_h = int((target_area_per_box * aspect_ratio) ** 0.5)
                            box_w = int(target_area_per_box / box_h)

                            # Clamp to valid range
                            box_h = min(max(box_h, 2), H_patch)
                            box_w = min(max(box_w, 2), W_patch)

                            # Random position
                            y = torch.randint(0, H_patch - box_h + 1, (1,)).item()
                            x = torch.randint(0, W_patch - box_w + 1, (1,)).item()

                            # Fill box region
                            box_masks[v_idx, 0, y : y + box_h, x : x + box_w] = 1.0

                    # Replace predicted masks with box masks for RE10K
                    start_idx = batch_idx * v_input
                    end_idx = start_idx + v_input
                    dynamic_mask[start_idx:end_idx] = box_masks

                # DRE10K: Add random dropout ON TOP of motion mask (not elif!)
                if dataset_source == "dre10k" and self.config.model.get(
                    "dre10k_random_dropout", False
                ):
                    # DRE10K (dynamic): Add additional random dropout on top of motion mask
                    dropout_ratio = self.config.model.get(
                        "dre10k_dropout_ratio", 0.05
                    )  # Default 5%

                    for v_idx in range(v_input):
                        # Generate small random box masks (1-2 boxes)
                        num_boxes = torch.randint(1, 3, (1,)).item()
                        target_area_per_box = (dropout_ratio * H_patch * W_patch) / num_boxes

                        for _ in range(num_boxes):
                            # Random box size (aspect ratio between 0.5 and 2.0)
                            aspect_ratio = 0.5 + 1.5 * torch.rand(1).item()  # [0.5, 2.0]
                            box_h = int((target_area_per_box * aspect_ratio) ** 0.5)
                            box_w = int(target_area_per_box / box_h)

                            # Clamp to valid range
                            box_h = min(max(box_h, 2), H_patch)
                            box_w = min(max(box_w, 2), W_patch)

                            # Random position
                            y = torch.randint(0, H_patch - box_h + 1, (1,)).item()
                            x = torch.randint(0, W_patch - box_w + 1, (1,)).item()

                            # Fill box region (add to existing motion mask using OR operation)
                            view_global_idx = batch_idx * v_input + v_idx
                            dynamic_mask[view_global_idx, 0, y : y + box_h, x : x + box_w] = 1.0

        # Reshape to match token sequence
        dynamic_mask_flat = rearrange(
            dynamic_mask, "(b v) 1 h w -> b (v h w) 1", b=b, v=v_input
        )  # [b, v_input*n, 1]

        # Token dropping: zero out dynamic tokens (no learnable noise!)
        img_tokens_dropped = img_tokens_fused * (1.0 - dynamic_mask_flat)

        return img_tokens_dropped, dynamic_mask

    def _apply_mae_masking(
        self,
        img_tokens_input,
        predicted_masks,
        b,
        v_input,
        n,
        d,
        dataset_sources=None,
        using_gt_masks=False,
        copy_paste_mask_input=None,
    ):
        """
        Apply MAE-style token masking to input tokens using predicted motion masks.
        """
        if copy_paste_mask_input is None and (predicted_masks is None or not self.use_motion_mask):
            return img_tokens_input

        # Downsample masks from 256×256 to token grid (16×16)
        H_patch = W_patch = int(n**0.5)

        if copy_paste_mask_input is not None:
            mask_tokens_probs = copy_paste_mask_input.to(img_tokens_input.device).float()
            # mask_tokens = F.adaptive_avg_pool2d(mask_tokens_probs, (H_patch, W_patch))
            # dynamic_mask = (mask_tokens > 0.5).float()
        else:
            # Binarize: sigmoid + threshold
            if using_gt_masks:
                mask_tokens_probs = predicted_masks  # Already binary probabilities {0, 1}
            else:
                mask_tokens_probs = torch.sigmoid(predicted_masks)
        mask_tokens = F.adaptive_avg_pool2d(mask_tokens_probs, (H_patch, W_patch))
        dynamic_mask = (mask_tokens > self.motion_mask_threshold).float()  # [b*v_input, 1, 16, 16]

        # Apply random dropout based on dataset source
        if dataset_sources is not None and self.training and copy_paste_mask_input is None:
            for batch_idx in range(b):
                dataset_source = dataset_sources[batch_idx]

                if dataset_source == "re10k" and self.config.model.get(
                    "re10k_random_dropout", False
                ):
                    # RE10K (static): Replace predicted masks with semantic box masks
                    dropout_ratio = self.config.model.get("re10k_dropout_ratio", 0.2)  # Default 20%
                    box_masks = torch.zeros(
                        v_input, 1, H_patch, W_patch, device=dynamic_mask.device
                    )

                    for v_idx in range(v_input):
                        # Number of boxes per view (1-3)
                        num_boxes = torch.randint(1, 4, (1,)).item()
                        target_area_per_box = (dropout_ratio * H_patch * W_patch) / num_boxes

                        for _ in range(num_boxes):
                            # Random box size (aspect ratio between 0.5 and 2.0)
                            aspect_ratio = 0.5 + 1.5 * torch.rand(1).item()  # [0.5, 2.0]
                            box_h = int((target_area_per_box * aspect_ratio) ** 0.5)
                            box_w = int(target_area_per_box / box_h)

                            # Clamp to valid range
                            box_h = min(max(box_h, 2), H_patch)
                            box_w = min(max(box_w, 2), W_patch)

                            # Random position
                            y = torch.randint(0, H_patch - box_h + 1, (1,)).item()
                            x = torch.randint(0, W_patch - box_w + 1, (1,)).item()

                            # Fill box region
                            box_masks[v_idx, 0, y : y + box_h, x : x + box_w] = 1.0

                    # Replace predicted masks with box masks for RE10K
                    start_idx = batch_idx * v_input
                    end_idx = start_idx + v_input
                    dynamic_mask[start_idx:end_idx] = box_masks

                # DRE10K: Add random dropout ON TOP of motion mask (not elif!)
                if dataset_source == "dre10k" and self.config.model.get(
                    "dre10k_random_dropout", False
                ):
                    # DRE10K (dynamic): Add additional random dropout on top of motion mask
                    dropout_ratio = self.config.model.get(
                        "dre10k_dropout_ratio", 0.05
                    )  # Default 5%

                    for v_idx in range(v_input):
                        # Generate small random box masks (1-2 boxes)
                        num_boxes = torch.randint(1, 3, (1,)).item()
                        target_area_per_box = (dropout_ratio * H_patch * W_patch) / num_boxes

                        for _ in range(num_boxes):
                            # Random box size (aspect ratio between 0.5 and 2.0)
                            aspect_ratio = 0.5 + 1.5 * torch.rand(1).item()  # [0.5, 2.0]
                            box_h = int((target_area_per_box * aspect_ratio) ** 0.5)
                            box_w = int(target_area_per_box / box_h)

                            # Clamp to valid range
                            box_h = min(max(box_h, 2), H_patch)
                            box_w = min(max(box_w, 2), W_patch)

                            # Random position
                            y = torch.randint(0, H_patch - box_h + 1, (1,)).item()
                            x = torch.randint(0, W_patch - box_w + 1, (1,)).item()

                            # Fill box region (add to existing motion mask using OR operation)
                            view_global_idx = batch_idx * v_input + v_idx
                            dynamic_mask[view_global_idx, 0, y : y + box_h, x : x + box_w] = 1.0

        # Reshape to match token sequence
        dynamic_mask_flat = rearrange(
            dynamic_mask, "(b v) 1 h w -> b (v h w) 1", b=b, v=v_input
        )  # [b, v_input*n, 1]

        # MAE-style replacement: keep static tokens, replace dynamic with learnable noise
        noise_tokens = self.noise_token.expand(b, v_input * n, d)
        img_tokens_masked = (
            img_tokens_input * (1.0 - dynamic_mask_flat)  # Keep static
            + noise_tokens * dynamic_mask_flat  # Replace dynamic with noise
        )
        return img_tokens_masked, dynamic_mask

    @torch.no_grad()
    def reconstruct_images(self, data_batch):
        """
        Minimal API used by train.py for visualization.
        Delegates to forward with create_visual=True to produce renderings and visuals.
        Returns an edict with at least: input, target, render, and optional render_video.
        """
        return self.forward(data_batch, create_visual=True, render_video=False)

    def forward(self, data, create_visual=False, render_video=False, iter=0):
        """
        Split all images into two sets, use one set to get scene representation,
        use the other to render & train
        """
        input, target = self.process_data(
            data, has_target_image=True, target_has_input=True, compute_rays=False
        )
        image = input.image * 2.0 - 1.0  # [b, v, c, h, w], range (0,1) to (-1,1)
        b, v_input, c, h, w = image.shape
        image_all = data["image"] * 2.0 - 1.0  # [b, v_all, c, h, w], range (0,1) to (-1,1)
        v_all = image_all.shape[1]
        v_target = (
            target.image.shape[1] if hasattr(target, "image") and target.image is not None else 0
        )
        device = image.device
        batch_idx = torch.arange(b).unsqueeze(1).to(device)

        """se3 pose prediction for all views"""
        # tokenize images, add spatial-temporal p.e.
        img_tokens = self.image_tokenizer(image_all)  # [b * v, n, d]
        _, n, d = img_tokens.shape
        if self.use_pe_embedding_layer:
            img_tokens = self.add_sptial_temporal_pe(img_tokens, b, v_all, h, w)
        img_tokens = rearrange(img_tokens, "(b v) n d -> b (v n) d", b=b, v=v_all)  # [b, v * n, d]

        # get camera tokens, add temporal p.e.
        cam_tokens = self.get_camera_tokens(b, v_all)  # [b, v_all * n_cam, d]
        n_cam = cam_tokens.shape[1] // v_all
        assert n_cam == 1
        cam_tokens = rearrange(cam_tokens, "b (v n) d -> b v n d", v=v_all)  # [b, v_all, n_cam, d]
        cam_tokens = rearrange(cam_tokens, "b v n d -> b (v n) d")  # [b, v_all * n_cam, d]

        # pose estimation for all views
        all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)
        all_tokens = self.run_encoder(all_tokens)
        cam_tokens, _ = all_tokens.split([v_all * n_cam, v_all * n], dim=1)

        # get se3 poses and intrinsics
        cam_tokens = rearrange(cam_tokens, "b (v n) d -> (b v) n d", b=b, v=v_all, n=n_cam)[
            :, 0
        ]  # [b * v_all, d]
        cam_info = self.pose_predictor(
            cam_tokens, v_all
        )  # [b * v_all, num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
        c2w, fxfycxcy = get_cam_se3(cam_info)  # [b * v_all,4,4], [b * v_all,4]
        normalized = True

        # get plucker ray and embeddings
        plucker_rays = cam_info_to_plucker(
            c2w, fxfycxcy, self.config.model.target_image, normalized=normalized
        )
        plucker_rays = rearrange(plucker_rays, "(b v) c h w -> b v c h w", b=b, v=v_all)
        # Split plucker rays into input and target
        plucker_input = plucker_rays[:, :v_input]  # [b, v_input, 6, h, w]
        plucker_target = (
            plucker_rays[:, v_input : v_input + v_target] if v_target > 0 else None
        )  # [b, v_target, 6, h, w]

        plucker_emb_input_tokens = self.target_pose_tokenizer(plucker_input)  # [b * v_input, n, d]
        plucker_emb_input_render_tokens = self.target_pose_tokenizer2(
            plucker_input
        )  # [b * v_input, n, d]
        plucker_emb_target_tokens = (
            self.target_pose_tokenizer2(plucker_target) if plucker_target is not None else None
        )  # [b * v_target, n, d]

        # ===== COPY-PASTE AUGMENTATION MASKS (GROUND TRUTH TRANSIENTS) =====
        copy_paste_mask = data.get("copy_paste_mask", None)
        copy_paste_mask_input = None
        copy_paste_mask_target = None
        copy_paste_mask_input_flat = None
        copy_paste_mask_target_flat = None

        if copy_paste_mask is not None:
            if not isinstance(copy_paste_mask, torch.Tensor):
                copy_paste_mask = torch.tensor(copy_paste_mask)
            copy_paste_mask = copy_paste_mask.to(device=device, dtype=torch.float32)

            if copy_paste_mask.dim() == 4:
                # Assume shape (views, 1, H, W) → add batch dim
                copy_paste_mask = copy_paste_mask.unsqueeze(0)

            # Ensure shape [b, v_all, 1, H, W]
            shape_is_valid = (
                copy_paste_mask.dim() == 5
                and copy_paste_mask.shape[0] == b
                and copy_paste_mask.shape[1] == v_all
            )

            if not shape_is_valid:
                print(
                    "[Model] Warning: copy_paste_mask shape mismatch. "
                    f"Got {copy_paste_mask.shape}, expected ({b}, {v_all}, 1, H, W). "
                    "Ignoring copy-paste masks for this batch."
                )
                copy_paste_mask = None
            else:
                if v_input > 0:
                    copy_paste_mask_input = copy_paste_mask[:, :v_input]
                    copy_paste_mask_input_flat = rearrange(
                        copy_paste_mask_input, "b v c h w -> (b v) c h w"
                    )

                if v_target > 0:
                    copy_paste_mask_target = copy_paste_mask[:, v_input : v_input + v_target]
                    copy_paste_mask_target_flat = rearrange(
                        copy_paste_mask_target, "b v c h w -> (b v) c h w"
                    )

        # ===== PREDICT MOTION MASKS BEFORE RESHAPING =====
        # Use per-view Plücker embeddings [b*v, n, d] for motion mask predictor
        predicted_masks_input = None
        predicted_masks_target = None
        dinov3_masks = None
        if self.use_motion_mask and self.motion_mask_predictor is not None:

            input_image_tokens = rearrange(img_tokens, "b (v n) d -> (b v) n d", v=v_all)[
                : b * v_input
            ]  # [b*v_input, n, d]
            input_image_for_motion = input.image.reshape(b * v_input, c, h, w)
            # Extract target image tokens from all tokens
            target_image_tokens = rearrange(img_tokens, "b (v n) d -> (b v) n d", v=v_all)[
                b * v_input :
            ]  # [b*v_target, n, d]

            # Prepare target images for motion mask prediction (in range [0, 1])
            target_images_for_motion = target.image.reshape(
                b * v_target, c, h, w
            )  # [b*v_target, 3, h, w]

            # Predict motion masks BEFORE reshaping plucker_emb_input
            predicted_masks_input = self.motion_mask_predictor(
                input_image_for_motion, input_image_tokens, plucker_emb_input_tokens
            )  # [b*v_input, 1, 256, 256]

            if v_target > 0:
                predicted_masks_target = self.motion_mask_predictor(
                    target_images_for_motion, target_image_tokens, plucker_emb_target_tokens
                )  # [b*v_target, 1, 256, 256]

            # Override with GT motion masks if provided (inference only)
            # Store flag to skip sigmoid later (GT masks are already binary probabilities)
            using_gt_masks_input = False
            using_gt_masks_target = False
            # Safely check inference config (might be bool or dict)
            inference_cfg = self.config.inference
            if hasattr(inference_cfg, "get"):
                use_gt_motion_masks = inference_cfg.get("use_gt_motion_masks", False)
            else:
                use_gt_motion_masks = False
            if not self.training and use_gt_motion_masks:
                if hasattr(input, "gt_motion_masks") and input.gt_motion_masks is not None:
                    gt_masks = input.gt_motion_masks.to(predicted_masks_input.device).float()
                    predicted_masks_input = gt_masks.view(-1, 1, 256, 256)
                    using_gt_masks_input = True
                    print(
                        "[Model] Using GT motion masks for INPUT views "
                        f"(shape: {predicted_masks_input.shape})"
                    )

                if (
                    v_target > 0
                    and hasattr(target, "gt_motion_masks")
                    and target.gt_motion_masks is not None
                ):
                    gt_masks = target.gt_motion_masks.to(predicted_masks_target.device).float()
                    predicted_masks_target = gt_masks.view(-1, 1, 256, 256)
                    using_gt_masks_target = True
                    print(
                        "[Model] Using GT motion masks for TARGET views "
                        f"(shape: {predicted_masks_target.shape})"
                    )

        # ===== NOW RESHAPE FOR RENDERING PIPELINE =====
        plucker_emb_input_flat = rearrange(
            plucker_emb_input_tokens, "(b v) n d -> b (v n) d", b=b, v=v_input
        )  # [b, v_input * n, d]

        # breakpoint()
        # ## ===== TEMPORARY VISUALIZATION CODE =====
        # import matplotlib.pyplot as plt
        # import numpy as np
        # from pathlib import Path

        # vis_dir = Path("./experiments/temp_vis")
        # vis_dir.mkdir(parents=True, exist_ok=True)

        # # Get first sample from batch
        # sample_idx = 0

        # # Get input images [b, v_input, 3, 256, 256]
        # input_imgs = data["image"][sample_idx, :v_input].cpu()  # [v_input, 3, 256, 256]

        # # Get target images [b, v_target, 3, 256, 256]
        # target_imgs = data["image"][sample_idx, v_input:].cpu()  # [v_target, 3, 256, 256]

        # # Get masks for this sample (detach to avoid gradient issues)
        # input_masks_sample = predicted_masks_input[sample_idx * v_input:(sample_idx + 1) * v_input].detach().cpu()  # [v_input, 1, 256, 256]
        # target_masks_sample = predicted_masks_target[sample_idx * v_target:(sample_idx + 1) * v_target].detach().cpu()  # [v_target, 1, 256, 256]

        # # Convert to numpy and denormalize images
        # def to_numpy_img(img):
        #     # img: [3, 256, 256] in range [0, 1]
        #     img = img.permute(1, 2, 0).numpy()  # [256, 256, 3]
        #     img = np.clip(img, 0, 1)
        #     return img

        # def mask_to_numpy(mask):
        #     # mask: [1, 256, 256] logits
        #     mask_prob = torch.sigmoid(mask).squeeze(0).float().numpy()  # [256, 256]
        #     return mask_prob

        # # Create figure: 2 rows (input, target), columns = max(v_input, v_target)
        # n_cols = max(v_input, v_target)
        # fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
        # if n_cols == 1:
        #     axes = axes.reshape(2, 1)

        # # Row 0: Input views with masks
        # for i in range(n_cols):
        #     ax = axes[0, i]
        #     if i < v_input:
        #         img = to_numpy_img(input_imgs[i])
        #         mask = mask_to_numpy(input_masks_sample[i])

        #         ax.imshow(img)
        #         ax.imshow(mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        #         ax.set_title(f'Input {i}\n(mask overlay)', fontsize=10)
        #     else:
        #         ax.axis('off')
        #     ax.set_xticks([])
        #     ax.set_yticks([])

        # # Row 1: Target views with masks
        # for i in range(n_cols):
        #     ax = axes[1, i]
        #     if i < v_target:
        #         img = to_numpy_img(target_imgs[i])
        #         mask = mask_to_numpy(target_masks_sample[i])

        #         ax.imshow(img)
        #         ax.imshow(mask, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        #         ax.set_title(f'Target {i}\n(mask overlay)', fontsize=10)
        #     else:
        #         ax.axis('off')
        #     ax.set_xticks([])
        #     ax.set_yticks([])

        # plt.suptitle(f'Predicted Motion Masks (Iter {iter}, Sample {sample_idx})', fontsize=14, fontweight='bold')
        # plt.tight_layout()

        # save_path = vis_dir / f'masks_iter_{iter:08d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        # plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.close(fig)

        # print(f"[DEBUG] Saved mask visualization to {save_path}")
        # # ===== END TEMPORARY VISUALIZATION CODE =====
        # breakpoint()

        # Generate DINOv3 pseudo-labels (only after rendering, in loss computation)
        # This will be done in the loss computer to ensure rendered images are available

        """predict scene representation using (posed) input views"""
        # get posed image representation
        img_tokens_input = rearrange(img_tokens, "b (v n) d -> b v n d", v=v_all)[
            :, :v_input
        ]  # [b, v_input, n, d]
        img_tokens_input = rearrange(img_tokens_input, "b v n d -> b (v n) d")
        img_tokens_input = torch.cat(
            [img_tokens_input, plucker_emb_input_flat], dim=-1
        )  # [b, v_input * n, 2d]
        img_tokens_input = self.mlp_fuse(img_tokens_input)  # [b, v_input * n, d]

        # ===== TOKEN MASKING OR DROPPING (APPLY TO INPUT TOKENS) =====
        use_token_dropping = self.config.model.get("use_token_dropping", True)

        if use_token_dropping:
            img_tokens_input_masked_fused, input_patch_mask = self._apply_token_dropping(
                img_tokens_input,
                predicted_masks_input,
                b,
                v_input,
                n,
                d,
                dataset_sources=data.get("dataset_source", [None] * b),
                using_gt_masks=using_gt_masks_input,
                copy_paste_mask_input=copy_paste_mask_input_flat,
            )
        else:
            img_tokens_input_masked_fused, input_patch_mask = self._apply_mae_masking(
                img_tokens_input,
                predicted_masks_input,
                b,
                v_input,
                n,
                d,
                dataset_sources=data.get("dataset_source", [None] * b),
                using_gt_masks=using_gt_masks_input,
                copy_paste_mask_input=copy_paste_mask_input_flat,
            )

        ## TEMPORARY VISUALIZATION CODE ###
        ## ===== VISUALIZATION =====
        # breakpoint()
        # from pathlib import Path
        # sample_idx = 0
        # vis_dir = Path("./temp_vis")
        # vis_dir.mkdir(parents=True, exist_ok=True)

        # # Input images and their patch-masks (these masks include any RE10K random-dropout logic)
        # input_imgs = data["image"][sample_idx, :v_input].cpu()  # [v_input, 3, 256, 256]
        # input_masks = input_patch_mask[sample_idx * v_input:(sample_idx + 1) * v_input].detach().cpu()  # [(v_input),1,Hpatch,Wpatch]

        # # Optional: target masks (if you also predicted them)
        # # target_imgs = data["image"][sample_idx, v_input:].cpu()      # [v_target, 3, 256, 256]
        # # target_masks = target_patch_mask[sample_idx*v_target:(sample_idx+1)*v_target].detach().cpu()

        # # Save a grid for inputs
        # n_cols = v_input
        # fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 3))
        # if n_cols == 1:
        #     axes = [axes]

        # for i in range(v_input):
        #     overlay = self.red_patch_overlay(input_imgs[i], input_masks[i])   # PIL
        #     axes[i].imshow(overlay)
        #     axes[i].set_title(f"Input {i}")
        #     axes[i].axis("off")

        # fig.tight_layout()
        # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_path = vis_dir / f"inputs_overlay_iter_{iter:08d}_{ts}.png"
        # plt.savefig(save_path, dpi=150, bbox_inches="tight")
        # plt.close(fig)
        # print(f"[viz] saved {save_path}")
        # breakpoint()
        ### END TEMPORARY VISUALIZATION CODE ###

        # replicate scene tokens
        scene_tokens = self.scene_code.expand(b, -1, -1)  # [b, n_scene, d]
        n_scene = scene_tokens.shape[1]
        all_tokens = torch.cat(
            [scene_tokens, img_tokens_input_masked_fused], dim=1
        )  # [b, n_scene + v_input * n, d]

        # encoder layers, update scene representation
        all_tokens = self.run_encoder_geom(all_tokens)
        scene_tokens, _ = all_tokens.split(
            [n_scene, v_input * n], dim=1
        )  # [b, n_scene, d], [b, v_input * n, d]

        """render with scene representation from input views and pose of target views"""
        # render target images
        render_input_images = None
        # Safely check inference config (might be bool or dict)
        inference_cfg = self.config.inference
        render_input_preds = False
        if hasattr(inference_cfg, "get"):
            render_input_preds = inference_cfg.get("render_input_predictions", False)

        if v_input > 0 and not self.training and render_input_preds:
            render_input_results = self.render_images(scene_tokens, plucker_emb_input_render_tokens)
            render_input_images = render_input_results.rendered_images

        if plucker_emb_target_tokens is not None and v_target > 0:
            render_results = self.render_images(scene_tokens, plucker_emb_target_tokens)  # dict

            # Generate DINOv3 pseudo-labels after rendering (for motion mask supervision)
            if (
                self.use_dinov3_pseudolabel
                and self.dinov3_pseudolabel_maker is not None
                and v_target > 0
            ):
                # Get dataset sources - PyTorch's default collate converts "dataset_source" (per sample)
                # into "dataset_source" (list) NOT "dataset_sources" (plural)
                dataset_sources = data.get("dataset_source", [None] * b)
                if not isinstance(dataset_sources, list):
                    # Fallback: if it's a single string, replicate it
                    dataset_sources = [dataset_sources] * b
                dinov3_masks_list = []
                cache_hits = 0
                cache_misses = 0

                for batch_idx_i in range(b):
                    is_re10k = dataset_sources[batch_idx_i] == "re10k"
                    uid = int(target.index[batch_idx_i, 0, -1].item())

                    sample_copy_mask = None
                    if copy_paste_mask_target is not None:
                        sample_copy_mask = copy_paste_mask_target[batch_idx_i]
                        if sample_copy_mask.numel() == 0:
                            sample_copy_mask = None

                    if sample_copy_mask is not None:
                        mask_per_sample = sample_copy_mask
                    elif is_re10k:
                        # Static RE10K - skip DINOv3 computation, just use all-zeros
                        # This teaches the model: "static scenes → zero motion mask"
                        mask_per_sample = torch.zeros(v_target, 1, 256, 256, device=device)
                    else:
                        # Dynamic DRE10K - compute or load from cache
                        # Try loading from cache if enabled
                        # SAFE during motion mask training: rendering is frozen, so masks are deterministic!
                        cached_mask = None
                        if self.enable_mask_cache:
                            cached_mask = self._try_load_mask_from_cache(uid)
                            if cached_mask is not None:
                                mask_per_sample = cached_mask.to(device)
                                cache_hits += 1

                        # Compute if cache miss
                        if cached_mask is None:
                            cache_misses += 1

                            # Dynamic DRE10K - generate actual pseudo-labels with DINOv3
                            # Prepare GT and rendered images for this batch element
                            gt_sample = target.image[batch_idx_i]  # [v_target, 3, h, w]
                            pred_sample = render_results.rendered_images[
                                batch_idx_i
                            ]  # [v_target, 3, h, w]

                            # Generate pseudo-labels using DINOv3 with debug visualization
                            debug_path = None
                            if create_visual and batch_idx_i == 0:
                                # Save debug visualization for first batch only
                                if not dist.is_initialized() or dist.get_rank() == 0:
                                    debug_path = (
                                        f"./debug_dinov3/step_{iter:06d}_batch{batch_idx_i}.png"
                                    )

                            mask_per_sample = self.dinov3_pseudolabel_maker(
                                gt_sample, pred_sample, debug_save_path=debug_path
                            )  # [v_target, 1, 256, 256]

                            # Save to cache for future use
                            if self.enable_mask_cache:
                                self._save_mask_to_cache(uid, mask_per_sample)

                    dinov3_masks_list.append(mask_per_sample)

                # Log cache stats every iteration (only for DRE10K samples that were actually computed)
                if self.enable_mask_cache and cache_misses > 0:
                    total_dre10k = cache_hits + cache_misses
                    hit_rate = cache_hits / total_dre10k * 100
                    print(
                        f"[Iter {iter}] DINOv3 mask cache (DRE10K only): {cache_hits}/{total_dre10k} hits ({hit_rate:.1f}% hit rate)"
                    )

                dinov3_masks = torch.stack(dinov3_masks_list, dim=0)  # [b, v_target, 1, 256, 256]
                dinov3_masks = rearrange(
                    dinov3_masks, "b v c h w -> (b v) c h w"
                )  # [b*v_target, 1, 256, 256]

        if dinov3_masks is not None:
            # NOTE: Copy-paste masks are already merged per-sample in dinov3_masks_list
            # at line 2051-2052. No need to replace here.

            # compute loss with motion mask distillation
            # Pass dataset_sources to enable per-dataset masking logic (RE10K vs DRE10K)
            dataset_sources = data.get("dataset_source", None)
            target_images_for_loss = (
                target.image_clean if hasattr(target, "image_clean") else target.image
            )
            loss_metrics = self.loss_computer(
                render_results.rendered_images,
                target_images_for_loss,
                input,
                predicted_target_masks=predicted_masks_target,
                dinov3_target_masks=dinov3_masks,
                predicted_input_masks=predicted_masks_input,
                copy_paste_input_masks=copy_paste_mask_input_flat,
                create_visual=create_visual,
                dataset_sources=dataset_sources,
            )
        else:
            # No pseudo-labels (e.g. inference with use_dinov3_pseudolabel=false).
            # Keep the real render_results from above; only produce a dummy loss.
            device = scene_tokens.device
            loss_metrics = edict(
                loss=torch.tensor(0.0, device=device, requires_grad=True),
                l2_loss=torch.tensor(0.0, device=device),
                psnr=torch.tensor(0.0, device=device),
                lpips_loss=torch.tensor(0.0, device=device),
                perceptual_loss=torch.tensor(0.0, device=device),
                ssim_loss=torch.tensor(0.0, device=device),
                visual=None,
            )
            if v_target == 0 or plucker_emb_target_tokens is None:
                render_results = edict(rendered_images=None)

        vis_only_results = None
        if (
            create_visual
            and v_target > 0
            and not (self.training and self.copy_paste_training_enabled)
        ):
            with torch.no_grad():
                c2w_all_reshaped = rearrange(c2w, "(b v) c d -> b v c d", v=v_all)
                fxfycxcy_all_reshaped = rearrange(fxfycxcy, "(b v) c -> b v c", v=v_all)

                c2w_target = c2w_all_reshaped[
                    :, v_input : v_input + v_target
                ]  # [b, v_target, 4, 4]
                fxfycxcy_target = fxfycxcy_all_reshaped[
                    :, v_input : v_input + v_target
                ]  # [b, v_target, 4]
                c2w_target = rearrange(c2w_target, "b v c d -> (b v) c d")
                fxfycxcy_target = rearrange(fxfycxcy_target, "b v c -> (b v) c")
                vis_only_results = self.render_images_video(
                    scene_tokens, c2w_target, fxfycxcy_target, normalized=normalized
                )

        # Reshape masks for visualization (from flattened to per-batch)
        if dinov3_masks is not None:
            dinov3_masks_viz = rearrange(dinov3_masks, "(b v) c h w -> b v c h w", b=b, v=v_target)
        else:
            dinov3_masks_viz = None

        if predicted_masks_target is not None:
            predicted_masks_viz = rearrange(
                predicted_masks_target, "(b v) c h w -> b v c h w", b=b, v=v_target
            )
        else:
            predicted_masks_viz = None

        # return results
        if hasattr(target, "image_clean"):
            target.image_aug = target.image
            target.image = target.image_clean

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=render_results.rendered_images,
            c2w=rearrange(c2w, "(b v) c d -> b v c d", b=b, v=v_all),
            fxfycxcy=fxfycxcy,
        )

        if render_input_images is not None:
            result.render_input = render_input_images

        # Add motion mask visualization keys if available
        if dinov3_masks_viz is not None:
            result.dinov3_masks = dinov3_masks_viz.detach()
        if predicted_masks_viz is not None:
            result.predicted_target_masks = predicted_masks_viz.detach()

        if input_patch_mask is not None:
            result.input_patch_mask = input_patch_mask.detach()  # [b*v_input, 1, H_patch, W_patch]
        if predicted_masks_input is not None:
            result.predicted_input_masks = (
                predicted_masks_input.detach()
            )  # [b*v_input, 1, H_patch, W_patch]
        if predicted_masks_target is not None:
            result.predicted_target_masks = (
                predicted_masks_target.detach()
            )  # [b*v_target, 1, H_patch, W_patch]

        if vis_only_results is not None:
            result.video_rendering = vis_only_results.rendered_images_video.detach()

        return result

    def render_images(self, scene_tokens, target_tokens):
        """
        Render target views based on the scene representation, target view pose tokens and target view tokens
        Args:
            scene_tokens: [b, n_scene, d]
            target_tokens: plucker embedding tokens of input images [b*v, n_target, d]
        Return:
            rendered_images: rendered target views in [b,v,c,h,w]
        """
        b, _, d = scene_tokens.shape
        bv = target_tokens.shape[0]
        v = bv // b

        # repeat scene tokens
        scene_tokens = scene_tokens.unsqueeze(1).repeat(1, v, 1, 1)
        scene_tokens = rearrange(scene_tokens, "b v n d -> (b v) n d")  # [b*v, n_scene, d]

        # get all tokens
        n_target, n_scene = target_tokens.shape[1], scene_tokens.shape[1]
        all_tokens = torch.cat(
            [target_tokens, scene_tokens], dim=1
        )  # [b*v, n_target+n_pose+n_scene, d]

        # render
        rendered_images_all = self.render(all_tokens, n_target, n_scene, v)  # [b,v,c,h,w]

        render_results = edict(rendered_images=rendered_images_all)
        return render_results

    def render(self, all_tokens, n_target, n_scene, v):
        """
        Run decoder layers and output mlp to render target views
        Args:
            all_tokens: [b*v, n_target+n_pose+n_scene, d]
        Return:
            rendered views: [b, v, c, h, w]
        """
        all_tokens = self.decoder_ln(all_tokens)
        all_tokens = self.run_decoder(all_tokens)

        # split tokens
        target_tokens, _ = all_tokens.split([n_target, n_scene], dim=1)  # [b*v, n_target, d]

        # regress image
        rendered_images_all = self.image_token_decoder(target_tokens)  # [b*v, n_target, p*p*3]
        patch_size = self.config.model.target_image.patch_size
        rendered_images_all = rearrange(
            rendered_images_all,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v,
            h=self.target_latent_h,
            w=self.target_latent_w,
            p1=patch_size,
            p2=patch_size,
            c=3,
        )
        return rendered_images_all

    def get_camera_tokens(self, b, v):
        n, d = self.cam_code.shape[-2:]

        cam_tokens = rearrange(self.cam_code, "n d -> 1 1 n d")  # [1, 1, n_cam, d]
        cam_tokens = repeat(cam_tokens, "1 1 n d -> 1 v n d", v=v)  # [1, v, n_cam, d]
        cam_tokens = rearrange(cam_tokens, "1 v n d -> 1 (v n) d")  # [1, v*n_cam, d]

        # get temporal pe
        img_indices = torch.arange(v).repeat_interleave(n)  # [v*n_cam]
        img_indices = img_indices.to(cam_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d, pos=img_indices, device=cam_tokens.device
        ).to(
            cam_tokens.dtype
        )  # [v*n_cam, d]
        temporal_pe = temporal_pe.reshape(1, v * n, d)  # [1, v*n_cam, d]
        temporal_pe = self.temporal_pe_embedder(temporal_pe)  # [1, v*n_cam, d]

        return (cam_tokens + temporal_pe).repeat(b, 1, 1)  # [b, v*n_cam, d]

    def add_sptial_temporal_pe(self, img_tokens, b, v, h_origin, w_origin):
        """
        Adding spatial-temporal pe to input image tokens
        Args:
            img_tokens: shape [b*v, n, d]
        Return:
            image tokens with positional embedding
        """
        patch_size = self.config.model.image_tokenizer.patch_size
        num_h_tokens = h_origin // patch_size
        num_w_tokens = w_origin // patch_size
        assert (num_h_tokens * num_w_tokens) == img_tokens.shape[1]
        bv, n, d = img_tokens.shape

        # get temporal pe
        img_indices = torch.arange(v).repeat_interleave(n)  # [v*n]
        img_indices = img_indices.unsqueeze(0).repeat(b, 1).reshape(-1)  # [b*v*n]
        img_indices = img_indices.to(img_tokens.device)
        temporal_pe = get_1d_sincos_pos_emb_from_grid(
            embed_dim=d // 2, pos=img_indices, device=img_tokens.device
        ).to(
            img_tokens.dtype
        )  # [b*v*n, d2]
        temporal_pe = temporal_pe.reshape(b, v, n, d // 2)  # [b,v,n,d2]

        # get spatial pe
        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d // 2,
            grid_size=(num_h_tokens, num_w_tokens),
            device=img_tokens.device,
        ).to(
            img_tokens.dtype
        )  # [n, d3]
        spatial_pe = spatial_pe.reshape(1, 1, n, d // 2).repeat(b, v, 1, 1)  # [b,v,n,d3]

        # embed pe
        pe = self.pe_embedder(
            torch.cat([spatial_pe, temporal_pe], dim=-1).reshape(bv, n, d)
        )  # [b*v,n,d]

        return img_tokens + pe

    def run_layers_encoder(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                concat_nerf_img_tokens = self.transformer_encoder[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens

        return custom_forward

    def run_layers_encoder_geom(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder_geom))):
                concat_nerf_img_tokens = self.transformer_encoder_geom[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens

        return custom_forward

    def run_layers_decoder(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_decoder))):
                concat_nerf_img_tokens = self.transformer_decoder[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens

        return custom_forward

    def run_encoder(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder(i + 1, i + checkpoint_every)(
                    all_tokens_encoder
                )
        return all_tokens_encoder

    def run_encoder_geom(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder_geom), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder_geom(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder_geom(i + 1, i + checkpoint_every)(
                    all_tokens_encoder
                )
        return all_tokens_encoder

    def run_decoder(self, all_tokens_encoder):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_decoder), checkpoint_every):
            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_decoder(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_decoder(i + 1, i + checkpoint_every)(
                    all_tokens_encoder
                )
        return all_tokens_encoder

    def red_patch_overlay(self, img_3chw, patch_mask_1hw, alpha=0.5):
        """
        img_3chw: torch.Tensor [3, H, W], values in [0,1]
        patch_mask_1hw: torch.Tensor [1, Hpatch, Wpatch] with {0,1} per token
        returns: PIL.Image (H, W, 3) with red patches overlayed
        """
        H, W = img_3chw.shape[-2:]
        # nearest upsample so each token becomes a solid square
        mask_hw = F.interpolate(patch_mask_1hw.unsqueeze(0), size=(H, W), mode="nearest").squeeze(
            0
        )  # [1,H,W]
        img = img_3chw.permute(1, 2, 0).cpu().numpy()  # [H,W,3], 0..1
        mask = mask_hw.squeeze(0).cpu().numpy()[..., None]  # [H,W,1], {0,1}

        red = np.zeros_like(img)
        red[..., 0] = 1.0  # pure red
        vis = img * (1.0 - alpha * mask) + red * (alpha * mask)  # simple alpha blend
        vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(vis)

    def render_images_video(self, scene_tokens_all, c2w_all, fxfycxcy_all, normalized=False):
        """
        scene_tokens_all: [b, n_scene, d]
        c2w_all: [b*v, 4, 4]
        fxfycxcy_all: [b*v, 4]
        """
        with torch.no_grad():
            scene_tokens_all = scene_tokens_all.detach()
            c2w_all = c2w_all.detach()
            fxfycxcy_all = fxfycxcy_all.detach()

            b, _, d = scene_tokens_all.shape
            bv = c2w_all.shape[0]
            v = bv // b
            c2w_all = rearrange(c2w_all, "(b v) x y -> b v x y", v=v)
            fxfycxcy_all = rearrange(fxfycxcy_all, "(b v) x -> b v x", v=v)
            device = scene_tokens_all.device

            all_renderings = []
            num_frames = 30
            traj_type = "interpolate"
            order_poses = False

            for i in range(b):
                scene_tokens = scene_tokens_all[i]  # [n_scene, d]
                c2ws = c2w_all[i]  # [v, 4, 4]
                fxfycxcy = fxfycxcy_all[i]  # [v, 4]
                if traj_type == "interpolate":
                    # build Ks from fxfycxcy
                    Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                    Ks[:, 0, 0] = fxfycxcy[:, 0]
                    Ks[:, 1, 1] = fxfycxcy[:, 1]
                    Ks[:, 0, 2] = fxfycxcy[:, 2]
                    Ks[:, 1, 2] = fxfycxcy[:, 3]
                    c2ws, Ks = camera_utils.get_interpolated_poses_many(
                        c2ws[:, :3, :4], Ks, num_frames, order_poses=order_poses
                    )
                    frame_c2ws = torch.cat(
                        [
                            c2ws.to(device),
                            torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(
                                c2ws.shape[0], 1, 1
                            ),
                        ],
                        dim=1,
                    )  # [v',4,4]
                    frame_fxfycxcy = torch.zeros((c2ws.shape[0], 4), device=device)
                    frame_fxfycxcy[:, 0] = Ks[:, 0, 0]  # [v',4]
                    frame_fxfycxcy[:, 1] = Ks[:, 1, 1]
                    frame_fxfycxcy[:, 2] = Ks[:, 0, 2]
                    frame_fxfycxcy[:, 3] = Ks[:, 1, 2]
                elif traj_type == "same":
                    frame_c2ws = c2ws.clone()
                    frame_fxfycxcy = fxfycxcy.clone()
                else:
                    raise NotImplementedError

                plucker_rays = cam_info_to_plucker(
                    frame_c2ws,
                    frame_fxfycxcy,
                    self.config.model.target_image,
                    normalized=normalized,
                )  # [v',6,h,w]
                plucker_embeddings = self.target_pose_tokenizer2(
                    plucker_rays.unsqueeze(0)
                )  # [v',n_target,d]

                v_render = plucker_embeddings.shape[0]

                scene_tokens = scene_tokens.unsqueeze(0).repeat(v_render, 1, 1)  # [v',n_scene,d]
                all_tokens = torch.cat(
                    [plucker_embeddings, scene_tokens], dim=1
                )  # [v', n_target+n_scene, d]

                # render
                n_target, n_scene = plucker_embeddings.shape[1], scene_tokens.shape[1]
                rendered_images_all = self.render(all_tokens, n_target, n_scene, v_render).squeeze(
                    0
                )  # [v',c,h,w]
                all_renderings.append(rendered_images_all)

            all_renderings = torch.stack(all_renderings)  # [b,v',c,h,w]

        render_results = edict(rendered_images_video=all_renderings)
        return render_results

    @torch.no_grad()
    def visualize_dinov3_pseudolabels(self, data_batch, save_dir, render_only=False):
        """
        Visualize complete DINOv3 pseudo-mask generation pipeline.

        Shows the full pipeline: GT, Rendered, SSIM dissim, DINO dissim,
        soft mask (percentile-based), and binary co-seg mask (KMeans+GrabCut).

        Args:
            data_batch: Input batch from dataloader
            save_dir: Directory to save visualizations
            render_only: If True, skip pseudo-label generation (just show GT vs rendered)

        Saves:
            - 4-column grid: GT | Rendered | Soft Mask on GT | Binary Mask on GT
            - Per-view metrics (PSNR, LPIPS, SSIM)
            - Debug visualization with SSIM/DINO/Combined dissimilarity
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        os.makedirs(save_dir, exist_ok=True)

        # Run forward pass to get renderings
        result = self.forward(data_batch, create_visual=False, render_video=False)

        input, target = result.input, result.target
        rendered_images = result.render  # [b, v_target, 3, h, w]

        b = input.image.shape[0]
        v_target = (
            target.image.shape[1] if hasattr(target, "image") and target.image is not None else 0
        )
        h, w = target.image.shape[-2:]
        device = target.image.device

        # Generate complete pseudo-labels (soft + binary)
        soft_masks = None
        binary_masks = None
        debug_info = None

        if not render_only and self.use_dinov3_pseudolabel and self.dinov3_pseudolabel_maker:
            dataset_sources = data_batch.get("dataset_sources", [None] * b)

            soft_masks_list = []
            binary_masks_list = []
            debug_info_list = []

            for batch_idx_i in range(b):
                is_re10k = dataset_sources[batch_idx_i] == "re10k"

                if is_re10k:
                    # Static RE10K - use all-zeros
                    soft_mask = torch.zeros(v_target, 1, h, w, device=device)
                    binary_mask = torch.zeros(v_target, 1, h, w, device=device)
                    debug = None
                else:
                    # Dynamic RE10K - compute full pseudo-mask pipeline
                    gt_sample = target.image[batch_idx_i]  # [v_target, 3, h, w]
                    pred_sample = rendered_images[batch_idx_i]  # [v_target, 3, h, w]

                    # Generate soft mask (DINO + SSIM percentile-based)
                    self.dinov3_pseudolabel_maker.use_coseg_binary = False
                    soft_mask = self.dinov3_pseudolabel_maker(gt_sample, pred_sample)

                    # Generate binary co-seg mask (KMeans + GrabCut)
                    self.dinov3_pseudolabel_maker.use_coseg_binary = True
                    binary_mask = self.dinov3_pseudolabel_maker(gt_sample, pred_sample)

                    # Extract debug info (SSIM, DINO dissimilarity)
                    dino_dissim = self.dinov3_pseudolabel_maker._compute_cosine_dissimilarity(
                        gt_sample, pred_sample
                    )
                    ssim_dissim = self.dinov3_pseudolabel_maker._compute_ssim_dissimilarity(
                        gt_sample, pred_sample
                    )
                    combined_dissim = torch.minimum(dino_dissim, ssim_dissim)

                    debug = {
                        "dino_dissim": dino_dissim.cpu(),
                        "ssim_dissim": ssim_dissim.cpu(),
                        "combined_dissim": combined_dissim.cpu(),
                    }

                soft_masks_list.append(soft_mask)
                binary_masks_list.append(binary_mask)
                debug_info_list.append(debug)

            soft_masks = torch.stack(soft_masks_list, dim=0)  # [b, v_target, 1, h, w]
            binary_masks = torch.stack(binary_masks_list, dim=0)  # [b, v_target, 1, h, w]
            debug_info = debug_info_list

        # Compute metrics
        psnr_per_view = []
        lpips_per_view = []
        ssim_per_view = []

        for batch_idx_i in range(b):
            psnr = compute_psnr(target.image[batch_idx_i], rendered_images[batch_idx_i])
            lpips = compute_lpips(target.image[batch_idx_i], rendered_images[batch_idx_i])
            ssim = compute_ssim(
                target.image[batch_idx_i].to(torch.float32),
                rendered_images[batch_idx_i].to(torch.float32),
            )
            psnr_per_view.append(psnr)
            lpips_per_view.append(lpips)
            ssim_per_view.append(ssim)

        # Save visualizations
        for batch_idx_i in range(b):
            scene_idx = int(target.index[batch_idx_i, 0, -1].item())
            dataset_source = data_batch.get("dataset_sources", ["unknown"] * b)[batch_idx_i]

            # Get scene name and original filenames for saving masks
            scene_name = data_batch.get("scene_name", [f"scene_{scene_idx:06d}"] * b)[batch_idx_i]
            original_filenames = data_batch.get("original_filenames", [None] * b)[batch_idx_i]

            scene_dir = os.path.join(save_dir, f"scene_{scene_idx:06d}_{dataset_source}")
            os.makedirs(scene_dir, exist_ok=True)

            # Create masks directory for .npy files
            masks_dir = os.path.join(scene_dir, "masks_npy")
            os.makedirs(masks_dir, exist_ok=True)

            gt_imgs = target.image[batch_idx_i].cpu()  # [v_target, 3, h, w]
            rendered_imgs = rendered_images[batch_idx_i].cpu()  # [v_target, 3, h, w]

            v_target_curr = gt_imgs.shape[0]

            # Save binary masks as .npy files with original image names
            if binary_masks is not None and original_filenames is not None:
                binary_masks_curr = binary_masks[batch_idx_i].cpu().numpy()  # [v_target, 1, h, w]
                soft_masks_curr_np = soft_masks[batch_idx_i].cpu().numpy()  # [v_target, 1, h, w]

                # Get number of input views to correctly index target filenames
                v_input = input.image.shape[1]

                for v_idx in range(v_target_curr):
                    # Get original filename for this target view
                    # original_filenames contains all views (input + target)
                    # Target views start after input views
                    original_path = original_filenames[v_input + v_idx]
                    # Extract filename without extension (e.g., "00042" from "images/scene/00042.png")
                    from pathlib import Path

                    image_name = Path(original_path).stem  # e.g., "00042"

                    # Save binary mask as .npy
                    binary_mask_np = binary_masks_curr[v_idx, 0]  # [h, w]
                    binary_mask_path = os.path.join(masks_dir, f"{image_name}_binary.npy")
                    np.save(binary_mask_path, binary_mask_np.astype(np.float32))

                    # Also save soft mask for comparison
                    soft_mask_np = soft_masks_curr_np[v_idx, 0]  # [h, w]
                    soft_mask_path = os.path.join(masks_dir, f"{image_name}_soft.npy")
                    np.save(soft_mask_path, soft_mask_np.astype(np.float32))

                # Save a metadata file for this scene
                metadata = {
                    "scene_name": scene_name,
                    "scene_idx": scene_idx,
                    "dataset_source": dataset_source,
                    "num_views": v_target_curr,
                    "mask_files": [
                        f"{Path(original_filenames[v_input + v_idx]).stem}_binary.npy"
                        for v_idx in range(v_target_curr)
                    ],
                    "original_filenames": [
                        original_filenames[v_input + v_idx] for v_idx in range(v_target_curr)
                    ],
                }
                metadata_path = os.path.join(masks_dir, "metadata.json")
                import json

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"  ✓ Saved {v_target_curr} mask pairs (.npy) for scene {scene_name}")

            # Main visualization: GT | Rendered | Soft Mask on GT | Binary Mask on GT
            if soft_masks is not None and binary_masks is not None:
                fig, axes = plt.subplots(v_target_curr, 4, figsize=(20, 5 * v_target_curr))
                if v_target_curr == 1:
                    axes = axes.reshape(1, -1)

                soft_masks_curr = soft_masks[batch_idx_i].cpu()  # [v_target, 1, h, w]
                binary_masks_curr = binary_masks[batch_idx_i].cpu()  # [v_target, 1, h, w]
                cmap = cm.get_cmap("hot")

                for v_idx in range(v_target_curr):
                    gt_np = gt_imgs[v_idx].permute(1, 2, 0).float().numpy()
                    rendered_np = rendered_imgs[v_idx].permute(1, 2, 0).float().numpy()
                    soft_mask_np = soft_masks_curr[v_idx, 0].float().numpy()
                    binary_mask_np = binary_masks_curr[v_idx, 0].float().numpy()

                    psnr_val = psnr_per_view[batch_idx_i][v_idx].item()
                    lpips_val = lpips_per_view[batch_idx_i][v_idx].item()
                    ssim_val = ssim_per_view[batch_idx_i][v_idx].item()

                    # Column 1: GT
                    axes[v_idx, 0].imshow(gt_np)
                    title = f"GT View {v_idx}"
                    if v_idx == 0:
                        title += f"\n({dataset_source.upper()})"
                    axes[v_idx, 0].set_title(title, fontsize=12, fontweight="bold")
                    axes[v_idx, 0].axis("off")

                    # Column 2: Rendered
                    axes[v_idx, 1].imshow(rendered_np)
                    axes[v_idx, 1].set_title(
                        f"Rendered\nPSNR:{psnr_val:.2f} SSIM:{ssim_val:.3f}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    axes[v_idx, 1].axis("off")

                    # Column 3: Soft Mask on GT
                    axes[v_idx, 2].imshow(gt_np)
                    mask_colored = cmap(soft_mask_np)[:, :, :3]
                    axes[v_idx, 2].imshow(mask_colored, alpha=0.6)
                    axes[v_idx, 2].set_title(
                        f"Soft Mask (Percentile)\n[0,1] continuous", fontsize=12, fontweight="bold"
                    )
                    axes[v_idx, 2].axis("off")

                    # Column 4: Binary Mask on GT
                    axes[v_idx, 3].imshow(gt_np)
                    binary_colored = cmap(binary_mask_np)[:, :, :3]
                    axes[v_idx, 3].imshow(binary_colored, alpha=0.6)
                    axes[v_idx, 3].set_title(
                        f"Binary Mask (Co-Seg)\n{{0,1}} KMeans+GrabCut",
                        fontsize=12,
                        fontweight="bold",
                    )
                    axes[v_idx, 3].axis("off")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(scene_dir, "pseudolabels.png"), dpi=150, bbox_inches="tight"
                )
                plt.close()

                # Debug visualization: SSIM | DINO | Combined dissimilarity
                if debug_info[batch_idx_i] is not None:
                    debug = debug_info[batch_idx_i]
                    fig, axes = plt.subplots(v_target_curr, 3, figsize=(15, 5 * v_target_curr))
                    if v_target_curr == 1:
                        axes = axes.reshape(1, -1)

                    for v_idx in range(v_target_curr):
                        dino = debug["dino_dissim"][v_idx, 0].float().numpy()
                        ssim_d = debug["ssim_dissim"][v_idx, 0].float().numpy()
                        combined = debug["combined_dissim"][v_idx, 0].float().numpy()

                        # SSIM dissimilarity
                        im = axes[v_idx, 0].imshow(ssim_d, cmap="hot", vmin=0, vmax=1)
                        axes[v_idx, 0].set_title(
                            f"SSIM Dissimilarity\nView {v_idx}", fontsize=12, fontweight="bold"
                        )
                        axes[v_idx, 0].axis("off")
                        plt.colorbar(im, ax=axes[v_idx, 0], fraction=0.046)

                        # DINO dissimilarity
                        im = axes[v_idx, 1].imshow(dino, cmap="hot", vmin=0, vmax=1)
                        axes[v_idx, 1].set_title(
                            f"DINO Dissimilarity\nView {v_idx}", fontsize=12, fontweight="bold"
                        )
                        axes[v_idx, 1].axis("off")
                        plt.colorbar(im, ax=axes[v_idx, 1], fraction=0.046)

                        # Combined (min)
                        im = axes[v_idx, 2].imshow(combined, cmap="hot", vmin=0, vmax=1)
                        axes[v_idx, 2].set_title(
                            f"Combined (min)\nView {v_idx}", fontsize=12, fontweight="bold"
                        )
                        axes[v_idx, 2].axis("off")
                        plt.colorbar(im, ax=axes[v_idx, 2], fraction=0.046)

                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(scene_dir, "debug_dissimilarity.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

            else:
                # Fallback: GT vs Rendered only
                fig, axes = plt.subplots(v_target_curr, 2, figsize=(10, 5 * v_target_curr))
                if v_target_curr == 1:
                    axes = axes.reshape(1, -1)

                for v_idx in range(v_target_curr):
                    gt_np = gt_imgs[v_idx].permute(1, 2, 0).float().numpy()
                    rendered_np = rendered_imgs[v_idx].permute(1, 2, 0).float().numpy()

                    psnr_val = psnr_per_view[batch_idx_i][v_idx].item()
                    ssim_val = ssim_per_view[batch_idx_i][v_idx].item()

                    axes[v_idx, 0].imshow(gt_np)
                    axes[v_idx, 0].set_title(f"GT View {v_idx}", fontsize=12)
                    axes[v_idx, 0].axis("off")

                    axes[v_idx, 1].imshow(rendered_np)
                    axes[v_idx, 1].set_title(
                        f"Rendered\nPSNR:{psnr_val:.2f} SSIM:{ssim_val:.3f}", fontsize=12
                    )
                    axes[v_idx, 1].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(scene_dir, "rendering.png"), dpi=150, bbox_inches="tight")
                plt.close()

            # Save metrics
            with open(os.path.join(scene_dir, "metrics.txt"), "w") as f:
                f.write(f"Scene: {scene_idx:06d}\n")
                f.write(f"Dataset: {dataset_source}\n")
                f.write(f"Num views: {v_target_curr}\n\n")
                f.write("Per-view metrics:\n")
                for v_idx in range(v_target_curr):
                    f.write(f"  View {v_idx}: ")
                    f.write(f"PSNR={psnr_per_view[batch_idx_i][v_idx].item():.2f}, ")
                    f.write(f"LPIPS={lpips_per_view[batch_idx_i][v_idx].item():.4f}, ")
                    f.write(f"SSIM={ssim_per_view[batch_idx_i][v_idx].item():.4f}\n")
                f.write(f"\nAverage:\n")
                f.write(f"  PSNR: {psnr_per_view[batch_idx_i].mean().item():.2f}\n")
                f.write(f"  LPIPS: {lpips_per_view[batch_idx_i].mean().item():.4f}\n")
                f.write(f"  SSIM: {ssim_per_view[batch_idx_i].mean().item():.4f}\n")

            print(
                f"✓ Saved scene {scene_idx:06d} ({dataset_source}): "
                f"PSNR={psnr_per_view[batch_idx_i].mean().item():.2f}"
            )

        print(f"\n✓ Visualization complete! Saved to: {save_dir}")

    @torch.no_grad()
    def save_visuals(self, out_dir, result, batch, save_all=False):
        os.makedirs(out_dir, exist_ok=True)

        input, target = result.input, result.target

        # save comparison
        if result.loss_metrics.visual is not None:
            uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]

            uid_firstlast = f"{uids[0]:08}_{uids[-1]:08}"
            Image.fromarray(result.loss_metrics.visual).save(
                os.path.join(out_dir, f"supervision_{uid_firstlast}.jpg")
            )
            with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
                uids = "_".join([f"{uid:08}" for uid in uids])
                f.write(uids)

        # save input, rendered interpolated imaged, video
        for b in range(input.image.size(0)):
            uid = input.index[b, 0, -1].item()

            # vis input
            vis_image = rearrange(input.image[b], "v c h w -> h (v w) c")
            vis_image = (vis_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(vis_image).save(os.path.join(out_dir, f"input_{uid}.png"))

            if ("render_video" in result.keys()) and torch.is_tensor(result.render_video):
                render_video = result.render_video[b].detach().cpu()  # [v, c, h, w]
                render_video = np.ascontiguousarray(np.array(render_video.to(torch.float32)))
                render_video = rearrange(render_video, "v c h w -> v h w c")
                create_video_from_frames(
                    render_video, f"{out_dir}/render_video_{uid}.mp4", framerate=30
                )

            if not save_all:
                break

    @torch.no_grad()
    def save_evaluations(self, out_dir, result, batch, dataset):
        os.makedirs(out_dir, exist_ok=True)
        input, target = result.input, result.target
        for b in range(input.image.size(0)):
            uid = input.index[b, 0, -1].item()
            curr_out_dir = os.path.join(out_dir, "%06d" % (uid))
            os.makedirs(curr_out_dir, exist_ok=True)
            curr_out_color_dir = curr_out_dir
            os.makedirs(curr_out_color_dir, exist_ok=True)

            target_idx = target.index[b, :, 0].cpu().numpy()
            _, v, _, h, w = input.image.size()

            # vis gt and predicted target views
            im = torch.cat((result.target.image[b], result.render[b]), dim=2).detach().cpu()
            im = rearrange(im, "v c h w -> h (v w) c")
            im = (im.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(im).save(os.path.join(curr_out_color_dir, f"gt_vs_pred.png"))

            # vis input views
            input_im = result.input.image[b]  # V C H W
            input_im = rearrange(input_im, "v c h w -> h (v w) c")
            input_im = (input_im.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            Image.fromarray(input_im).save(os.path.join(curr_out_color_dir, f"input.png"))

            # # vis interpolated views
            # if torch.is_tensor(result.render_interpolate):
            #     interpolate_im = result.render_interpolate[b].detach().cpu().to(torch.float32)
            #     interpolate_im = rearrange(interpolate_im, 'v c h w -> h (v w) c')
            #     interpolate_im = (interpolate_im.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
            #     Image.fromarray(interpolate_im).save(
            #         os.path.join(curr_out_color_dir, f"pred_interpolate.png"))

            # get metrics
            psnr = compute_psnr(target.image[b], result.render[b])  # .mean()
            lpips = compute_lpips(target.image[b], result.render[b])  # .mean()
            ssim = compute_ssim(
                target.image[b].to(torch.float32), result.render[b].to(torch.float32)
            )  # .mean()

            with open(os.path.join(curr_out_dir, "per_view_metrics.txt"), "w") as f:
                for i in range(psnr.shape[0]):
                    f.write(f"view: {target_idx[i]}, psnr: {psnr[i]},")
                    f.write(f"lpips: {lpips[i]},")
                    f.write(f"ssim: {ssim[i]}\n")

            with open(os.path.join(curr_out_dir, "metrics.txt"), "w") as f:
                metrics = f"psnr: {psnr.mean()}\nlpips: {lpips.mean()}\nssim: {ssim.mean()}\n"
                f.write(metrics)

            # save rendered video
            if torch.is_tensor(result.get("render_video", None)):
                all_frames = result.render_video[b].detach().cpu()  # [v h w c]
                all_frames = np.ascontiguousarray(np.array(all_frames.to(torch.float32)))
                all_frames = rearrange(all_frames, "v c h w -> v h w c")
                create_video_from_frames(all_frames, f"{curr_out_dir}/input_traj.mp4", framerate=30)

            # save rendered video
            if torch.is_tensor(result.get("render_video", None)):
                all_frames = result.render_video[b].detach().cpu()  # [v h w c]
                all_frames = np.ascontiguousarray(np.array(all_frames.to(torch.float32)))
                all_frames = rearrange(all_frames, "v c h w -> v h w c")
                create_video_from_frames(all_frames, f"{curr_out_dir}/input_traj.mp4", framerate=30)

            # visualize cameras (placeholder - implement if needed)
            # c2w = result.c2w[b].detach().cpu()
            # input_idx_cur = input_idx[b].detach().cpu()
            # c2w_input = c2w[input_idx_cur]
            # TODO: Implement camera trajectory visualization if needed
            # pass


def get_cam_se3(cam_info):
    """
    cam_info: [b,num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
    """
    b, n = cam_info.shape

    if n == 13:
        rot_6d = cam_info[:, :6]
        R = rot6d2mat(rot_6d)  # [b,3,3]
        t = cam_info[:, 6:9].unsqueeze(-1)  # [b,3,1]
        fxfycxcy = cam_info[:, 9:]  # normalized by resolution / shift from average, [b,4]
    elif n == 11:
        rot_quat = cam_info[:, :4]
        R = quat2mat(rot_quat)
        t = cam_info[:, 4:7].unsqueeze(-1)  # [b,3,1]
        fxfycxcy = cam_info[:, 7:]  # normalized by resolution / shift from average, [b,4]
    else:
        raise NotImplementedError

    Rt = torch.cat([R, t], dim=2)  # [b,3,4]
    c2w = torch.cat(
        [
            Rt,
            torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
            .view(1, 1, 4)
            .repeat(b, 1, 1),
        ],
        dim=1,
    )  # [b,4,4]
    return c2w, fxfycxcy


def cam_info_to_plucker(c2w, fxfycxcy, target_imgs_info, normalized=True):
    """
    c2w: [b,4,4]
    fxfycxcy: [b,4]
    """
    b = c2w.shape[0]
    device = c2w.device
    h, w = target_imgs_info.height, target_imgs_info.width

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)  # [b,3,h,w]
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    plucker = torch.cat(
        [
            torch.cross(ray_o, ray_d, dim=1),
            ray_d,
        ],
        dim=1,
    )
    return plucker  # [b,c=6,h,w]


class PoseEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.pose_rep = self.config.model.pose_latent.get("representation", "6d")
        print("Pose representation:", self.pose_rep)
        if self.pose_rep == "6d":
            self.num_pose_element = 6
        elif self.pose_rep == "quat":
            self.num_pose_element = 4
        else:
            raise NotImplementedError

        self.rel_head = nn.Sequential(
            nn.Linear(
                config.model.transformer.d * 2,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                self.num_pose_element + 3,
                bias=True,
            ),
        )
        self.rel_head.apply(_init_weights)

        self.canonical_k_head = nn.Sequential(
            nn.Linear(
                config.model.transformer.d,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                1,
                bias=True,
            ),
        )
        self.canonical_k_head.apply(_init_weights)

        self.f_bias = 1.25

    def forward(self, x, v):
        """
        x: [b*v, d]
        """
        canonical = self.config.model.pose_latent.get("canonical", "first")
        x = rearrange(x, "(b v) d -> b v d", v=v)
        b = x.shape[0]
        if canonical == "first":
            x_canonical = x[:, 0:1]  # [b,1,d]
            x_rel = x[:, 1:]  # [b,v-1,d]
        elif canonical == "middle":
            cano_idx = v // 2
            rel_indices = torch.cat([torch.arange(cano_idx), torch.arange(cano_idx + 1, v)])
            x_canonical = x[:, cano_idx : cano_idx + 1]  # [b,1,d]
            x_rel = x[:, rel_indices]  # [b,v-1,d]
        else:
            raise NotImplementedError

        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias  # [b,1]
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)  # [b,1,2]

        if self.pose_rep == "6d":
            rt_canonical = (
                torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0])
                .reshape(1, 1, 9)
                .to(fxfy_canonical)
                .repeat(b, 1, 1)
            )  # [b,1,9]
        elif self.pose_rep == "quat":
            rt_canonical = (
                torch.tensor([1, 0, 0, 0, 0, 0, 0])
                .reshape(1, 1, 7)
                .to(fxfy_canonical)
                .repeat(b, 1, 1)
            )  # [b,1,7]
        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)  # [b,1,11]

        feat_rel = torch.cat([x_canonical.repeat(1, v - 1, 1), x_rel], dim=-1)  # [b,v-1,2*d]
        info_rel = self.rel_head(feat_rel)  # [b,v-1,num_pose_element+3]
        info_all = info_canonical.repeat(1, v, 1)  # [b,v,num_pose_element+3+2]

        if canonical == "first":
            info_all[:, 1:, : self.num_pose_element + 3] += info_rel
        elif canonical == "middle":
            info_all[:, rel_indices, : self.num_pose_element + 3] += info_rel
        else:
            raise NotImplementedError

        cxcy_all = torch.tensor([0.5, 0.5]).reshape(1, 1, 2).repeat(b, v, 1).to(info_all)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)
        return rearrange(info_all, "b v d -> (b v) d")

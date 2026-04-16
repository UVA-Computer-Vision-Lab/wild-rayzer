#!/usr/bin/env python3
"""
Example script showing how to generate comprehensive pipeline visualizations
using the DinoV3UncertaintyPseudoLabelMaker class.

This creates a figure similar to the one in the paper showing all intermediate steps:
Row 1: GT, Rendering, MSE, DINO, SSIM
Row 2: min(DINO, SSIM), K-means Clusters, FG Selection, Final Mask, Overlay

Usage:
    python scripts/generate_comprehensive_viz.py \
        --gt_image path/to/gt.png \
        --rendered_image path/to/rendered.png \
        --output_path output/comprehensive_viz.png \
        --mode binary  # or "soft"
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModel, AutoImageProcessor


def load_image(image_path, device):
    """Load and preprocess image to tensor [1, 3, H, W] in range [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive pipeline visualization"
    )
    
    # Input/Output
    parser.add_argument("--gt_image", type=str, required=True, help="Path to ground truth image")
    parser.add_argument("--rendered_image", type=str, required=True, help="Path to rendered image")
    parser.add_argument(
        "--output_path",
        type=str,
        default="comprehensive_viz.png",
        help="Output path for visualization",
    )
    
    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov2-large",
        help="DINOv3 model name from HuggingFace",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["soft", "binary"],
        default="binary",
        help="Mask generation mode: 'soft' (percentile) or 'binary' (K-means+GrabCut)",
    )
    parser.add_argument(
        "--target_size", type=int, default=256, help="Target mask resolution"
    )
    
    # Binary mode parameters
    parser.add_argument(
        "--k", type=int, default=64, help="Number of K-means clusters (binary mode)"
    )
    parser.add_argument(
        "--w_dino", type=float, default=0.5, help="Weight for DINO features"
    )
    parser.add_argument(
        "--w_ssim", type=float, default=0.5, help="Weight for SSIM features"
    )
    parser.add_argument(
        "--consistency_min_frames",
        type=int,
        default=4,
        help="Minimum frames for consistency (binary mode, use 1 for single image)",
    )
    
    # Soft mode parameters
    parser.add_argument(
        "--percentile",
        type=int,
        default=75,
        help="Percentile threshold for soft masks (75, 85, or 90)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load DINOv3 model
    print(f"Loading DINOv3 model: {args.model_name}...")
    dinov3_backbone = AutoModel.from_pretrained(args.model_name).to(device).eval()
    dinov3_processor = AutoImageProcessor.from_pretrained(args.model_name)
    print("✓ Model loaded")
    
    # Import the pseudo-label maker
    from model.rayzer_official_v3 import DinoV3UncertaintyPseudoLabelMaker
    
    # Create pseudo-label generator
    use_binary = args.mode == "binary"
    print(f"\nMode: {'Binary (K-means + GrabCut)' if use_binary else 'Soft (Percentile)'}")
    
    pseudo_labeler = DinoV3UncertaintyPseudoLabelMaker(
        dinov3_backbone=dinov3_backbone,
        dinov3_processor=dinov3_processor,
        percentile_threshold=args.percentile,
        use_ssim=True,
        target_size=args.target_size,
        use_coseg_binary=use_binary,
        coseg_k=args.k,
        coseg_consistency_min_frames=args.consistency_min_frames,
        coseg_frame_saliency_quantile=0.75,
        coseg_cluster_top_percent=0.05,
        coseg_morph_kernel=3,
        coseg_morph_iters=1,
        coseg_min_component_area_ratio=0.0025,
        coseg_grabcut_kernel=7,
        w_dino=args.w_dino,
        w_ssim=args.w_ssim,
    )
    
    # Load images
    print(f"\nLoading images...")
    print(f"  GT: {args.gt_image}")
    print(f"  Rendered: {args.rendered_image}")
    gt_img = load_image(args.gt_image, device)
    rendered_img = load_image(args.rendered_image, device)
    print(f"  Image shape: {gt_img.shape}")
    
    # Generate mask with comprehensive visualization
    print(f"\nGenerating motion mask and comprehensive visualization...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        motion_mask = pseudo_labeler(
            gt_img=gt_img,
            pred_img=rendered_img,
            debug_save_path=str(output_path),
            use_comprehensive_viz=True,  # Enable comprehensive visualization!
        )
    
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Mask shape: {motion_mask.shape}")
    print(f"  Mask range: [{motion_mask.min():.3f}, {motion_mask.max():.3f}]")
    print(f"  Mask coverage: {motion_mask.mean():.3f}")
    
    # Save the mask as numpy for inspection
    mask_np_path = output_path.with_suffix(".npy")
    np.save(str(mask_np_path), motion_mask.cpu().numpy())
    print(f"  Mask array saved to: {mask_np_path}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()


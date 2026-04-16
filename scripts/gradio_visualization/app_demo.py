#!/usr/bin/env python3
"""
Gradio Demo for WildRayZer

Upload input/target images and generate:
- Rendered target views
- Ground truth comparison
- PSNR metrics
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent  # Go up from scripts/gradio_visualization/
sys.path.insert(0, str(project_root))

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
import tempfile
import cv2

from easydict import EasyDict as edict
from omegaconf import OmegaConf
from model.rayzer_official_v3 import Images2Latent4D


class WildRayZerDemo:
    def __init__(self, checkpoint_path, config_path):
        print("Loading WildRayZer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        self.config = edict(config)

        # Set inference mode
        self.config.inference = True
        self.config.evaluation = False

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model
        self.model = Images2Latent4D(self.config)

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint["model"], strict=False
        )

        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {unexpected_keys}")

        self.model.to(self.device)
        self.model.eval()

        print(f"✓ WildRayZer loaded on {self.device}")
        print(f"  Encoder layers: {len(self.model.transformer_encoder)}")
        print(f"  Encoder geom layers: {len(self.model.transformer_encoder_geom)}")
        print(f"  Decoder layers: {len(self.model.transformer_decoder)}")
        print(f"  Motion mask enabled: {self.model.use_motion_mask}")
        if self.model.use_motion_mask:
            print(f"  Motion mask predictor: {self.model.motion_mask_predictor is not None}")

    def preprocess_image(self, image_input):
        """Preprocess image to model input format."""
        if isinstance(image_input, tuple):
            image_input = image_input[0]

        if isinstance(image_input, str):
            pil_image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_image = image_input
        else:
            pil_image = Image.fromarray(image_input)

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image = pil_image.resize((256, 256), Image.LANCZOS)
        img_array = np.array(pil_image) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

        return img_tensor

    def create_dummy_batch(self, input_images, target_images):
        """
        Create a batch from uploaded images.

        CRITICAL: Use nested "input"/"target" format to bypass random sampling in ProcessData!
        The experiments version of ProcessData has special handling for this format
        that preserves deterministic ordering without random shuffling.
        """
        num_input = len(input_images)
        num_target = len(target_images)

        print(f"Creating batch: {num_input} input images, {num_target} target images")

        input_tensors = [self.preprocess_image(img) for img in input_images]
        target_tensors = [self.preprocess_image(img) for img in target_images]

        # Create nested format: experiments ProcessData detects this and preserves order!
        input_images_tensor = torch.stack(input_tensors).unsqueeze(0)  # [1, num_input, 3, 256, 256]
        target_images_tensor = torch.stack(target_tensors).unsqueeze(
            0
        )  # [1, num_target, 3, 256, 256]

        # Create dummy camera parameters
        dummy_c2w_input = torch.eye(4).unsqueeze(0).repeat(1, num_input, 1, 1)
        dummy_c2w_target = torch.eye(4).unsqueeze(0).repeat(1, num_target, 1, 1)
        dummy_fxfy_input = torch.tensor([[1.0, 1.0, 0.5, 0.5]]).repeat(1, num_input, 1)
        dummy_fxfy_target = torch.tensor([[1.0, 1.0, 0.5, 0.5]]).repeat(1, num_target, 1)

        # CRITICAL: Model needs concatenated images with c2w and fxfycxcy!
        all_images_tensor = torch.cat([input_images_tensor, target_images_tensor], dim=1)
        v_all = all_images_tensor.shape[1]
        device = all_images_tensor.device

        # Create dummy camera parameters for all views (ON SAME DEVICE!)
        dummy_c2w = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(1, v_all, 1, 1)
        dummy_fxfycxcy = (
            torch.tensor([[1.0, 1.0, 0.5, 0.5]], device=device).unsqueeze(0).repeat(1, v_all, 1)
        )

        # Nested format for ProcessData + flat fields for model
        batch = {
            "input": {
                "image": input_images_tensor,  # [1, num_input, 3, 256, 256]
            },
            "target": {
                "image": target_images_tensor,  # [1, num_target, 3, 256, 256]
            },
            "image": all_images_tensor,  # [1, v_all, 3, 256, 256] - MODEL NEEDS THIS!
            "c2w": dummy_c2w,  # [1, v_all, 4, 4]
            "fxfycxcy": dummy_fxfycxcy,  # [1, v_all, 4]
            "scene_name": "demo_scene",
        }

        print(f"Batch format: nested input/target (experiments-compatible)")
        print(f"  input/image: {input_images_tensor.shape}")
        print(f"  target/image: {target_images_tensor.shape}")
        print(f"  Image range: [{input_images_tensor.min():.3f}, {input_images_tensor.max():.3f}]")

        return batch

    def extract_predicted_masks(self, result, batch_idx, v_input, v_target):
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

    def create_overlay_from_mask(self, base_images, mask_values, alpha=0.6, mode="bilinear"):
        if mask_values is None or base_images is None:
            return None

        if mask_values.ndim != 4 or base_images.ndim != 4:
            return None

        base = base_images.cpu()
        mask = mask_values.cpu().float()
        H, W = base.shape[-2:]

        if mode in ["bilinear", "bicubic"]:
            mask_up = F.interpolate(mask, size=(H, W), mode=mode, align_corners=False)
        else:
            mask_up = F.interpolate(mask, size=(H, W), mode=mode)

        mask_rgb = mask_up.repeat(1, 3, 1, 1)
        red = torch.zeros_like(base)
        red[:, 0, :, :] = 1.0

        overlay = base * (1 - alpha * mask_rgb) + red * (alpha * mask_rgb)
        overlay = overlay.clamp(0.0, 1.0)
        return self.create_image_grid(overlay)

    def create_binary_mask_grid(self, mask_probs, threshold, target_size):
        if mask_probs is None:
            return None

        mask_binary = (mask_probs > threshold).float()
        mask_binary = F.interpolate(mask_binary, size=target_size, mode="nearest")
        mask_rgb = mask_binary.repeat(1, 3, 1, 1)
        return self.create_image_grid(mask_rgb)

    @torch.no_grad()
    def render(self, input_files, target_files, render_video=False, motion_mask_threshold=0.1):
        """Run inference on uploaded images."""
        if not input_files or not target_files:
            return None, None, None, "Please upload both input and target images."

        try:
            # Convert file objects to PIL images
            input_images = [Image.open(f.name).convert("RGB") for f in input_files]
            target_images = [Image.open(f.name).convert("RGB") for f in target_files]

            batch = self.create_dummy_batch(input_images, target_images)

            # Move ALL tensors to device (including nested ones!)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                elif isinstance(batch[key], dict):
                    # Move nested tensors (input/target)
                    for subkey in batch[key]:
                        if isinstance(batch[key][subkey], torch.Tensor):
                            batch[key][subkey] = batch[key][subkey].to(self.device)

            print("Running model inference...")

            # CRITICAL: Override num_input_views in config to match actual upload counts!
            # Otherwise ProcessData will use default from config and split incorrectly
            num_input_actual = len(input_images)
            num_target_actual = len(target_images)
            original_num_input = self.config.training.num_input_views
            original_target_has_input = self.config.training.get("target_has_input", True)

            print(f"Using nested batch format with ARBITRARY view counts:")
            print(f"  - {num_input_actual} input views (K)")
            print(f"  - {num_target_actual} target views (T)")
            print(f"  - Total: {num_input_actual + num_target_actual} views")
            print(
                f"  - Target indices will be: [{num_input_actual}, ..., {num_input_actual + num_target_actual - 1}]"
            )
            print("ProcessData will detect pre-separated input/target and use sequential ordering")

            # Update motion mask threshold dynamically
            threshold_val = float(motion_mask_threshold)
            print(f"Setting motion_mask_threshold to {threshold_val}")
            if hasattr(self.config, "training"):
                self.config.training.motion_mask_threshold = threshold_val
            if hasattr(self.config, "model"):
                self.config.model.motion_mask_threshold = threshold_val
            if hasattr(self.model, "config"):
                if hasattr(self.model.config, "training"):
                    self.model.config.training.motion_mask_threshold = threshold_val
                if hasattr(self.model.config, "model"):
                    self.model.config.model.motion_mask_threshold = threshold_val
            if hasattr(self.model, "motion_mask_threshold"):
                self.model.motion_mask_threshold = threshold_val
            if hasattr(self.model, "loss_computer") and hasattr(
                self.model.loss_computer, "motion_mask_threshold"
            ):
                self.model.loss_computer.motion_mask_threshold = threshold_val

            # No config override needed - nested format handles arbitrary counts!
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
            ):
                result = self.model(batch, create_visual=True, render_video=render_video, iter=0)

            print("Processing results...")
            rendered_targets = result.render[0].float().cpu()
            gt_targets = result.target.image[0].float().cpu()
            input_imgs = result.input.image[0].float().cpu()

            print(f"Rendered shape: {rendered_targets.shape}")
            print(f"GT shape: {gt_targets.shape}")

            # DEBUG: Check if there's index information
            if hasattr(result, "target") and hasattr(result.target, "index"):
                target_indices = result.target.index[0, :, 0].cpu().numpy()
                print(f"Target view indices from dataset: {target_indices}")

            # DEBUG: Compute MSE between all pairs to detect mismatch
            print("\nDEBUG: Pairwise MSE matrix (rendered vs GT):")
            mse_matrix = []
            for i in range(rendered_targets.shape[0]):
                row = []
                for j in range(gt_targets.shape[0]):
                    mse = ((rendered_targets[i] - gt_targets[j]) ** 2).mean().item()
                    row.append(f"{mse:.6f}")
                mse_matrix.append(row)
                print(f"  Rendered[{i}] vs GT: {row}")

            # Find best matching for each rendered view
            print("\nBest GT match for each rendered view:")
            for i in range(rendered_targets.shape[0]):
                mse_scores = [
                    (j, ((rendered_targets[i] - gt_targets[j]) ** 2).mean().item())
                    for j in range(gt_targets.shape[0])
                ]
                best_j, best_mse = min(mse_scores, key=lambda x: x[1])
                psnr = -10 * torch.log10(torch.tensor(best_mse + 1e-10)).item()
                print(
                    f"  Rendered[{i}] best matches GT[{best_j}] (MSE={best_mse:.6f}, PSNR={psnr:.2f} dB)"
                )

            # Compute PSNR for each target view
            psnrs = self.compute_psnr(rendered_targets, gt_targets)
            avg_psnr = psnrs.mean().item()

            rendered_grid = self.create_image_grid(rendered_targets)
            gt_grid = self.create_image_grid(gt_targets)

            # Video rendering (if enabled).
            # The model writes the interpolated trajectory to result.video_rendering
            # (populated whenever create_visual=True and v_target > 0).
            video_path = None
            if render_video and hasattr(result, "video_rendering") and result.video_rendering is not None:
                video_frames = result.video_rendering[0].float().cpu()
                video_path = self.create_video(video_frames)

            # Motion mask visualizations
            mask_overlay_img = None
            binary_mask_img = None
            input_mask_overlay_img = None
            dropout_overlay_img = None

            pred_masks_input, pred_masks_target = self.extract_predicted_masks(
                result, batch_idx=0, v_input=num_input_actual, v_target=num_target_actual
            )

            print("\n=== MOTION MASK DEBUG ===")
            print(f"Has predicted_target_masks: {hasattr(result, 'predicted_target_masks')}")
            if hasattr(result, "predicted_target_masks"):
                print(f"predicted_target_masks is None: {result.predicted_target_masks is None}")
                if result.predicted_target_masks is not None:
                    print(f"predicted_target_masks shape: {result.predicted_target_masks.shape}")
            print(f"Has predicted_input_masks: {hasattr(result, 'predicted_input_masks')}")
            if hasattr(result, "predicted_input_masks"):
                print(f"predicted_input_masks is None: {result.predicted_input_masks is None}")
                if result.predicted_input_masks is not None:
                    print(f"predicted_input_masks shape: {result.predicted_input_masks.shape}")
            print(f"pred_masks_target is None: {pred_masks_target is None}")
            print(f"pred_masks_input is None: {pred_masks_input is None}")
            print("========================\n")

            if pred_masks_target is not None:
                print("✓ Creating target motion mask overlays...")
                mask_probs_target = torch.sigmoid(pred_masks_target)
                mask_overlay_img = self.create_overlay_from_mask(
                    gt_targets, mask_probs_target, alpha=0.6, mode="bilinear"
                )
                binary_mask_img = self.create_binary_mask_grid(
                    mask_probs_target, threshold_val, (gt_targets.shape[-2], gt_targets.shape[-1])
                )
                print(f"  mask_overlay_img: {type(mask_overlay_img)}")
                print(f"  binary_mask_img: {type(binary_mask_img)}")
            else:
                print("⚠️  NO TARGET MASKS - Model did not generate predicted_target_masks!")

            if pred_masks_input is not None:
                print("✓ Creating input motion mask overlay...")
                mask_probs_input = torch.sigmoid(pred_masks_input)
                input_mask_overlay_img = self.create_overlay_from_mask(
                    input_imgs, mask_probs_input, alpha=0.6, mode="bilinear"
                )
                print(f"  input_mask_overlay_img: {type(input_mask_overlay_img)}")
            else:
                print("⚠️  NO INPUT MASKS - Model did not generate predicted_input_masks!")

            input_patch_mask = None
            if hasattr(result, "input_patch_mask") and result.input_patch_mask is not None:
                print("✓ Extracting input patch mask (dropout)...")
                patch_mask = result.input_patch_mask
                if patch_mask.ndim == 5:
                    input_patch_mask = patch_mask[0].detach().cpu().float()
                elif patch_mask.ndim == 4 and num_input_actual > 0:
                    input_patch_mask = patch_mask[:num_input_actual].detach().cpu().float()
                print(
                    f"  input_patch_mask shape: {input_patch_mask.shape if input_patch_mask is not None else 'None'}"
                )
            else:
                print("⚠️  NO INPUT PATCH MASK - No MAE dropout mask found!")

            if input_patch_mask is not None:
                print("✓ Creating dropout overlay...")
                dropout_overlay_img = self.create_overlay_from_mask(
                    input_imgs, input_patch_mask, alpha=0.6, mode="nearest"
                )
                print(f"  dropout_overlay_img: {type(dropout_overlay_img)}")
            else:
                print("⚠️  NO DROPOUT OVERLAY - input_patch_mask is None!")

            status = f"✓ Rendered {len(target_images)} target views\n"
            status += f"✓ Input: {len(input_images)} images, Target: {len(target_images)} images\n"
            status += f"✓ motion_mask_threshold = {threshold_val:.3f}\n"
            status += "\n📊 PSNR Metrics:\n"
            status += f"  Average PSNR: {avg_psnr:.2f} dB\n"
            for i, psnr in enumerate(psnrs):
                status += f"  Target {i+1}: {psnr:.2f} dB\n"
            if video_path:
                status += f"\n✓ Generated interpolated video"

            return (
                rendered_grid,
                gt_grid,
                mask_overlay_img,
                binary_mask_img,
                input_mask_overlay_img,
                dropout_overlay_img,
                video_path,
                status,
            )

        except Exception as e:
            import traceback

            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # Must match the 8-tuple shape of the success path:
            # rendered, gt, mask_overlay, binary_mask, input_mask, dropout, video, status
            return None, None, None, None, None, None, None, error_msg

    def compute_psnr(self, rendered, gt):
        """Compute PSNR between rendered and GT images."""
        mse = torch.mean((rendered - gt) ** 2, dim=[1, 2, 3])  # [v]
        mse = torch.clamp(mse, min=1e-10)
        psnr = -10.0 * torch.log10(mse)  # [v]
        return psnr

    def create_image_grid(self, images):
        """Create horizontal grid of images."""
        grid = rearrange(images, "v c h w -> h (v w) c")
        grid = (grid.numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(grid)

    def create_video(self, frames):
        """Create video from frames."""
        frames_np = frames.permute(0, 2, 3, 1).numpy()
        frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_path = temp_file.name
        temp_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30, (256, 256))

        for frame in frames_np:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

        writer.release()
        return video_path


def create_demo(checkpoint_path, config_path):
    demo_model = WildRayZerDemo(checkpoint_path, config_path)

    with gr.Blocks(title="WildRayZer Demo") as demo:
        gr.Markdown(
            """
        # 🎬 WildRayZer Demo
        
        Upload **arbitrary numbers** of input and target images to render novel views.
        - **Input images**: Context views for the model (1+ images, any count works!)
        - **Target images**: Novel views to render (1+ images, any count works!)
        
        **Examples:**
        - 2 inputs + 6 targets ✓
        - 3 inputs + 5 targets ✓
        - 4 inputs + 10 targets ✓
        - Any combination! ✓
        
        **Model Architecture:**
        - Encoder: 12 layers
        - Encoder Geom: 8 layers  
        - Decoder: 12 layers
        """
        )

        # Top row: File upload with list display
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📥 Input Images (any count ≥1)")
                input_images = gr.File(
                    label="Upload Input Images", file_count="multiple", file_types=["image"]
                )
                input_file_list = gr.Textbox(
                    label="Input Files",
                    lines=5,
                    interactive=False,
                    placeholder="No files uploaded yet...",
                )

            with gr.Column():
                gr.Markdown("### 🎯 Target Images (any count ≥1)")
                target_images = gr.File(
                    label="Upload Target Images", file_count="multiple", file_types=["image"]
                )
                target_file_list = gr.Textbox(
                    label="Target Files",
                    lines=5,
                    interactive=False,
                    placeholder="No files uploaded yet...",
                )

        # Control row
        with gr.Row():
            render_video_checkbox = gr.Checkbox(
                label="Generate Interpolated Video (slower)", value=False
            )
            motion_mask_slider = gr.Slider(
                label="Motion Mask Threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.1,
                step=0.01,
            )
            render_btn = gr.Button("🚀 Render", variant="primary", size="lg")

        # Middle: Rendered and GT target views side-by-side
        gr.Markdown("### 🎨 Rendered Target Views | ✅ Ground Truth Target Views")
        with gr.Row():
            with gr.Column():
                rendered_output = gr.Image(label="Rendered Target Views")
            with gr.Column():
                gt_output = gr.Image(label="Ground Truth Target Views")

        gr.Markdown("### 🎯 Motion Masks & Dropped Tokens")
        with gr.Row():
            mask_overlay_output = gr.Image(label="Target Motion Mask Overlay", interactive=False)
            binary_mask_output = gr.Image(
                label="Target Binary Mask (> threshold)", interactive=False
            )
            input_mask_output = gr.Image(label="Input Motion Mask Overlay", interactive=False)
            dropout_output = gr.Image(label="Input Dropped Tokens", interactive=False)

        # Bottom: Video and status side-by-side
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎥 Interpolated Video")
                video_output = gr.Video(label="Interpolated Video", height=400)

            with gr.Column():
                gr.Markdown("### 📊 Status & Metrics")
                status_text = gr.Textbox(label="Status", lines=15, interactive=False)

        # Update file lists when files are uploaded
        def update_input_list(files):
            if not files:
                return "No files uploaded yet..."
            filenames = [f.name.split("/")[-1] for f in files]
            file_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(filenames))
            return f"{len(filenames)} files:\n{file_list}"

        def update_target_list(files):
            if not files:
                return "No files uploaded yet..."
            filenames = [f.name.split("/")[-1] for f in files]
            file_list = "\n".join(f"  {i+1}. {name}" for i, name in enumerate(filenames))
            return f"{len(filenames)} files:\n{file_list}"

        input_images.change(fn=update_input_list, inputs=[input_images], outputs=[input_file_list])

        target_images.change(
            fn=update_target_list, inputs=[target_images], outputs=[target_file_list]
        )

        render_btn.click(
            fn=lambda inp, tgt, vid, thr: demo_model.render(
                inp, tgt, render_video=vid, motion_mask_threshold=thr
            ),
            inputs=[input_images, target_images, render_video_checkbox, motion_mask_slider],
            outputs=[
                rendered_output,
                gt_output,
                mask_overlay_output,
                binary_mask_output,
                input_mask_output,
                dropout_output,
                video_output,
                status_text,
            ],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.checkpoint, args.config)
    demo.launch(server_port=args.port, share=args.share)

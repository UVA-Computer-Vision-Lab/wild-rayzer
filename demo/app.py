#!/usr/bin/env python3
"""
Gradio demo for the released WildRayZer checkpoint.

Two input modes:
  • Pick example — one click on a scene thumbnail loads the pre-defined
    2-context / 6-target split from D-RE10K.
  • Upload your own — drop 2 input images and 6 target images.

Paths default to the canonical release layout. Override via CLI flags or env
vars (CLI flags win):

    WILDRAYZER_CKPT=./checkpoints/wildrayzer_2view.pt \
    WILDRAYZER_CFG=./configs/wildrayzer_inference.yaml \
    WILDRAYZER_TEST_ROOT=./data/dynamic_re10k/test \
    python app.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

# ZeroGPU: import before torch CUDA touches anything. When the `spaces` package
# isn't installed (local dev), fall back to a no-op decorator.
try:
    import spaces
    _GPU_DECORATOR = spaces.GPU(duration=120)
except ImportError:
    def _GPU_DECORATOR(fn):
        return fn

from app_demo import WildRayZerDemo


def _resolve_default_ckpt() -> str:
    """Fetch the checkpoint from HF Hub on first launch (cached under HF_HOME)."""
    override = os.environ.get("WILDRAYZER_CKPT")
    if override:
        return override
    local = PROJECT_ROOT / "checkpoints" / "wildrayzer_2view.pt"
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download
    return hf_hub_download("uva-cv-lab/wildrayzer-2view", "wildrayzer_2view.pt")


DEFAULT_CKPT = _resolve_default_ckpt()
DEFAULT_CFG = str(PROJECT_ROOT / "configs/wildrayzer_inference.yaml")
DEFAULT_TEST_ROOT = str(PROJECT_ROOT / "data/dynamic_re10k/test")

# This release ships a K=2 checkpoint only.
NUM_INPUT = 2
NUM_TARGET = 6


HEADER_HTML = """
<div style="text-align:center;">
  <h1 style="margin:0 0 0.4em 0;">WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments</h1>
  <p style="margin:0.3em 0;">
    <a href="https://xuweiyichen.github.io/">Xuweiyi Chen</a> ·
    <a href="https://smirkkkk.github.io/">Wentao Zhou</a> ·
    <a href="https://sites.google.com/site/zezhoucheng/">Zezhou Cheng</a>
  </p>
  <p style="margin:0.3em 0;">University of Virginia · <b>CVPR 2026 Highlight</b></p>
  <p style="margin:0.3em 0;">
    <a href="https://arxiv.org/abs/2601.10716">arXiv</a> ·
    <a href="https://wild-rayzer.cs.virginia.edu/">Project Page</a> ·
    <a href="https://huggingface.co/datasets/uva-cv-lab/Dynamic-RE10K">Dynamic-RE10K</a>
  </p>
</div>
"""


class _FakeFile:
    """Mimics an uploaded-file object (`.name` attribute) for WildRayZerDemo.render."""

    __slots__ = ("name",)

    def __init__(self, path: str):
        self.name = path


def _load_view_idx(test_root: str) -> dict:
    json_path = os.path.join(test_root, f"dre10k_final_context_{NUM_INPUT}_view_idx.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


def _resolve_paths(
    view_idx: dict, images_dir: str, scene: str
) -> Tuple[List[str], List[str]]:
    idx = view_idx.get(scene)
    if idx is None:
        return [], []
    ctx = [os.path.join(images_dir, scene, f"{i:05d}.png") for i in idx["context"]]
    tgt = [os.path.join(images_dir, scene, f"{i:05d}.png") for i in idx["target"]]
    return ctx, tgt


def _call_render(demo_model, ctx_paths, tgt_paths, thr, render_video, label=""):
    """Shared rendering core; returns the 8-tuple Gradio expects."""
    if len(ctx_paths) != NUM_INPUT:
        return (None, None, None, None, None, None, None,
                f"Need {NUM_INPUT} context images, got {len(ctx_paths)}.")
    if len(tgt_paths) != NUM_TARGET:
        return (None, None, None, None, None, None, None,
                f"Need {NUM_TARGET} target images, got {len(tgt_paths)}.")
    inp = [_FakeFile(p) for p in ctx_paths]
    tgt = [_FakeFile(p) for p in tgt_paths]
    t0 = time.time()
    outs = demo_model.render(inp, tgt, render_video=render_video, motion_mask_threshold=float(thr))
    dt = time.time() - t0
    (*imgs, video, status) = outs
    status = f"⏱  {dt:.2f} s — {label} (threshold={thr:.2f})\n{status}"
    return (*imgs, video, status)


def build_ui(checkpoint: str, config: str, test_root: str) -> Tuple[gr.Blocks, List[str]]:
    images_dir = os.path.join(test_root, "images")

    view_idx = _load_view_idx(test_root)
    scenes = sorted(view_idx.keys())
    if not scenes:
        raise RuntimeError(
            f"No scenes found under {test_root}. "
            "Download D-RE10K from https://huggingface.co/datasets/uva-cv-lab/Dynamic-RE10K "
            "or set WILDRAYZER_TEST_ROOT to an existing directory."
        )
    if not os.path.exists(checkpoint):
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint}. "
            "Set WILDRAYZER_CKPT or pass --ckpt."
        )

    # Example thumbnails: use the first context frame of each scene.
    example_thumbs: List[Tuple[str, str]] = []
    for scene in scenes:
        ctx, _ = _resolve_paths(view_idx, images_dir, scene)
        if ctx and os.path.exists(ctx[0]):
            example_thumbs.append((ctx[0], scene))

    print(f"Loading model from:\n  ckpt={checkpoint}\n  cfg={config}")
    demo_model = WildRayZerDemo(checkpoint, config)

    @_GPU_DECORATOR
    def run_example(scene: str, thr: float, render_video: bool):
        ctx, tgt = _resolve_paths(view_idx, images_dir, scene) if scene else ([], [])
        if not ctx or not tgt:
            return (ctx, tgt, None, None, None, None, None, None, None,
                    f"Scene {scene!r} not found.")
        outs = _call_render(demo_model, ctx, tgt, thr, render_video, label=f"scene: {scene}")
        return (ctx, tgt, *outs)

    @_GPU_DECORATOR
    def run_upload(input_files, target_files, thr, render_video):
        if not input_files or not target_files:
            return ([], [], None, None, None, None, None, None, None,
                    "Upload both input and target images.")
        ctx_paths = [f if isinstance(f, str) else f.name for f in input_files]
        tgt_paths = [f if isinstance(f, str) else f.name for f in target_files]
        outs = _call_render(demo_model, ctx_paths, tgt_paths, thr, render_video, label="upload")
        return (ctx_paths, tgt_paths, *outs)

    with gr.Blocks(title="WildRayZer") as demo:
        gr.HTML(HEADER_HTML)

        # ===== Controls =====
        with gr.Row():
            motion_mask_thr = gr.Slider(
                label="Motion mask threshold",
                minimum=0.0, maximum=1.0, value=0.1, step=0.01,
            )
            render_video_ck = gr.Checkbox(label="Render interpolated video", value=False)

        # ===== Input source tabs =====
        with gr.Tabs() as input_tabs:
            with gr.Tab("📁 Pick example", id="examples"):
                gr.Markdown(
                    f"Click a scene below (pre-defined **{NUM_INPUT} context + "
                    f"{NUM_TARGET} target** views per scene)."
                )
                # height=None lets the gallery grow to fit every thumbnail; with
                # columns=6 and ~25 scenes this renders ~5 rows inline (no hidden
                # scrollbar inside the widget — users were missing rows 2-5).
                example_gallery = gr.Gallery(
                    value=example_thumbs,
                    label=f"D-RE10K test scenes ({len(example_thumbs)})",
                    columns=6,
                    rows=(len(example_thumbs) + 5) // 6,  # ceil(N / columns)
                    height=None,
                    object_fit="cover",
                    allow_preview=False,
                    show_label=True,
                )
                selected_scene = gr.Textbox(
                    label="Selected scene", interactive=False,
                )
                run_example_btn = gr.Button("Run on selected scene", variant="primary")

            with gr.Tab("⬆ Upload your own", id="upload"):
                gr.HTML(
                    f"""
<div style="padding:14px 18px; margin: 6px 0 12px 0;
            border:2px solid #ff7f00; border-radius:10px;
            background: rgba(255,127,0,0.08);
            font-size: 1.05em; text-align:center;">
  <b>Required:</b>
  exactly
  <span style="display:inline-block; padding:2px 10px; margin:0 4px;
               border-radius:999px; background:#ff7f00; color:white; font-weight:700;">
    {NUM_INPUT} context images
  </span>
  and
  <span style="display:inline-block; padding:2px 10px; margin:0 4px;
               border-radius:999px; background:#ff7f00; color:white; font-weight:700;">
    {NUM_TARGET} target images
  </span>.
  <br/><span style="font-size:0.9em; opacity:0.85;">
    Anything else will be rejected. Drop images in chronological order.
  </span>
</div>
"""
                )
                gr.Markdown(
                    """
**What works best:**
- Frames from a *single short clip* (a few seconds of footage).
- Camera moves around a *mostly-static indoor scene* (matches training distribution).
- Small-to-moderate motion between frames (big jumps will fail pose estimation).

**Automatic preprocessing:** non-square images are *center-cropped* to a square,
then resized to 256×256. EXIF orientation is respected (iPhone photos are
auto-rotated). RGBA / grayscale inputs are converted to RGB.
"""
                )
                with gr.Row():
                    input_upload = gr.File(
                        label=f"📥 Context images — drop exactly {NUM_INPUT}",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    target_upload = gr.File(
                        label=f"🎯 Target images — drop exactly {NUM_TARGET}",
                        file_count="multiple",
                        file_types=["image"],
                    )
                run_upload_btn = gr.Button("Run on uploaded images", variant="primary")

        # ===== Selected views preview =====
        with gr.Row():
            ctx_gallery = gr.Gallery(
                label="Context (input) views",
                columns=4, height=180, object_fit="contain",
            )
            tgt_gallery = gr.Gallery(
                label="Target (GT) views",
                columns=6, height=180, object_fit="contain",
            )

        # ===== Outputs =====
        with gr.Tabs():
            with gr.Tab("Rendering"):
                rendered_out = gr.Image(label="Rendered target views", height=360)
                gt_out = gr.Image(label="Ground truth", height=360)
                with gr.Row():
                    with gr.Column(scale=1):
                        video_out = gr.Video(label="Interpolated video", height=320)
                    with gr.Column(scale=2):
                        pass  # spacer so the 256×256 video doesn't stretch full width
            with gr.Tab("Motion masks"):
                with gr.Row():
                    mask_overlay_out = gr.Image(label="Target motion mask (soft)", height=320)
                    binary_mask_out = gr.Image(label="Target binary mask", height=320)
                with gr.Row():
                    input_mask_out = gr.Image(label="Input motion mask overlay", height=320)
                    dropout_out = gr.Image(label="Dropped tokens (MAE)", height=320)
            with gr.Tab("Diagnostics"):
                status = gr.Textbox(label="Per-view PSNR + log", lines=14, interactive=False)

        gr.Markdown(
            "<div align='center'><sub>© 2026 University of Virginia</sub></div>"
        )

        output_components = [
            rendered_out, gt_out,
            mask_overlay_out, binary_mask_out,
            input_mask_out, dropout_out,
            video_out, status,
        ]

        # ----- Example gallery → select scene -----
        def on_gallery_select(evt: gr.SelectData):
            if evt is None or evt.index is None:
                return gr.update(), [], []
            idx = evt.index if isinstance(evt.index, int) else evt.index[0]
            if idx >= len(example_thumbs):
                return gr.update(), [], []
            scene = example_thumbs[idx][1]
            ctx, tgt = _resolve_paths(view_idx, images_dir, scene)
            return scene, ctx, tgt

        example_gallery.select(
            fn=on_gallery_select,
            inputs=None,
            outputs=[selected_scene, ctx_gallery, tgt_gallery],
        )

        run_example_btn.click(
            fn=run_example,
            inputs=[selected_scene, motion_mask_thr, render_video_ck],
            outputs=[ctx_gallery, tgt_gallery, *output_components],
        )
        run_upload_btn.click(
            fn=run_upload,
            inputs=[input_upload, target_upload, motion_mask_thr, render_video_ck],
            outputs=[ctx_gallery, tgt_gallery, *output_components],
        )

    return demo, [test_root, images_dir]


# ---------------------------------------------------------------------------
# Module-level app construction
# ---------------------------------------------------------------------------
# HF Spaces invokes the file in hot-reload mode ("gradio app.py"), which does
# NOT execute `if __name__ == '__main__':` — it just imports the module and
# looks for a top-level `demo` Blocks instance. So we build `demo` at import
# time; the CLI path (`python app.py`) still works via main().

_CKPT_PATH = os.environ.get("WILDRAYZER_CKPT", DEFAULT_CKPT)
_CFG_PATH = os.environ.get("WILDRAYZER_CFG", DEFAULT_CFG)
_TEST_ROOT = os.environ.get("WILDRAYZER_TEST_ROOT", DEFAULT_TEST_ROOT)

demo, _ALLOWED_PATHS = build_ui(_CKPT_PATH, _CFG_PATH, _TEST_ROOT)
# Teach Gradio which on-disk paths it's allowed to serve (the D-RE10K images
# live outside the CWD on some deployments). Safe to set on Blocks directly —
# it's honoured whether launched via .launch() or the hot-reload CLI.
try:
    demo.allowed_paths = list(_ALLOWED_PATHS)
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt",      default=_CKPT_PATH)
    parser.add_argument("--config",    default=_CFG_PATH)
    parser.add_argument("--test-root", default=_TEST_ROOT)
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )
    parser.add_argument("--no-share", action="store_true")
    args = parser.parse_args()

    on_space = bool(os.environ.get("SPACE_ID"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False if on_space else (not args.no_share),
        allowed_paths=_ALLOWED_PATHS,
    )


if __name__ == "__main__":
    main()

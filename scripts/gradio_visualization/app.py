#!/usr/bin/env python3
"""
Gradio demo for the released WildRayZer 2-input-view checkpoint, backed by the
D-RE10K test set as built-in examples.

Paths default to the canonical release layout. Override via CLI flags or env
vars (CLI flags win):

    WILDRAYZER_CKPT=./checkpoints/wildrayzer_2view.pt \
    WILDRAYZER_CFG=./configs/wildrayzer_inference.yaml \
    WILDRAYZER_TEST_ROOT=./data/dynamic_re10k/test \
    python scripts/gradio_visualization/app.py

To deploy as a Hugging Face Space, copy this file to the Space root as
`app.py` and see scripts/gradio_visualization/README.md for the full layout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from scripts.gradio_visualization.app_demo import WildRayZerDemo


DEFAULT_CKPT = str(PROJECT_ROOT / "checkpoints/wildrayzer_2view.pt")
DEFAULT_CFG = str(PROJECT_ROOT / "configs/wildrayzer_inference.yaml")
DEFAULT_TEST_ROOT = str(PROJECT_ROOT / "data/dynamic_re10k/test")

# This release ships a K=2 checkpoint only.
CONTEXT_K = 2


HEADER_MD = """
<div align="center">

# WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments

[Xuweiyi Chen](https://xuweiyichen.github.io/) · [Wentao Zhou](https://smirkkkk.github.io/) · [Zezhou Cheng](https://sites.google.com/site/zezhoucheng/)

University of Virginia

[arXiv](https://arxiv.org/abs/2601.10716)  ·  [Project Page](https://wild-rayzer.cs.virginia.edu/)  ·  [Dynamic-RE10K](https://huggingface.co/datasets/uva-cv-lab/Dynamic-RE10K)

</div>
"""


class _FakeFile:
    """Mimics an uploaded-file object (`.name` attribute) for WildRayZerDemo.render."""

    __slots__ = ("name",)

    def __init__(self, path: str):
        self.name = path


def _load_view_idx(test_root: str, context_count: int) -> dict:
    json_path = os.path.join(test_root, f"dre10k_final_context_{context_count}_view_idx.json")
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


def build_ui(checkpoint: str, config: str, test_root: str) -> Tuple[gr.Blocks, List[str]]:
    images_dir = os.path.join(test_root, "images")

    view_idx = _load_view_idx(test_root, CONTEXT_K)
    if not view_idx:
        raise RuntimeError(
            f"Split file dre10k_final_context_{CONTEXT_K}_view_idx.json not found under {test_root}. "
            "Download the D-RE10K test set from "
            "https://huggingface.co/datasets/uva-cv-lab/Dynamic-RE10K or point "
            "--test-root (or WILDRAYZER_TEST_ROOT) at an existing directory."
        )
    if not os.path.exists(checkpoint):
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint}. "
            "Train WildRayZer first (see README §3) or pass --ckpt."
        )

    print(f"Loading model from:\n  ckpt={checkpoint}\n  cfg={config}")
    demo_model = WildRayZerDemo(checkpoint, config)

    scenes = sorted(view_idx.keys())
    default_scene = scenes[0] if scenes else None

    def render_scene(scene: str, thr: float, render_video: bool):
        if not scene:
            return (None, None, None, None, None, None, None, "Pick a scene first.")
        ctx_paths, tgt_paths = _resolve_paths(view_idx, images_dir, scene)
        if not ctx_paths or not tgt_paths:
            return (
                None, None, None, None, None, None, None,
                f"Scene {scene!r} has no resolvable frames.",
            )
        inp = [_FakeFile(p) for p in ctx_paths]
        tgt = [_FakeFile(p) for p in tgt_paths]
        t0 = time.time()
        outputs = demo_model.render(
            inp, tgt, render_video=render_video, motion_mask_threshold=float(thr)
        )
        dt = time.time() - t0

        (*images, video, status) = outputs
        status = f"⏱  {dt:.2f} s — scene: {scene}  (threshold={thr:.2f})\n{status}"
        return (*images, video, status)

    with gr.Blocks(title="WildRayZer") as demo:
        gr.Markdown(HEADER_MD)

        gr.Markdown(
            f"Select a scene from the **D-RE10K** test set; context and target frames "
            f"are pre-defined per the official split ({len(scenes)} scenes available)."
        )

        with gr.Row():
            with gr.Column(scale=2):
                scene_dd = gr.Dropdown(
                    choices=scenes,
                    value=default_scene,
                    label="Scene",
                    allow_custom_value=False,
                )
            with gr.Column(scale=1):
                motion_mask_thr = gr.Slider(
                    label="Motion mask threshold",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.01,
                )
            with gr.Column(scale=1, min_width=140):
                render_video_ck = gr.Checkbox(label="Render video", value=False)
                auto_run_ck = gr.Checkbox(label="Auto-render", value=False)

        with gr.Row():
            random_btn = gr.Button("Random scene")
            run_btn = gr.Button("Run", variant="primary")

        with gr.Row():
            ctx_gallery = gr.Gallery(
                label="Context (input) views",
                columns=4,
                height=180,
                object_fit="contain",
            )
            tgt_gallery = gr.Gallery(
                label="Target (GT) views",
                columns=6,
                height=180,
                object_fit="contain",
            )

        with gr.Tabs():
            with gr.Tab("Rendering"):
                with gr.Row():
                    rendered_out = gr.Image(label="Rendered target views", height=220)
                    gt_out = gr.Image(label="Ground truth", height=220)
                video_out = gr.Video(label="Interpolated video", visible=True)
            with gr.Tab("Motion masks"):
                with gr.Row():
                    mask_overlay_out = gr.Image(label="Target motion mask (soft)", height=220)
                    binary_mask_out = gr.Image(label="Target binary mask", height=220)
                with gr.Row():
                    input_mask_out = gr.Image(label="Input motion mask overlay", height=220)
                    dropout_out = gr.Image(label="Dropped tokens (MAE)", height=220)
            with gr.Tab("Diagnostics"):
                status = gr.Textbox(label="Per-view PSNR + log", lines=14, interactive=False)

        gr.Markdown(
            "<div align='center'><sub>© 2026 University of Virginia</sub></div>"
        )

        output_components = [
            rendered_out,
            gt_out,
            mask_overlay_out,
            binary_mask_out,
            input_mask_out,
            dropout_out,
            video_out,
            status,
        ]

        def on_scene_change(scene, auto_run, thr, rv):
            ctx, tgt = _resolve_paths(view_idx, images_dir, scene) if scene else ([], [])
            if auto_run and scene:
                outs = render_scene(scene, thr, rv)
                return ctx, tgt, *outs
            return (ctx, tgt, *[gr.update() for _ in output_components])

        def pick_random():
            return random.choice(scenes) if scenes else None

        scene_dd.change(
            fn=on_scene_change,
            inputs=[scene_dd, auto_run_ck, motion_mask_thr, render_video_ck],
            outputs=[ctx_gallery, tgt_gallery, *output_components],
        )
        random_btn.click(fn=pick_random, inputs=None, outputs=[scene_dd])
        run_btn.click(
            fn=render_scene,
            inputs=[scene_dd, motion_mask_thr, render_video_ck],
            outputs=output_components,
        )

        if default_scene is not None:
            demo.load(
                fn=lambda s=default_scene: _resolve_paths(view_idx, images_dir, s),
                inputs=None,
                outputs=[ctx_gallery, tgt_gallery],
            )

    return demo, [test_root, images_dir]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("WILDRAYZER_CKPT", DEFAULT_CKPT),
        help="Path to trained WildRayZer checkpoint.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("WILDRAYZER_CFG", DEFAULT_CFG),
        help="Path to inference config YAML.",
    )
    parser.add_argument(
        "--test-root",
        default=os.environ.get("WILDRAYZER_TEST_ROOT", DEFAULT_TEST_ROOT),
        help="Path to the D-RE10K test root (contains images/ and split JSON).",
    )
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--no-share", action="store_true")
    args = parser.parse_args()

    demo, allowed_paths = build_ui(args.ckpt, args.config, args.test_root)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=not args.no_share,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()

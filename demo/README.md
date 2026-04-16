---
title: WildRayZer
emoji: 🎥
colorFrom: indigo
colorTo: slate
sdk: gradio
app_file: app.py
pinned: false
license: other
---

# WildRayZer — Gradio Space

Demo for **WildRayZer: Self-supervised Large View Synthesis in Dynamic
Environments** (CVPR 2026, Highlight).

On first launch the Space downloads the 2-view checkpoint from
[`uva-cv-lab/wildrayzer-2view`](https://huggingface.co/uva-cv-lab/wildrayzer-2view)
via `huggingface_hub.hf_hub_download`. DINOv3 ViT-7B is also fetched from the
hub on first model construction (~30 GB). Both downloads are cached under
`HF_HOME` — set it to the persistent storage mount in your Space settings.

## Hardware

Requires a CUDA GPU with ≥ 40 GB VRAM (DINOv3 ViT-7B + scene encoder + renderer).
CPU Spaces will not work.

## Files

- `app.py` — Gradio UI (scene picker, threshold slider, video toggle).
- `app_demo.py` — `WildRayZerDemo` class: model loading, batching, rendering.
- `configs/wildrayzer_inference.yaml` — inference config.
- `model/` — `rayzer_official_v3.py` + `transformer.py` + `loss.py`.
- `utils/` — supporting utilities (camera, metrics, data, positional embeddings).
- `data/dynamic_re10k/test/` — a subset of the D-RE10K test split (6 scenes +
  the filtered view-index JSON). Full dataset at
  <https://huggingface.co/datasets/uva-cv-lab/Dynamic-RE10K>.
- `setup.py` — config loader / DDP helpers imported by `inference.py` (kept for
  cross-compatibility).

## Env vars

- `WILDRAYZER_CKPT` — local path to a checkpoint, overriding the HF download.
- `WILDRAYZER_CFG` — local path to a config YAML.
- `WILDRAYZER_TEST_ROOT` — path to a D-RE10K-style test directory.
- `HF_HOME` — cache dir for checkpoint + DINOv3; point at `/data` on paid Spaces.

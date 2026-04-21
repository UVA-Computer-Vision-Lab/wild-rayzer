---
license: cc-by-nc-4.0
tags:
  - novel-view-synthesis
  - nvs
  - dynamic-scene
  - self-supervised
  - 3d
  - computer-vision
  - cvpr
  - cvpr2026
datasets:
  - uva-cv-lab/Dynamic-RE10K
library_name: pytorch
pipeline_tag: image-to-image
---

# WildRayZer — 2-input-view checkpoint

This repository hosts the **2-input / 6-target-view** checkpoint of
**WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments**
(CVPR 2026, Highlight).

<p align="center">
  <a href="https://arxiv.org/abs/2601.10716">Paper</a> ·
  <a href="https://wild-rayzer.cs.virginia.edu/">Project page</a> ·
  <a href="https://huggingface.co/datasets/uva-cv-lab/Dynamic-RE10K">Dataset</a> ·
  <a href="https://github.com/uva-cv-lab/wildrayzer">Code</a>
</p>

## Model summary

WildRayZer is a self-supervised feed-forward framework for novel view synthesis
(NVS) in **dynamic in-the-wild videos** where both the camera and scene objects
move. It extends the static NVS model [RayZer](https://hwjiang1510.github.io/RayZer/)
to dynamic environments by adding:

1. a **learned motion mask estimator** that flags dynamic regions per input
   view, trained by distilling pseudo-masks from the residual between a static
   renderer and the observed frames (DINOv3 + SSIM + co-segmentation + GrabCut);
2. a **masked 3D scene encoder** that replaces dynamic image tokens with a
   learnable noise embedding before scene aggregation (MAE-style token masking).

All supervision is **self-supervised** — no ground-truth depth, camera poses,
or motion masks are used. Given a set of unposed, uncalibrated dynamic images,
the model predicts camera parameters and motion masks and renders novel static
views in a single feed-forward pass.

## This checkpoint

| Property | Value |
|---|---|
| File | `wildrayzer_2view.pt` (3.9 GB, fp32 state_dict) |
| Input resolution | 256 × 256 |
| Input / target views | 2 input → 6 target |
| Base dataset | Dynamic-RE10K (train split) + RealEstate10K (static mix-in) |
| Backbone | RayZer (28 transformer layers) + DINOv3 ViT-7B features |
| Framework | PyTorch ≥ 2.1, xFormers, transformers |

> The K=2 configuration matches the sparse-view setting used in the paper's
> main D-RE10K and D-RE10K-iPhone benchmarks. 3- and 4-input-view variants can
> be reproduced by retraining with the same pipeline — see
> [training details](#training).

## How to use

Download the checkpoint and run the reference demo:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="uva-cv-lab/wildrayzer-2view",
    filename="wildrayzer_2view.pt",
)
# Pass ckpt_path to the WildRayZerDemo class or to inference.py
# via --config configs/wildrayzer_inference.yaml.
```

The full inference pipeline, Gradio demo, and training code live in the
[companion repo](https://github.com/uva-cv-lab/wildrayzer). A ready-to-deploy
Space layout is provided under `demo/` in that repo.

Hardware requirements: CUDA GPU with **≥ 40 GB VRAM** (the motion-mask
predictor fuses DINOv3 ViT-7B patch features with image/Plücker tokens at
inference time — this 7B backbone is a hard dependency, not optional).

## Citation

```bibtex
@inproceedings{chen2026wildrayzer,
  title     = {WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments},
  author    = {Chen, Xuweiyi and Zhou, Wentao and Cheng, Zezhou},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
}
```

## License

Released under **CC BY-NC 4.0** — free for research and non-commercial use,
attribution required. For commercial licensing, contact the authors.

## Acknowledgements

This work was supported by the MathWorks Research Gift, Adobe Research Gift, the University of Virginia
Research Computing and Data Analytics Center, the AMD AI & HPC Cluster Program,
the ACCESS program, and the NAIRR Pilot. Computation was run on the Anvil
supercomputer (NSF OAC-2005632) at Purdue and on Delta / DeltaAI (NSF
OAC-2005572).

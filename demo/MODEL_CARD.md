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

## Training

Training proceeds in three stages over ~8× H100-days on 256×256 clips:

1. **Stage 1 — RayZer pretraining.** Train the static backbone on
   RealEstate10K with 2 input / 6 target views.
2. **Stage 2 — Motion mask training.** Freeze the renderer; train the motion
   mask predictor to match DINOv3+SSIM pseudo-labels derived from the frozen
   renderer's residuals. A PSNR filter (threshold 17 dB) drops noisy samples.
3. **Stage 3 — Joint masked reconstruction + copy-paste augmentation.** Unfreeze
   the renderer; train jointly with MAE-style token masking and COCO
   copy-paste augmentation that injects synthetic transients into static clips.

See `configs/wildrayzer_stage{1,2,3}_*.yaml` in the companion repo for exact
hyperparameters.

## Evaluation

Numbers are from Table 2/3/4 of the paper (CVPR 2026).

**D-RE10K (static-region NVS, K=2 input views):**

|            | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---         |---     |---     |---      |
| NeRF On-the-go | 15.90 | 0.518 | 0.582 |
| 3DGS           | 13.49 | 0.442 | 0.605 |
| WildGaussians  | 16.12 | 0.512 | 0.624 |
| RayZer + SAV   | 19.01 | 0.628 | 0.397 |
| **WildRayZer** | **21.78** | **0.734** | **0.308** |

**D-RE10K-iPhone (full-image NVS, K=2 input views):**

|                | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---             |---     |---     |---      |
| WildGaussians  | 18.43 | 0.514 | 0.643 |
| RayZer + SAV   | 18.52 | 0.516 | 0.502 |
| **WildRayZer** | **20.89** | **0.611** | **0.364** |

**Motion-mask quality (D-RE10K, self-supervised row):**

|            | mIoU (K=2) | Recall (K=2) |
|---         |---         |---           |
| Co-segmentation | 53.9  | 85.1 |
| **WildRayZer**  | **53.9** | **85.1** |

(WildRayZer also reports 52.1 / 54.2 mIoU for K=3 / K=8 — see paper.)

## Intended use & limitations

**Intended use.**
- Research on self-supervised novel-view synthesis in dynamic, in-the-wild video.
- Transient-aware sparse-view rendering on casually captured indoor clips.
- Motion-mask prediction for dynamic regions in static-camera or handheld footage.

**Out of scope.**
- Per-frame segmentation: the predicted masks flag regions that break multi-view
  consistency, not exhaustive instance boundaries — motion-mask may miss static
  parts of moving objects (e.g. a stationary limb of a moving person).
- Heavily dynamic scenes where transient objects cover a large fraction of all
  input views: reconstruction degrades because no view sees the occluded
  background.
- Non-indoor domains: the checkpoint is trained on RealEstate10K-derived
  indoor data; outdoor generalization is partial (see DAVIS examples in the
  paper's supplementary).

**Known caveats.**
- Sensitive to the number of input views due to image-index positional
  embeddings. This checkpoint is intended for K=2 input + T=6 target views.
- Requires ~40 GB VRAM at inference (DINOv3-7B backbone).
- The checkpoint was saved before the learnable `noise_token` was registered
  in the state dict; on load you'll see a harmless *Missing keys: ['noise_token']*
  warning. It's random-initialised (truncated normal, std 0.02) each launch.

## Citation

```bibtex
@inproceedings{chen2026wildrayzer,
  title     = {WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments},
  author    = {Chen, Xuweiyi and Zhou, Wentao and Cheng, Zezhou},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  note      = {Highlight},
  year      = {2026},
}
```

## License

Released under **CC BY-NC 4.0** — free for research and non-commercial use,
attribution required. For commercial licensing, contact the authors.

## Acknowledgements

This work was supported by the Adobe Research Gift, the University of Virginia
Research Computing and Data Analytics Center, the AMD AI & HPC Cluster Program,
the ACCESS program, and the NAIRR Pilot. Computation was run on the Anvil
supercomputer (NSF OAC-2005632) at Purdue and on Delta / DeltaAI (NSF
OAC-2005572).

# WildRayZer Gradio Demo

Interactive web interface for the released WildRayZer 2-input-view checkpoint.
Predicts motion masks on input views, renders novel target views, and
(optionally) produces interpolated videos.

## Scripts

- **`app.py`** — main demo UI backed by the D-RE10K test splits
  (`data/dynamic_re10k/test/dre10k_final_context_2_view_idx.json`).
  This is the file to deploy as a Hugging Face Space.
- **`app_demo.py`** — underlying `WildRayZerDemo` class (model loading,
  batching, rendering, metric computation). Imported by `app.py`.

## Local usage

```bash
# From the repo root — defaults resolve to
#   ./checkpoints/wildrayzer_2view.pt
#   ./configs/wildrayzer_inference.yaml
#   ./data/dynamic_re10k/test
python scripts/gradio_visualization/app.py --port 7870

# Override any of the paths:
python scripts/gradio_visualization/app.py \
    --ckpt ./checkpoints/wildrayzer_2view.pt \
    --config ./configs/wildrayzer_inference.yaml \
    --test-root ./data/dynamic_re10k/test \
    --no-share
```

Environment variable equivalents: `WILDRAYZER_CKPT`, `WILDRAYZER_CFG`,
`WILDRAYZER_TEST_ROOT`.

### Arguments

- `--ckpt` — Path to the trained checkpoint.
- `--config` — Path to the inference config YAML.
- `--test-root` — D-RE10K test root (must contain `images/` and the split JSON).
- `--port` — Port to run the server on (default: `7870`).
- `--no-share` — Disable the public Gradio tunnel.

## Deploying to a Hugging Face Space

1. Create a new Gradio Space on <https://huggingface.co/spaces>. Pick a
   GPU runtime — the motion-mask predictor needs a CUDA GPU with ≥ 40 GB VRAM
   (DINOv3 ViT-7B is a dependency). **CPU Spaces will not work.**

2. Your Space repo needs this flat layout:

   ```
   app.py                              # copy from scripts/gradio_visualization/app.py
   requirements.txt
   configs/
     wildrayzer_inference.yaml
   model/
     rayzer_official_v3.py
     transformer.py
     loss.py
     __init__.py                       # add an empty one if missing
   utils/
     camera_utils.py
     data_utils.py
     masked_metrics.py
     metric_utils.py
     pe_utils.py
     pe_utils_official.py
     pose_utils.py
     training_utils.py
     __init__.py
   data/
     dynamic_re10k/test/
       dre10k_final_context_2_view_idx.json
       images/<scene>/<frame>.png      # Use git-lfs or fetch at startup
   setup.py                            # required by inference pipeline
   checkpoints/
     wildrayzer_2view.pt               # 3.9 GB — use git-lfs or hf-hub download
   ```

   Alongside `app.py` copy `app_demo.py` too, and change `app.py`'s import
   from `scripts.gradio_visualization.app_demo` to `app_demo` (flat layout).

3. Create `requirements.txt` in the Space root. Minimum:

   ```txt
   torch
   torchvision
   transformers          # DINOv3 backbone
   einops
   omegaconf
   easydict
   gradio
   numpy
   Pillow
   opencv-python
   lpips
   tqdm
   ```

4. The 3.9 GB checkpoint is too large for a normal Git push. Two options:

   - **Git LFS** — `git lfs track "*.pt"` then push; the Space will fetch it
     automatically.
   - **Download at startup** — host the checkpoint on a HF dataset repo and
     use `huggingface_hub.hf_hub_download` inside `app.py` before loading the
     model. This keeps the Space repo small and the checkpoint swappable.

5. The D-RE10K test images (~GBs) should **not** live in the Space repo.
   Either:
   - Have the Space download the subset it needs from
     `uva-cv-lab/Dynamic-RE10K` on startup, or
   - Ship only a handful of scenes as examples (copy those scene folders into
     `data/dynamic_re10k/test/images/<scene>/` and keep the full split JSON —
     the dropdown will show every scene but only the shipped ones will render).

6. DINOv3 ViT-7B is downloaded from HF on first model-construction (~30 GB).
   Set `HF_HOME=/data` (or wherever the Space's persistent storage is mounted)
   via the Space's Settings → Variables, so the download is cached across
   restarts.

## Interface

1. Pick a scene from the dropdown (or click **Random scene**).
2. Adjust the motion-mask threshold if desired.
3. Toggle **Render video** / **Auto-render** as needed.
4. Click **Run**. Outputs include rendered target views, motion-mask overlays,
   binary masks, and per-view PSNR in the Diagnostics tab.

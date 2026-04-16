# Copyright (c) 2025 Hanwen Jiang, Xuweiyi Chen. Adapted for WildRayZer from the RayZer project.

import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
from setup import init_config, init_distributed
from utils.metric_utils import (
    export_results,
    export_results_with_original_filenames,
    summarize_evaluation,
)

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=False,  # CRITICAL: Don't drop last batch in inference!
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)

dist.barrier()


# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])


if config.inference.get("model_path", None) is not None and config.inference.get(
    "if_inference", False
):
    # use the specified checkpoint if specified
    checkpoint = torch.load(config.inference.model_path, map_location="cpu", weights_only=True)
    model.module.load_state_dict(
        checkpoint["model"], strict=False
    )  # strict=False because the loss computer is different from internal version, BE CAREFUL!
elif config.inference.get("model_path", None) is None and config.inference.get(
    "if_inference", False
):
    # otherwise, use the last training checkpoint
    model.module.load_ckpt(config.training.checkpoint_dir)

if config.inference.get("view_idx_file_path", None) is not None:
    inference_sampling = config.inference.view_idx_file_path.replace(".json", "").split("_")[-1]
    inference_out_dir = os.path.join(
        config.get("inference_out_root", None),
        config.training.wandb_exp_name
        + "_"
        + config.training.view_selector.type
        + "_"
        + inference_sampling,
    )
else:
    # No view index file, use full dataset
    inference_out_dir = os.path.join(
        config.get("inference_out_root", None),
        config.training.wandb_exp_name + "_" + config.training.view_selector.type + "_full",
    )
if ddp_info.is_main_process:
    print(f"Running inference; save results to: {inference_out_dir}")
    os.makedirs(inference_out_dir, exist_ok=True)
    # avoid multiple processes downloading LPIPS at the same time
    import lpips

    # Suppress the warning by setting weights_only=True
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()

with (
    torch.no_grad(),
    torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ),
):
    # Add progress bar (only on main process)
    if ddp_info.is_main_process:
        pbar = tqdm(dataloader, desc="Processing scenes", total=len(dataloader))
    else:
        pbar = dataloader

    for batch in pbar:
        batch_gpu = {
            k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        if "LVSM" in config.model.class_name:
            result = model(batch_gpu)
            result = model.module.render_video(result, **config.inference.render_video_config)
        elif "rayzer" in config.model.class_name:
            result = model(
                batch_gpu,
                create_visual=True,
                render_video=config.inference.get("render_video", False),
            )

        # Use original filename export if enabled
        if config.inference.get("save_with_original_filenames", False):
            export_results_with_original_filenames(
                result, batch, inference_out_dir, save_metadata=True
            )
        else:
            export_results(
                result,
                batch,
                inference_out_dir,
                compute_metrics=config.inference.get("compute_metrics"),
                config=config,
            )

    if ddp_info.is_main_process:
        pbar.close()
    torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)

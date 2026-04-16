# Copyright (c) 2025 Hanwen Jiang, Xuweiyi Chen. Adapted for WildRayZer from the RayZer project.

import importlib
import os
import time
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0


# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up wandb and backup source code
if ddp_info.is_main_process:
    init_wandb_and_backup(config)
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

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

# Set up dataloader
datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)

# Set up total train steps
total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = (
    total_train_steps * grad_accum_steps
)  # real train steps when using gradient accumulation
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))

# Set up model
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)

# Freeze all parameters except motion mask predictor (if motion_mask_only_training is True)
if config.model.get("motion_mask_only_training", False):
    if ddp_info.global_rank == 0:
        print("=" * 80)
        print("MOTION MASK ONLY TRAINING MODE")
        print("Freezing all parameters except motion_mask_predictor...")
        print("=" * 80)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only motion mask predictor
    if hasattr(model, "motion_mask_predictor") and model.motion_mask_predictor is not None:
        for param in model.motion_mask_predictor.parameters():
            param.requires_grad = True
        if ddp_info.global_rank == 0:
            trainable_mask_params = sum(p.numel() for p in model.motion_mask_predictor.parameters())
            print(f"Unfroze motion_mask_predictor: {trainable_mask_params:,} parameters")
            print("=" * 80)

# Use find_unused_parameters=True for motion-mask-only training OR when using standalone DINOv3 backbone
find_unused = config.model.get("motion_mask_only_training", False) or config.model.get(
    "use_find_unused_parameters", False
)
model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=find_unused)

# Set up optimizer
optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())

# Set up lr scheduler
scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)

# Load initialization checkpoint (if specified) - for model weights only
if config.training.get("init_checkpoint", "") != "":
    init_ckpt_path = config.training.init_checkpoint
    print_rank0(f"Loading initialization checkpoint from: {init_ckpt_path}")
    dist.barrier()  # Synchronize before loading
    init_checkpoint = torch.load(init_ckpt_path, map_location="cpu")
    if isinstance(model, DDP):
        missing_keys, unexpected_keys = model.module.load_state_dict(
            init_checkpoint["model"], strict=False
        )
    else:
        missing_keys, unexpected_keys = model.load_state_dict(
            init_checkpoint["model"], strict=False
        )
    dist.barrier()  # Synchronize after loading
    print_rank0(f"✓ Loaded init checkpoint from {os.path.abspath(init_ckpt_path)}")
    print_rank0(f"  Missing keys: {len(missing_keys)}")
    print_rank0(f"  Unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print_rank0(f"  First 5 missing: {missing_keys[:5]}")
    if unexpected_keys:
        print_rank0(f"  First 5 unexpected: {unexpected_keys[:5]}")

# Set up checkpoint resume (for resuming training state)
if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
)

# Set up grad scaler
enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler("cuda", enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()

# Set up curriculum-related variables
# Use actual dataloader length which accounts for DDP sharding and drop_last
num_fwdbwd_passes_per_epoch = len(dataloader)
if config.training.view_selector.get("use_curriculum", False):
    max_iter_epoch = config.training.get(
        "max_iter_epoch", 100
    )  # use a small number for iter per epoch, more flexible for curriculum
    num_fwdbwd_passes_per_epoch = min(num_fwdbwd_passes_per_epoch, max_iter_epoch)

# Start training
while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(
        cur_train_step * (total_batch_size / grad_accum_steps) // num_fwdbwd_passes_per_epoch
    )

    # Update deterministic view sampling every iteration (step-based refresh)
    if hasattr(dataset, "update_step"):
        dataset.update_step(cur_train_step)

    # Update dataloader
    if cur_train_step % num_fwdbwd_passes_per_epoch == 0:
        print(
            f"ddp_rank={ddp_info.local_rank}, Resetting dataloader epoch to {cur_epoch}; might take a while..."
        )

        if config.training.view_selector.get("use_curriculum", False):
            print(
                f"ddp_rank={ddp_info.local_rank}, Resetting dataset iteration number to {cur_train_step}; might take a while..."
            )
            dataset.update_iteration(
                cur_train_step
            )  # update dataset iter number for setting view interval for curriculum
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size_per_gpu,
                shuffle=False,
                num_workers=config.training.num_workers,
                persistent_workers=True,
                pin_memory=False,
                drop_last=True,
                prefetch_factor=config.training.prefetch_factor,
                sampler=datasampler,
            )
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)

    # Forward pass
    data = next(dataloader_iter)
    batch = {k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    create_visual = ((cur_train_step - 1) == start_train_step) or (
        cur_train_step % config.training.vis_every == 0
    )
    render_video = create_visual and config.training.get("render_video", False)

    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        if "LVSM" in config.model.class_name:
            ret_dict = model(batch)
        elif "rayzer" in config.model.class_name:
            ret_dict = model(batch, create_visual=create_visual, render_video=render_video)
        else:
            raise NotImplementedError(f"Model {config.model.class_name} is not supported")

    # Backward pass
    update_grads = (
        cur_train_step + 1
    ) % grad_accum_steps == 0 or cur_train_step == total_train_steps
    if update_grads:
        with model.no_sync():  # no sync grads for efficiency
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    else:
        scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    cur_train_step += 1

    total_grad_norm = 0.0  # Initialize to 0.0 instead of None to avoid NameError
    if update_grads:
        skip_optimizer_step = False
        # Skip optimizer step if loss is NaN or Inf
        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        total_grad_norm = 0.0  # Initialize to 0.0 instead of None to avoid NameError
        # Check gradient norm and update optimizer if everything is fine
        if not skip_optimizer_step:
            # Unscales the gradients
            scaler.unscale_(optimizer)
            # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)

            # visualize the grad norm of each layer of our transformer (FOR DEBUG)
            if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                grad_norms = {}  # Dictionary to store norms per layer
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Some parameters might not have gradients
                        grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                for layer_name, grad_norm in grad_norms.items():
                    wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    optim_param_list, max_norm=config.training.grad_clip_norm
                ).item()

                if total_grad_norm > config.training.grad_clip_norm * 2.0:
                    print(
                        f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}"
                    )

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get(
                    "allowed_gradnorm_factor", 5
                )
                if (total_grad_norm > allowed_gradnorm) and (
                    cur_train_step > config.training.get("no_pass_steps", -1)
                ):
                    skip_optimizer_step = True
                    print(
                        f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step"
                    )

                # show grad norm in wandb if it's too large
                display_grad_norm = (
                    total_grad_norm > config.training.grad_clip_norm * 2.0
                    or total_grad_norm > allowed_gradnorm
                )
                if display_grad_norm and ddp_info.is_main_process:
                    wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

            # since skip flag may be updated because of grad norm, we check it again
            if not skip_optimizer_step:
                scaler.step(optimizer)
                cur_param_update_step += 1

        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Create log and save checkpoint
    if ddp_info.is_main_process:
        # Only log scalar tensors (skip visual, etc.)
        loss_dict = {
            k: float(f"{v.item():.6f}")
            for k, v in ret_dict.loss_metrics.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (
            cur_train_step < 100 + start_train_step
        ):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            # Add loss values
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # log in wandb
        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step,
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(
                log_dict,
                step=cur_train_step,
            )

        # save checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (
            cur_train_step == total_train_steps
        ):
            if isinstance(model, DDP):
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt"
            )
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")

        # export intermediate visualization results
        if create_visual:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            os.makedirs(vis_path, exist_ok=True)
            visualize_intermediate_results(vis_path, ret_dict)
            torch.cuda.empty_cache()
            model.train()

    if create_visual:
        torch.cuda.empty_cache()
        dist.barrier()


dist.barrier()
dist.destroy_process_group()

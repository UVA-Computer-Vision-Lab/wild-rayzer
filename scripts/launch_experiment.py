#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional


def create_slurm_script(args, exp_dir, config_name, mode="train"):
    """Create SLURM sbatch script"""
    job_name = f"{args.exp_name}_{mode}"
    time_limit = "4:00:00" if mode == "eval" else args.time

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --gres=gpu:{args.gpus}
#SBATCH --time={time_limit}
#SBATCH --account={args.account}
#SBATCH --mail-user={args.email}
#SBATCH --mail-type=ALL
#SBATCH --output={exp_dir}/log/slurm_%j.out
#SBATCH --error={exp_dir}/log/slurm_%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Experiment: {args.exp_name}"
echo "Mode: {mode}"

# Change to code directory
cd {exp_dir}/code

# Create results directory
mkdir -p {exp_dir}/results/checkpoints
mkdir -p {exp_dir}/results/evaluation
"""

    if mode == "train":
        script += f"""
# Run training with torchrun
torchrun --standalone --nproc_per_node={args.gpus} train.py --config {config_name}

echo "Training completed!"
"""
    else:  # eval mode
        # Use the latest checkpoint from training - find the latest one automatically
        checkpoint_path = f"{exp_dir}/results/checkpoints"
        eval_cmd = (
            f"torchrun --nproc_per_node 1 --master_port 29502 inference.py --config {config_name}"
        )
        eval_cmd += f' training.checkpoint_dir="$(ls -t {checkpoint_path}/ckpt_*.pt | head -1)"'

        # Add dataset configuration if provided
        if args.eval_dataset and args.eval_dataset_path:
            if args.eval_dataset == "re10k":
                eval_cmd += f' training.dataset_name="data.dataset_scene_official.Dataset"'
                eval_cmd += f' training.dataset_path="{args.eval_dataset_path}/test_full_list.txt"'
                eval_cmd += f" training.num_views=5"
                eval_cmd += f" training.num_input_views=2"
                eval_cmd += f" training.num_target_views=3"
                eval_cmd += f' inference.view_idx_file_path="./data/evaluation_index_re10k.json"'
                eval_cmd += f' inference.render_video_config.traj_type="interpolate"'
            elif args.eval_dataset == "stereo4d":
                eval_cmd += f' training.dataset_name="data.simple_stereo4d_adapter.Dataset"'
                eval_cmd += f' training.dataset_path="{args.eval_dataset_path}"'
                eval_cmd += f' training.split="test"'
                eval_cmd += f" training.num_views=8"
                eval_cmd += f" training.num_input_views=2"
                eval_cmd += f" training.num_target_views=6"
                eval_cmd += f' inference.render_video_config.traj_type="interpolate"'
                eval_cmd += f" inference.max_eval_samples=100"
                # Add evaluation index file for proper evaluation
                eval_cmd += f' inference.view_idx_file_path="./data/evaluation_index_re10k.json"'
            else:
                eval_cmd += f' training.dataset_name="{args.eval_dataset}"'
                eval_cmd += f' training.dataset_path="{args.eval_dataset_path}"'
                eval_cmd += f' inference.render_video_config.traj_type="dataset"'
        else:
            eval_cmd += f' inference.render_video_config.traj_type="dataset"'

        # Add default evaluation settings
        eval_cmd += f" inference.if_inference=true"
        eval_cmd += f" inference.compute_metrics=true"
        eval_cmd += f" inference.render_video=true"
        eval_cmd += f' inference_out_dir="{exp_dir}/results/evaluation"'
        if args.max_eval_samples is not None:
            eval_cmd += f" inference.max_eval_samples={args.max_eval_samples}"

        script += f"""
# Run evaluation with torchrun
{eval_cmd}

echo "Evaluation completed!"
"""

    return script


def create_local_script(args, exp_dir, config_name, mode="train"):
    """Create local launch script"""
    script = f"""#!/bin/bash
echo "=== Local {mode.upper()} Job ==="
echo "Experiment: {args.exp_name}"
echo "Mode: {mode}"
echo "GPUs: {args.gpus}"
echo "Started at: $(date)"

# Change to code directory
cd {exp_dir}/code

# Create results directory
mkdir -p {exp_dir}/results/checkpoints
mkdir -p {exp_dir}/results/evaluation

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0
if [ {args.gpus} -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((args.gpus-1)))
fi
"""

    if mode == "train":
        script += f"""
# Run training with torchrun
torchrun --standalone --nproc_per_node={args.gpus} train.py --config {config_name} 2>&1 | tee {exp_dir}/log/local.out

echo "Training completed at: $(date)"
"""
    else:  # eval mode
        # Use the latest checkpoint from training - find the latest one automatically
        checkpoint_path = f"{exp_dir}/results/checkpoints"
        eval_cmd = (
            f"torchrun --nproc_per_node 1 --master_port 29502 inference.py --config {config_name}"
        )
        eval_cmd += f' training.checkpoint_dir="$(ls -t {checkpoint_path}/ckpt_*.pt | head -1)"'

        # Add dataset configuration if provided
        if args.eval_dataset and args.eval_dataset_path:
            if args.eval_dataset == "re10k":
                eval_cmd += f' training.dataset_name="data.dataset_scene_official.Dataset"'
                eval_cmd += f' training.dataset_path="{args.eval_dataset_path}/test_full_list.txt"'
                eval_cmd += f" training.num_views=5"
                eval_cmd += f" training.num_input_views=2"
                eval_cmd += f" training.num_target_views=3"
                eval_cmd += f' inference.render_video_config.traj_type="interpolate"'
            elif args.eval_dataset == "stereo4d":
                eval_cmd += f' training.dataset_name="data.simple_stereo4d_adapter.Dataset"'
                eval_cmd += f' training.dataset_path="{args.eval_dataset_path}"'
                eval_cmd += f' training.split="test"'
                eval_cmd += f" training.num_views=8"
                eval_cmd += f" training.num_input_views=2"
                eval_cmd += f" training.num_target_views=6"
                eval_cmd += f' inference.render_video_config.traj_type="interpolate"'
                # Add evaluation index file for proper evaluation
                eval_cmd += f' inference.view_idx_file_path="./data/evaluation_index_re10k.json"'
            else:
                eval_cmd += f' training.dataset_name="{args.eval_dataset}"'
                eval_cmd += f' training.dataset_path="{args.eval_dataset_path}"'
                eval_cmd += f' inference.render_video_config.traj_type="dataset"'
        else:
            eval_cmd += f' inference.render_video_config.traj_type="dataset"'

        # Add default evaluation settings
        eval_cmd += f" inference.if_inference=true"
        eval_cmd += f" inference.compute_metrics=true"
        eval_cmd += f" inference.render_video=true"
        eval_cmd += f' inference_out_dir="{exp_dir}/results/evaluation"'
        if args.max_eval_samples is not None:
            eval_cmd += f" inference.max_eval_samples={args.max_eval_samples}"

        script += f"""
# Run evaluation with torchrun
{eval_cmd} 2>&1 | tee {exp_dir}/log/local.out

echo "Evaluation completed at: $(date)"
"""

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Launch experiment with organized directory structure"
    )
    parser.add_argument("exp_name", help="Experiment name")
    parser.add_argument("config_file", help="Config file path (relative to project root)")

    # Resource configuration
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument(
        "--time", default="48:00:00", help="Time limit for training (default: 48:00:00)"
    )
    parser.add_argument("--partition", default="ai", help="SLURM partition (default: ai)")
    parser.add_argument(
        "--account", default="nairr250073-ai", help="SLURM account (default: nairr250073-ai)"
    )
    parser.add_argument(
        "--email",
        default="xuweic@email.virginia.edu",
        help="Email for notifications (default: xuweic@email.virginia.edu)",
    )
    parser.add_argument("--mem", default="128GB", help="Memory per node (default: 128GB)")
    parser.add_argument("--cpus-per-task", type=int, default=64, help="CPUs per task (default: 64)")

    # Optional evaluation arguments (will use defaults if not provided)
    parser.add_argument("--eval_dataset", help="Dataset name for evaluation (optional)")
    parser.add_argument("--eval_dataset-path", help="Dataset path for evaluation (optional)")
    parser.add_argument(
        "--max-eval-samples",
        type=Optional[int],
        default=None,
        help="Maximum evaluation samples (default: None)",
    )

    args = parser.parse_args()

    # Project paths — override via WILDRAYZER_PROJECT_ROOT env var.
    project_root = Path(os.environ.get("WILDRAYZER_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
    exp_dir = project_root / "experiments" / args.exp_name

    print(f"=== Setting up experiment: {args.exp_name} ===")
    print(f"Config: {args.config_file}")
    print(f"GPUs: {args.gpus}")
    print(f"Time limit: {args.time}")
    print(f"Memory: {args.mem}")
    print(f"CPUs per task: {args.cpus_per_task}")
    print(f"Email: {args.email}")
    print(f"Experiment dir: {exp_dir}")

    if args.eval_dataset:
        print(f"Eval Dataset: {args.eval_dataset}")
    if args.eval_dataset_path:
        print(f"Eval Dataset Path: {args.eval_dataset_path}")
    if args.max_eval_samples is not None:
        print(f"Max Eval Samples: {args.max_eval_samples}")

    # Create experiment directory structure
    for subdir in ["code", "log", "launch", "results"]:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Copy core code files (excluding large data directory)
    print("Copying code files...")
    code_files = ["train.py", "inference.py", "setup.py"]
    code_dirs = ["model", "utils", "configs"]

    for file in code_files:
        if (project_root / file).exists():
            shutil.copy2(project_root / file, exp_dir / "code" / file)

    for dir_name in code_dirs:
        src_dir = project_root / dir_name
        dst_dir = exp_dir / "code" / dir_name
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)

    # Create symlink to data directory instead of copying
    print("Creating symlink to data directory...")
    data_symlink = exp_dir / "code" / "data"
    if data_symlink.exists():
        data_symlink.unlink()
    data_symlink.symlink_to(project_root / "data")

    # Copy and modify config
    print("Copying config...")
    config_src = project_root / args.config_file
    config_name = config_src.name
    config_dst = exp_dir / "code" / config_name

    if not config_src.exists():
        print(f"Error: Config file {config_src} does not exist!")
        sys.exit(1)

    # Read and modify config to update checkpoint directory
    with open(config_src, "r") as f:
        config_content = f.read()

    # Update checkpoint directory to point to results
    checkpoint_dir = f"{exp_dir}/results/checkpoints"
    lines = config_content.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("checkpoint_dir:"):
            lines[i] = f"  checkpoint_dir: {checkpoint_dir}"
            break

    with open(config_dst, "w") as f:
        f.write("\n".join(lines))

    # Create ALL launch scripts
    print("Creating launch scripts...")

    # Generate all combinations
    scripts = {}
    script_configs = [("train", "slurm"), ("train", "local"), ("eval", "slurm"), ("eval", "local")]

    for mode, launch in script_configs:
        script_name = f"{mode}_{launch}.sh"
        if launch == "slurm":
            scripts[script_name] = create_slurm_script(args, exp_dir, config_name, mode=mode)
        else:
            scripts[script_name] = create_local_script(args, exp_dir, config_name, mode=mode)

    # Write all scripts
    for script_name, script_content in scripts.items():
        launch_file = exp_dir / "launch" / script_name
        with open(launch_file, "w") as f:
            f.write(script_content)
        launch_file.chmod(0o755)

    # Create experiment README
    readme_content = f"""# Experiment: {args.exp_name}

## Setup
- **Config**: {args.config_file}
- **GPUs**: {args.gpus}
- **Time limit**: {args.time}
- **Memory**: {args.mem}
- **CPUs per task**: {args.cpus_per_task}
- **Partition**: {args.partition}
- **Account**: {args.account}
- **Email**: {args.email}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    if args.eval_dataset:
        readme_content += f"- **Eval Dataset**: {args.eval_dataset}\n"
    if args.eval_dataset_path:
        readme_content += f"- **Eval Dataset Path**: {args.eval_dataset_path}\n"
    if args.max_eval_samples is not None:
        readme_content += f"- **Max Eval Samples**: {args.max_eval_samples}\n"

    readme_content += f"""
## Directory Structure
- `code/`: Snapshot of code and config used for this experiment
- `log/`: SLURM and local output logs
- `launch/`: All launch scripts (train_slurm.sh, train_local.sh, eval_slurm.sh, eval_local.sh)
- `results/`: Training checkpoints and evaluation results

## Usage
```bash
cd {exp_dir}/launch
```

### Training
**SLURM:**
```bash
sbatch train_slurm.sh
```

**Local:**
```bash
./train_local.sh
```

### Evaluation (uses latest checkpoint from training)
**SLURM:**
```bash
sbatch eval_slurm.sh
```

**Local:**
```bash
./eval_local.sh
```

### Monitoring
**SLURM:**
```bash
squeue -u $USER
tail -f {exp_dir}/log/slurm_*.out
```

**Local:**
```bash
tail -f {exp_dir}/log/local.out
```

## Files
- Training script: `code/train.py`
- Evaluation script: `code/inference.py`
- Config: `code/{config_name}`
- Checkpoints: `results/checkpoints/`
"""

    with open(exp_dir / "README.md", "w") as f:
        f.write(readme_content)

    print("")
    print(f"✅ Experiment setup complete!")
    print("")
    print("Directory structure:")
    print(f"{exp_dir}/")
    print("├── code/          # Code snapshot")
    print("├── log/           # SLURM/Local logs")
    print("├── launch/        # All launch scripts")
    print("├── results/       # Training & evaluation outputs")
    print("└── README.md      # Experiment info")
    print("")
    print("Generated launch scripts:")
    print(f"  {exp_dir}/launch/")
    print("  ├── train_slurm.sh    # Training on SLURM")
    print("  ├── train_local.sh    # Training locally")
    print("  ├── eval_slurm.sh     # Evaluation on SLURM (uses training checkpoints)")
    print("  └── eval_local.sh     # Evaluation locally (uses training checkpoints)")
    print("")
    print("Usage examples:")
    print(f"  cd {exp_dir}/launch")
    print("  sbatch train_slurm.sh    # Start training on SLURM")
    print("  ./train_local.sh         # Start training locally")
    print("  sbatch eval_slurm.sh     # Evaluate on SLURM")
    print("  ./eval_local.sh          # Evaluate locally")
    print("")
    print("Monitor:")
    print("  squeue -u $USER                    # SLURM jobs")
    print(f"  tail -f {exp_dir}/log/slurm_*.out  # SLURM logs")
    print(f"  tail -f {exp_dir}/log/local.out    # Local logs")


if __name__ == "__main__":
    main()

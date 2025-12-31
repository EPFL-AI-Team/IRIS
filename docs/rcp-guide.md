# RCP Training & Evaluation Guide

Guide for running IRIS VLM training, evaluation, and inference on EPFL's RCP cluster.

## Prerequisites

- Docker image built and pushed to registry
- Videos in `/scratch/iris/videos/`
- FineBio annotations in place

## Development Workflow

The Docker image includes the code at the user's home directory. However, for easy iteration during development, clone the repo to the persistent volume:

```bash
cd /scratch/iris
git clone <repo-url> iris_repo
```

This way you can:
- Edit code without rebuilding the image
- Pull latest changes with `git pull`
- Run scripts directly from the mounted volume

When running jobs, point to the scratch copy:
```bash
--command -- python -m iris.cli.finetune.train \
  --config /scratch/iris/iris_repo/configs/vlm/train_a100.yaml
```

## Training Configuration

Config files are in `configs/vlm/`:

| File | Description |
|------|-------------|
| `train.yaml` | Base config template |
| `train_a100.yaml` | A100-optimized settings (current) |

### Key Config Parameters

**Model:**
```yaml
model:
  name: "Qwen/Qwen2.5-VL-3B-Instruct"
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
```

**LoRA:**
```yaml
lora:
  r: 16          # Rank
  alpha: 32      # Alpha
  dropout: 0.05
```

**Data:**
```yaml
data:
  train_path: "/scratch/iris/finebio_processed/splits/<split>/finebio_train.jsonl"
  val_path: "/scratch/iris/finebio_processed/splits/<split>/finebio_val.jsonl"
  max_frames: 4
  max_pixels: 600000
```

**Training:**
```yaml
training:
  output_dir: "/scratch/iris/checkpoints/<run-name>"
  num_train_epochs: 3
  learning_rate: 2.0e-4      # or 5.0e-5 for fine-tuning
  batch_size: 16
  save_strategy: "steps"
  save_steps: 100
  eval_steps: 100
```

### Starting a New Run

1. Copy an existing config:
   ```bash
   cp configs/vlm/train_a100.yaml configs/vlm/train_my_run.yaml
   ```

2. Modify key fields:
   - `training.output_dir`: New checkpoint directory
   - `data.train_path` / `data.val_path`: Your data splits

3. Run with your config:
   ```bash
   python -m iris.cli.finetune.train \
     --config /scratch/iris/iris_repo/configs/vlm/train_my_run.yaml \
     --wandb-run-name my-descriptive-run-name
   ```

## CLI Reference

### Training

```bash
python -m iris.cli.finetune.train \
  --config <path>           # Config YAML path (required)
  --hardware <profile>      # Hardware profile: v100, mac, etc. (optional)
  --wandb-project <name>    # WandB project (default: iris-qwen-training)
  --wandb-run-name <name>   # Run identifier (optional)
```

### Evaluation

```bash
python -m iris.cli.finetune.evaluate \
  --checkpoint_dir <path>   # Model checkpoint directory
  --val_path <path>         # Validation/test JSONL file
  --batch_size <n>          # Batch size (default: 8)
  --max_samples <n>         # Max samples, 0=all (default: 100)
  --compare-base            # Also evaluate base model for comparison
  --prompt-mode <mode>      # with_visual | without_visual | both
  --eval-name <name>        # Subdirectory name for results
```

### Inference

```bash
python -m iris.cli.finetune.inference \
  --video <path>            # Video file (required)
  --output-dir <path>       # Results directory (required)
  --checkpoint <path>       # Fine-tuned model checkpoint
  --base-only               # Only run base model
  --finetuned-only          # Only run fine-tuned model
  --segment-duration <sec>  # Segment duration (default: 2.0)
  --num-frames <n>          # Frames per segment (default: 4)
  --frame-overlap <n>       # Frame overlap (default: 1)
  --max-new-tokens <n>      # Max tokens to generate (default: 512)
  --prompt <text>           # Custom prompt (uses default if not set)
```

## RCP Commands

### Start Interactive Container

```bash
export WANDB_API_KEY=<your-key>
runai submit iris-interactive \
  --image registry.rcp.epfl.ch/iris-qwen/iris-qwen:dev \
  --pvc aiteam-scratch:/scratch \
  --large-shm \
  --gpu 1 \
  -e PYTHONPATH=/scratch/iris/iris_repo/src \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --attach
```

### Dataset Preprocessing

Run from within the interactive container:

```bash
cd /scratch/iris/iris_repo
python src/iris/dataset/process_dataset.py
python src/iris/dataset/create_training_data.py
```

**Note:** Videos should be in the expected location per `dataset_config.yaml`. FineBio annotations need to be in place before running preprocessing.

### Training Job

```bash
export WANDB_API_KEY=<your-key>
runai submit iris-train \
  --image registry.rcp.epfl.ch/iris-qwen/iris-qwen:dev \
  --image-pull-policy=Always \
  --gpu 1 \
  --large-shm \
  --pvc aiteam-scratch:/scratch \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --backoff-limit 0 \
  --command -- python -m iris.cli.finetune.train \
    --config /scratch/iris/iris_repo/configs/vlm/train_a100.yaml \
    --wandb-run-name <run-name>
```

### Evaluation Job

```bash
runai submit iris-eval \
  --image registry.rcp.epfl.ch/iris-qwen/iris-qwen:dev \
  --pvc aiteam-scratch:/scratch \
  --large-shm \
  --gpu 1 \
  -e PYTHONPATH=/scratch/iris/iris_repo/src \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --backoff-limit 0 \
  --command -- sh -c "cd /scratch/iris/iris_repo && \
    python -m iris.cli.finetune.evaluate \
    --checkpoint_dir /scratch/iris/checkpoints/<checkpoint-name> \
    --val_path /scratch/iris/finebio_processed/splits/<split>/finebio_val.jsonl \
    --batch_size 16"
```

### Quick Evaluation (Interactive)

For quick evaluation during training runs:

```bash
cd /scratch/iris/iris_repo && python -m iris.cli.finetune.evaluate \
  --checkpoint_dir /scratch/iris/checkpoints/<run>/checkpoint-100 \
  --val_path /scratch/iris/finebio_processed/splits/<split>/finebio_val.jsonl \
  --batch_size 16 \
  --max_samples 3
```

### Full Comparison Evaluation

```bash
python -m iris.cli.finetune.evaluate \
  --checkpoint_dir /scratch/iris/checkpoints/<run> \
  --compare-base \
  --prompt-mode both \
  --max_samples 100 \
  --eval-name full_comparison
```

## Docker Image

### Build and Push (Recommended)

Use the build script:

```bash
./build_image.sh [version]   # Default version: dev
```

### Manual Build

```bash
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 . \
  --tag registry.rcp.epfl.ch/iris-qwen/iris-qwen:<version> \
  --build-arg LDAP_GROUPNAME=<group-name> \
  --build-arg LDAP_GID=<group-id> \
  --build-arg LDAP_USERNAME=<username> \
  --build-arg LDAP_UID=<user-id>
docker push registry.rcp.epfl.ch/iris-qwen/iris-qwen:<version>
```

**Note:** LDAP credentials are required for RCP file permissions. Get these from your RCP admin or LDAP settings.

## Directory Structure on RCP

```
/scratch/iris/
├── iris_repo/              # This repository
├── checkpoints/            # Training checkpoints
│   └── qwen3b_finebio_run*/
├── finebio_processed/      # Processed dataset
│   └── splits/
│       └── train_*/
│           ├── finebio_train.jsonl
│           └── finebio_val.jsonl
└── videos/                 # Source videos
```


# Running the Pipeline on EPFL Clusters

Guide for running IRIS on EPFL's Izar (V100) and RCP (A100) clusters.

## Setup for `~/.ssh/config`

Here is the configuration in `~/.ssh/config` to allow you to write `izar` or `rcp` instead of `ip:port` in the terminal and have the login taken care of for you.

Supposing you followed the steps for both clusters to add your public key to `authorized_keys` (IdentityFile points to your corresponding private key).

```bash
Host izar
    HostName izar.hpc.epfl.ch
    User GASPAR_USERNAME
    IdentityFile ~/.ssh/izar_key
    IdentitiesOnly yes

Host rcp
    HostName jumphost.rcp.epfl.ch
    User GASPAR_USERNAME
    IdentityFile ~/.ssh/izar_key
    IdentitiesOnly yes
```

## Running the Demo

### Workflow with Izar (V100 GPUs)

#### On Izar

Assuming you cloned the IRIS repo to `~/projects/IRIS`

```bash
cd ~/projects/IRIS
Sinteract -t 04:00:00 -g gpu:1 -m 32G -q team-ai
hostname
./run_iris.sh
```

#### On Personal Machine (or Raspberry Pi)

**Terminal 1:**
```bash
uv run iris-client
```

**Terminal 2:**
```bash
ssh -N -L 8005:[RUN hostname ON NODE TO SEE]:8001 GASPAR_USERNAME@izar.hpc.epfl.ch
```

Then go to http://localhost:8006

⚠️ **Important:** Make sure to modify the hostname with the output from the `hostname` command on the Izar node.

### Workflow on RCP (A100 GPUs)

**Terminal 1:**
```bash
ssh -N -L 8005:localhost:8005 rcp
```

**Terminal 2:**
```bash
uv run iris-client
```

**Terminal 3 (SSH to RCP):**
```bash
ssh rcp
runai submit iris-server-interactive \
	--image registry.rcp.epfl.ch/iris-qwen/iris-qwen:dev \
	--gpu 1 \
	--node-pools default \
	--pvc aiteam-scratch:/scratch \
	--interactive --attach

# Once in interactive container:
IRIS_CONFIG_FILE=config.rcp.yaml uv run iris-server

# Or if running on a V100:
IRIS_CONFIG_FILE=config.rcp.yaml uv run iris-server --vlm-hardware v100
```

**Terminal 4 (SSH to RCP):**
```bash
ssh rcp
runai port-forward iris-server-interactive --port 8005:8005
```

Then access the client demo at http://localhost:8006

## Dataset Scripts (RCP)

In the interactive container:

```bash
python /scratch/iris/iris_repo/src/iris/cli/finetune/evaluate.py \
    --checkpoint_dir /scratch/iris/checkpoints/qwen3b_finebio_run8_wo_vis \
    --compare-base \
    --prompt-mode both \
    --max_samples 100 \
    --eval-name full_comparison_100_samples_1
```

## Training (RCP)

```bash
export WANDB_API_KEY=your_wandb_api_key_here
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
    --wandb-run-name qwen2.5-3b-lora-finebio-4f-3ep-v1
```

## Building the Docker Image

Use `build_image.sh` for an optimized building experience with BuildKit (allows for multi-platform).

## Evaluation

```bash
runai submit iris-eval \
    --image registry.rcp.epfl.ch/iris-qwen/iris-qwen:dev \
    --image-pull-policy=Always \
    --gpu 1 \
    --node-pools default \
    --large-shm \
    --pvc aiteam-scratch:/scratch \
    -e PYTHONPATH=/scratch/iris/iris_repo/src \
    --backoff-limit 0 \
    --command -- python /scratch/iris/iris_repo/src/iris/cli/finetune/evaluate.py \
    --checkpoint_dir '/scratch/iris/checkpoints/qwen3b_finebio_finetune_iris' \
    --val_path '/scratch/iris/finebio_processed/splits/train_without_vis_analysis/finebio_test.jsonl' \
    --max_samples 0 \
    --batch_size 16 \
    --compare-base \
    --eval-name "report_final_evaluation"
```

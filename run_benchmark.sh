#!/bin/bash

# 1. Load System Modules
# GCC is required for many Python C-extensions (like Pillow/Numpy) to run correctly
#
module load gcc ffmpeg
module load cuda/12.1.1

# nvidia-smi
# /home/mhamelin/projects/IRIS/.venv/bin/python -c "
# import torch
# print('PyTorch:', torch.__version__)
# print('CUDA available:', torch.cuda.is_available())
# if torch.cuda.is_available():
#     print('GPU:', torch.cuda.get_device_name(0))
# "

# 2. Set Environment Variables
# Critical: Keep heavy model weights in /scratch, not /home
export HF_HOME="/scratch/izar/mhamelin/hf_cache"

# Optional: Helps with "CUDA out of memory" fragmentation if hit again with it
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Print Connection Instructions
echo "========================================================"
echo "Benchmark is starting on NODE: $(hostname)"
echo "========================================================"

# 4. Navigate to Project
cd ~/projects/IRIS || exit

# 5. Run the Server
# -u : Unbuffered mode (so logs appear instantly, not in chunks)
# --host 0.0.0.0 : Essential for the tunnel to work
# --port 8001 : Matches your tunnel config
echo "Starting uvicorn..."
uv run python -u -m iris.cli.benchmark

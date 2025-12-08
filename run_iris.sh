#!/bin/bash

# 1. Load System Modules
# GCC is required for many Python C-extensions (like Pillow/Numpy) to run correctly
module load gcc

# 2. Set Environment Variables
# Critical: Keep heavy model weights in /scratch, not /home
export HF_HOME="/scratch/izar/mhamelin/hf_cache"

# Optional: Helps with "CUDA out of memory" fragmentation if you hit it again
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. Print Connection Instructions
# This prints the exact command you need to copy-paste to your Mac
echo "========================================================"
echo "IRIS Server is starting on NODE: $(hostname)"
echo "--------------------------------------------------------"
echo "To connect, open a NEW terminal on your Mac and run:"
echo "ssh -N -L 8001:$(hostname):8001 mhamelin@izar.hpc.epfl.ch"
echo "========================================================"

# 4. Navigate to Project
cd ~/projects/IRIS || exit

# 5. Run the Server
# -u : Unbuffered mode (so logs appear instantly, not in chunks)
# --host 0.0.0.0 : Essential for the tunnel to work
# --port 8001 : Matches your tunnel config
echo "Starting uvicorn..."
uv run python -u -m iris.cli.server --host 0.0.0.0 --port 8001

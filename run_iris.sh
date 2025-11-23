#!/bin/bash

source ~/projects/IRIS/.venv/bin/activate

export HF_HOME="/scratch/izar/mhamelin/hf_cache"

# 3. Debug Info (Optional but helpful)
echo "--------------------------------"
echo "Node:   $(hostname)"
echo "Python: $(uv run which python)"
# Only run nvidia-smi if we actually have a GPU (avoids error on login node)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
else
    echo "GPU:    None detected (or nvidia-smi missing)"
fi
echo "--------------------------------"

cd ~/projects/IRIS

echo "Starting IRIS server on NODE $(hostname)..."
uv run python -m iris.cli.server

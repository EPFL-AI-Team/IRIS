#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# For example, if you want to use an image with pytorch already installed
# FROM nvcr.io/nvidia/pytorch:25.03-py3 or FROM nvcr.io/nvidia/ai-workbench/pytorch:1.0.6
# In this example, we'll use the second image.

FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

#####################################
# Install system dependencies (as root)
#####################################
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} && \
    useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

#####################################
# Dependencies (pre-installation)
#####################################
WORKDIR /home/${LDAP_USERNAME}
RUN chown ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}

COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} pyproject.toml uv.lock ./

ENV UV_CACHE_DIR=/home/${LDAP_USERNAME}/.cache/uv
RUN --mount=type=cache,target=/home/${LDAP_USERNAME}/.cache/uv,uid=${LDAP_UID},gid=${LDAP_GID} \
    uv venv && \
    uv sync --frozen --group server --no-dev --no-install-project

#####################################
# Source code & Final setup
#####################################
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} README.md ./
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} src/ ./src/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} configs/ ./configs/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} config.yaml ./

RUN --mount=type=cache,target=/home/${LDAP_USERNAME}/.cache/uv,uid=${LDAP_UID},gid=${LDAP_GID} \
    uv sync --frozen --group server --no-dev

#####################################
# Environment
#####################################
ENV PATH="/home/${LDAP_USERNAME}/.venv/bin:$PATH" \
    HF_HOME=/scratch/iris/cache/hf_cache \
    TRANSFORMERS_CACHE=/scratch/iris/cache/hf_cache \
    HF_DATASETS_CACHE=/scratch/iris/cache/hf_cache/datasets \
    TORCH_HOME=/scratch/iris/cache/torch_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_VERBOSITY=warning \
    PYTHONUNBUFFERED=1

#####################################
# Create venv and install dependencies
#####################################

# Activate venv by adding to PATH
ENV PATH="/home/${LDAP_USERNAME}/.venv/bin:$PATH"

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

#####################################
# RCP CaaS requirement (Image)
#####################################
FROM nvcr.io/nvidia/pytorch:24.12-py3

#####################################
# RCP CaaS requirement (Storage)
#####################################
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

#####################################
# Install system dependencies (as root)
#####################################
RUN apt update && \
    apt install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Install uv (as root)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

#####################################
# Set up user directory and deps ONLY
#####################################
RUN mkdir -p /home/${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}

# Copy ONLY dependency files first (for caching)
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} pyproject.toml uv.lock README.md ./

#####################################
# Environment setup
#####################################
ENV HF_HOME=/scratch/iris/cache/hf_cache \
    HF_CACHE=/scratch/iris/cache/hf_cache \
    HF_DATASETS_CACHE=/scratch/iris/cache/hf_cache/datasets \
    TORCH_HOME=/scratch/iris/cache/torch_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_VERBOSITY=warning \
    PYTHONUNBUFFERED=1 \
    PATH="/home/${LDAP_USERNAME}/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

#####################################
# Install dependencies with uv (with caching)
#####################################
RUN --mount=type=cache,target=/root/.cache/uv,uid=${LDAP_UID} \
    uv sync --frozen --group server

#####################################
# Copy code AFTER dependencies (code changes don't rebuild deps)
#####################################
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} src/ ./src/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} configs/ ./configs/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} config.yaml ./

USER ${LDAP_USERNAME}

#####################################
# Verification
#####################################
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

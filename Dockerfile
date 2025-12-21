#####################################
# RCP CaaS requirement (Image)
#####################################
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Zurich 

#####################################
# Install Python 3.12 + system deps
#####################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

#####################################
# RCP CaaS requirement (Storage)
#####################################
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

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
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group server

RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}
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

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
RUN apt update && apt install -y curl git && \
    rm -rf /var/lib/apt/lists/*

# Install uv (as root, makes it available system-wide)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

#####################################
# Copy project files
#####################################
RUN mkdir -p /home/${LDAP_USERNAME}

COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} pyproject.toml /home/${LDAP_USERNAME}/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} uv.lock /home/${LDAP_USERNAME}/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} README.md /home/${LDAP_USERNAME}/

WORKDIR /home/${LDAP_USERNAME}

#####################################
# Environment variables (set for user)
#####################################
ENV HF_HOME=/scratch/iris/cache/hf_cache
ENV TRANSFORMERS_CACHE=/scratch/iris/cache/hf_cache
ENV HF_DATASETS_CACHE=/scratch/iris/cache/hf_cache/datasets
ENV TORCH_HOME=/scratch/iris/cache/torch_cache
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TRANSFORMERS_VERBOSITY=warning
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/${LDAP_USERNAME}/.local/bin:$PATH"

#####################################
# Install dependencies with uv
#####################################
RUN uv pip install --system --no-cache --break-system-packages ".[server]"

USER ${LDAP_USERNAME}
#####################################
# Copy rest of project files - AFTER dependencies installed
#####################################
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} src/ /home/${LDAP_USERNAME}/src/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} configs/ /home/${LDAP_USERNAME}/configs/
COPY --chown=${LDAP_USERNAME}:${LDAP_GROUPNAME} config.yaml /home/${LDAP_USERNAME}/

#####################################
# Verification
#####################################
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

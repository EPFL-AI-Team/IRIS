#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# For example, if you want to use an image with pytorch already installed
# FROM nvcr.io/nvidia/pytorch:25.03-py3 or FROM nvcr.io/nvidia/ai-workbench/pytorch:1.0.6
# In this example, we'll use the second image.

FROM nvcr.io/nvidia/pytorch:24.12-py3

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
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

#####################################
# Install system dependencies (as root)
#####################################
RUN apt update && apt install -y curl git

# Copy uv binary from official image (accessible to all users)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN mkdir -p /home/${LDAP_USERNAME}
COPY pyproject.toml /home/${LDAP_USERNAME}/

# Create virtual environment
RUN uv venv

# Install project into venv (uv will use existing PyTorch from system when possible)
RUN uv sync --group server --no-dev

#####################################
# Copy project files
#####################################
COPY README.md /home/${LDAP_USERNAME}/

COPY src/ /home/${LDAP_USERNAME}/src/
COPY configs/ /home/${LDAP_USERNAME}/
COPY config.yaml /home/${LDAP_USERNAME}/

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Set the working directory in your user's home
WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}

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

#####################################
# Create venv and install dependencies
#####################################

# Activate venv by adding to PATH
ENV PATH="/home/${LDAP_USERNAME}/.venv/bin:$PATH"

# Verify installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

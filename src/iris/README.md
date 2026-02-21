# IRIS - Intelligent Real-time Inference System

Real-time video analysis using vision-language models. The system streams frames from a camera or video file to a GPU inference server and displays results live in a web UI.

## Quick Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/EPFL-AI-Team/IRIS
cd IRIS
uv sync
```

Run two terminals:

**Terminal 1 - Inference server** (needs GPU or CPU for testing):
```bash
uv run iris-server
```

**Terminal 2 - Client + Web UI:**
```bash
uv run iris-client
```

Open **http://localhost:8006** in your browser.

## Running on a Cluster

**RCP (A100)** - four terminals:

```bash
# T1: local port-forward
ssh -N -L 8005:localhost:8005 rcp

# T2: local client
uv run iris-client

# T3: start interactive GPU container on RCP
ssh rcp
runai submit iris-server-interactive \
  --image registry.rcp.epfl.ch/iris-qwen/iris-qwen:dev \
  --gpu 1 --node-pools default --pvc aiteam-scratch:/scratch \
  --interactive --attach
# inside container:
IRIS_CONFIG_FILE=configs/config.rcp.yaml uv run iris-server

# T4: expose server port
ssh rcp
runai port-forward iris-server-interactive --port 8005:8005
```

**Izar (V100)** - run on the login node, then tunnel:

```bash
# On Izar node:
Sinteract -t 04:00:00 -g gpu:1 -m 32G -q team-ai
hostname   # note the node name
./run_iris.sh

# On local machine - tunnel + client:
ssh -N -L 8005:<NODE_HOSTNAME>:8001 GASPAR_USERNAME@izar.hpc.epfl.ch
uv run iris-client
```

## Configuration

| Config file               | Use case                |
| ------------------------- | ----------------------- |
| `configs/config.yaml`     | Local / Mac development |
| `configs/config.rcp.yaml` | EPFL RCP cluster        |

Override the config file at runtime:
```bash
IRIS_CONFIG_FILE=configs/config.rcp.yaml uv run iris-server
```

Key CLI flags for `iris-server`:

| Flag                             | Description                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------- |
| `--model-id smolvlm`             | Which VLM to load (`smolvlm`, `smolvlm2`, `qwen2.5-3b`, `qwen2.5-7b`, or HF path) |
| `--vlm-hardware mac\|v100\|a100` | Hardware profile (dtype, attention, quantization)                                 |
| `--port 8005`                    | Port to listen on                                                                 |

## Fine-tuning & Evaluation

Training, evaluation and inference scripts live in [`sft-vlm-finetune/`](../../sft-vlm-finetune/README.md).

## Further Reading

| Doc                                                    | Contents                           |
| ------------------------------------------------------ | ---------------------------------- |
| [`docs/setup.md`](../../docs/setup.md)                 | Full setup, Raspberry Pi, SSL      |
| [`docs/cluster-setup.md`](../../docs/cluster-setup.md) | Izar and RCP cluster workflows     |
| [`docs/rcp-guide.md`](../../docs/rcp-guide.md)         | Training, evaluation, Docker image |
| [`docs/API.md`](../../docs/API.md)                     | REST and WebSocket API reference   |

# Setup Guide

Complete setup instructions for running IRIS locally or on a cluster.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU server with CUDA support (for inference)

## Installation
```bash
git clone https://github.com/EPFL-AI-Team/IRIS
cd IRIS
uv sync
```

## Local Development

The system requires two processes: a server (GPU) and a client (camera/web interface).

### Terminal 1: Server
```bash
uv run iris-server
```

Server runs on `http://localhost:8005` by default.

### Terminal 2: Client
```bash
uv run iris-client
```

Web interface available at `http://localhost:8006`

## Raspberry Pi Setup

The idea of this setup is to ensure the Pi can communicate to the cluster, if the server script is running there.

### Generate SSL Certificate (First Time Only, for SCITAS or RCP usage)

For HTTPS streaming from Raspberry Pi:
```bash
mkdir -p ~/iris-certs
cd ~/iris-certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=$(hostname -I | awk '{print $1}')"
```

### Enable HTTPS

In `config.yaml`, set:
```yaml
client:
  web:
    use_ssl: true
    cert_dir: "~/iris-certs"
```

### Start Client
```bash
uv run iris-client
```

Connect via `https://<raspberry-pi-ip>:8006`

## Configuration

Two config files are provided in the `configs/` directory:

- `configs/config.yaml` - Local/Mac development
- `configs/config.rcp.yaml` - EPFL cluster deployment

Using arguments with the scripts you're running, you can specify which config you want to use. By default, the usual `configs/config.yaml` is used.

**Key settings:**
- `server.model_id` - Model checkpoint path
- `server.vlm_hardware` - Hardware profile (`mac`, `v100`, `a100`)
- `server.batch_inference.enabled` - Enable batch processing

See [Configuration Reference](API.md#configuration) for all options.

## Cluster Deployment

For running on EPFL clusters (Izar/RCP), see [Cluster Setup Guide](cluster-setup.md).

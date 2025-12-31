# IRIS

A research project done with the AI Team by Myriam Benlamri (Data Science MSc 2nd year) and Marcus Hamelink (Computer Science BSc 3rd year) as a collaborative research semester project.

More info at [https://epflaiteam.ch/projects/iris](https://epflaiteam.ch/projects/iris)


## Set up

### Client

On Your Raspberry Pi, generate the self-signed certificate:
```bash
mkdir -p ~/iris-certs
cd ~/iris-certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem \
  -out cert.pem \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=$(hostname -I | awk '{print $1}')"
```

To use HTTPS, make sure to specify `use_ssl=true` under `client` in `config.yaml`. In that case, make sure to connect to the HTTPS IP.

Run `uv run iris-client` to start a client instance.


### Server

```bash
uv sync
uv sync --group server

```

#### Running the pipeline

Run `uv run iris-server` to start a server instance.


## VLM Training & Evaluation

For training and evaluation on RCP, see [docs/rcp-guide.md](docs/rcp-guide.md).

**Quick CLI reference:**

```bash
# Training
python -m iris.cli.finetune.train --config <path> --wandb-run-name <name>

# Evaluation
python -m iris.cli.finetune.evaluate --checkpoint_dir <path> --val_path <path>

# Inference on video
python -m iris.cli.finetune.inference --video <path> --output-dir <path> --checkpoint <path>
```

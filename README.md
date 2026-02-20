# IRIS: Intelligent Recognition and Interpretation System

> Automated laboratory documentation using wearable cameras and vision-language models  
> Semester Project | EPFL AI Team | Fall 2025

A research collaboration between Annaelle Myriam Benlamri (MSc Data Science) and Marcus Hamelink (BSc Computer Science).

**Presented at**: [AMLD 2026](https://appliedmldays.org/) | **Project Page**: [epflaiteam.ch/projects/iris](https://epflaiteam.ch/projects/iris)

---

## What We Built

Manual lab documentation is error-prone. What if both your hands are busy? What if you forget to write down critical details?

IRIS solves this by automatically documenting laboratory procedures in real-time:

1) **End-to-end pipeline:** Camera on glasses → Raspberry Pi → WebSocket → GPU server

![Sketch of how the pipeline works](img/system-architecture-sketch.jpg)

2) **Fine-tune and model training:** Different Qwen2.5-VL (3B) models to generate documentation from video frames. This went from standard SFT, to a custom action-recognition vision encoder fused with a Qwen model.

**Demo use case**: Colony counting workflows at CHUV (Lausanne University Hospital)

---

## Project Components

### Action Recognition (Annaelle Myriam Benlamri)

Knowledge distillation and fusion MLP architecture for egocentric video action recognition.

**- Report**: [PDF](reports/Annaelle-Benlamri-Action-Recognition-Report.pdf) | **Details**: [action-recognition/](action-recognition/)

### VLM Fine-tuning & Pipeline (Marcus Hamelink)

Fine-tuned Qwen2.5-VL (3B) on FineBio dataset and built an end-to-end streaming system spanning client capture, network transport, and GPU-accelerated inference. The architecture diagram below illustrates the complete pipeline from camera to inference results.

<p align="center">
  <img src="img/system-architecture-diagram.jpg" alt="System architecture diagram" width="300">
</p>

**- Code**: [`src/iris/`](src/iris/) | **Model**: [HuggingFace](https://huggingface.co/animarcus/iris-qwen2.5-vl-3b-finebio) | **Report**: [PDF](reports/Marcus-Hamelink-IRIS-VLM-Report.pdf)  
**- Training guide**: [vlm-finetuning/](vlm-finetuning/) | [docs/rcp-guide.md](docs/rcp-guide.md)

---

## Getting Started (demo)
```bash
git clone https://github.com/your-username/IRIS-semester-project
cd IRIS-semester-project
uv sync

# Run server + client (see setup guide for details)
uv run iris-server  # Terminal 1 (GPU machine)
uv run iris-client  # Terminal 2 (local/RPi)
```

**→ Full setup instructions**: [docs/setup.md](docs/setup.md)  
**→ Cluster deployment**: [docs/cluster-setup.md](docs/cluster-setup.md)

---

## Documentation

- [Setup Guide](docs/setup.md) - Local development and deployment
- [API Reference](docs/API.md) - REST/WebSocket endpoints
- [Training Guide](docs/rcp-guide.md) - VLM fine-tuning and evaluation
- [Cluster Deployment](docs/cluster-setup.md) - EPFL Izar/RCP clusters

---

## Acknowledgments

**Supervisor**: Prof. Andrea Cavallaro (EPFL AI Team)  
**Track Lead**: Louis Vasseur (EPFL AI Team)  
**Collaboration**: CHUV (Lausanne University Hospital)

---

**License**: Apache 2.0

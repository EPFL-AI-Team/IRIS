# IRIS: Intelligent Recognition and Interpretation System

> Automated laboratory documentation using wearable cameras and vision-language models  
> Semester Project | EPFL AI Team | Fall 2025

A research collaboration between Annaelle Myriam Benlamri (MSc Data Science) and Marcus Hamelink (BSc Computer Science).

**Presented at**: [AMLD 2026](https://appliedmldays.org/) | **Project Page**: [epflaiteam.ch/projects/iris](https://epflaiteam.ch/projects/iris)

---

## What We Built

Manual lab documentation is error-prone - what if both your hands are busy? What if you forget to write down critical details?

IRIS solves this by automatically documenting laboratory procedures in real-time:

- **Hardware**: Camera on glasses → Raspberry Pi → WebSocket → GPU server
- **AI**: Fine-tuned Qwen2.5-VL (3B) generates structured JSON logs
- **Latency**: 3-4 seconds on A100 GPU for real-time feedback

**Demo use case**: Colony counting workflows at CHUV (Lausanne University Hospital)

---

## Getting Started
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

## Project Components

### Action Recognition (Annaelle Myriam Benlamri)

Knowledge distillation and fusion MLP architecture for egocentric video action recognition.

**- Report**: [PDF](reports/Annaelle-Benlamri-Action-Recognition-Report.pdf) | **Details**: [action-recognition/](action-recognition/)

### VLM Fine-tuning & Pipeline (Marcus Hamelink)

Fine-tuned Qwen2.5-VL (3B) on FineBio dataset. Built end-to-end system (client, server, inference).

**- Code**: [`src/iris/`](src/iris/) | **Model**: [HuggingFace](https://huggingface.co/animarcus/iris-qwen2.5-vl-3b-finebio) | **Report**: [PDF](reports/Marcus-Hamelink-IRIS-VLM-Report.pdf)  
**- Training guide**: [vlm-finetuning/](vlm-finetuning/) | [docs/rcp-guide.md](docs/rcp-guide.md)

---

## Documentation

- [Setup Guide](docs/setup.md) - Local development and deployment
- [API Reference](docs/API.md) - REST/WebSocket endpoints
- [Training Guide](docs/rcp-guide.md) - VLM fine-tuning and evaluation
- [Cluster Deployment](docs/cluster-setup.md) - EPFL Izar/RCP clusters
- [Architecture Notes](docs/architecture.md) - System design decisions

---

## Acknowledgments

**Supervisor**: Prof. Andrea Cavallaro (EPFL AI Team)  
**Track Lead**: Louis Vasseur (EPFL AI Team)  
**Collaboration**: CHUV (Lausanne University Hospital)

---

**License**: Apache 2.0

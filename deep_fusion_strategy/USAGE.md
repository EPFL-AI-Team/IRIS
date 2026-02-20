# Usage Guide

This document provides instructions to use the results reported in this project.


## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/videomae-qwen-fusion.git
cd videomae-qwen-fusion
```

### 2. Create Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download Pre-trained Models

```bash
huggingface-cli download AnnaelleMyriam/videomaev2-finetuned-finebio
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
```
Or use your finetuned videomaev2 encoder instead of "AnnaelleMyriam/videomaev2-finetuned-finebio"

## Dataset Preparation

See [DATASET.md](DATASET.md) for full details.

### Format

Expected directory structure:
```
data/
├── annotations/
│   ├── train_annotations.json      # Training set annotations
│   ├── val_annotations.json        # Validation set annotations
│   └── test_annotations.json       # Test set annotations
├── splits/
│   ├── train.csv                   # Training video IDs
│   ├── val.csv                     # Validation video IDs
│   └── test.csv                    # Test video IDs
└── class_mappings/
    └── class_mapping.json          # Action class mapping 
└── videos/
    └── [your video frames]
```

Annotation format:
```json
[
    {
        "video_path": "path/to/video_frames/",
        "frames": ["frame_0001.jpg", "frame_0002.jpg", ...],
        "caption": "Ground truth protocol description"
    }
]
```

## Training

### Configuration

All hyperparameters are in `configs/config_videomae_qwen_contrastive_new_resampler.yaml`:

**Key settings:**
- Random seed: `42`
- Precision: `bf16-mixed`
- Num frames: `16`
- Batch size: `1`
- Gradient accumulation: `32`
- Learning rate: `1e-4`
- Epochs: `20`
- Optimizer: AdamW with weight decay `0.01`
- Scheduler: Cosine with warmup (500 steps)

### Run Training

```bash
bash train.sh
```

## Checkpoint Availability

Trained checkpoints available at:
- Hugging Face: `?`
- Or train from scratch using instructions above

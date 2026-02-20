# Model Card: VideoMAE-Qwen Fusion

## Model Description

**Model Name:** VideoMAE-Qwen Fusion for Laboratory Protocol Documentation

**Model Type:** Video-to-Text Generation (Multimodal)

**Architecture:** Deep fusion of VideoMAE-v2 encoder and Qwen2.5-VL-3B-Instruct language model via Perceiver Resampler and MLP projection.

## Model Details

### Overview

This model generates natural language descriptions of laboratory protocols from video recordings. It combines temporal visual understanding (VideoMAE-v2) with language generation capabilities (Qwen2.5-VL-3B-Instruct) through a learned fusion mechanism.

### Architecture Components

1. **Video Encoder**: VideoMAE-v2 (ViT-Base)
   - Pre-trained (https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/distill)
   - Fine-tuned on FineBio (32 lab actions)
   - Model: `AnnaelleMyriam/videomaev2-finetuned-finebio`
   - Status: Frozen during fusion training

2. **Perceiver Resampler**
   - Compresses 1568 VideoMAE tokens to 64 tokens
   - 2 layers, 8 attention heads
   - Learnable query tokens
   - Status: Trainable

3. **MLP Projector**
   - Projects VideoMAE embeddings (768D) to Qwen space (3584D)
   - 3 layers with hidden dim 4096
   - LayerNorm + GELU activation
   - Status: Trainable

4. **Language Model**: Qwen2.5-VL-3B-Instruct
   - 3B parameter vision-language model
   - Model: `Qwen/Qwen2.5-VL-3B-Instruct`
   - Status: Frozen during fusion training

### Training Details

**Training Data:**
- FineBio laboratory video dataset
- ~2000 training videos, ~500 validation videos
- 16 frames per video, 224x224 resolution

**Hyperparameters:**
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Scheduler: Cosine with warmup (500 steps)
- Batch size: 1 (effective: 32 with gradient accumulation)
- Precision: bfloat16 mixed precision
- Epochs: 20
- Loss: Cross-entropy + Contrastive loss (alpha=0.1)


## Performance

### Evaluation Metrics

Evaluated on test set (520 videos):

| Metric | Baseline (Qwen) | Deep Fusion | Improvement |
|--------|-----------------|-------------|-------------|
| BLEU-1 | 0.0127 | 0.0914 | +621.5% |
| ROUGE-1 | 0.0336 | 0.1499 | +346.5% |
| METEOR | 0.0490 | 0.3521 | +619.2% |
| BERTScore-F1 | 0.8183 | 0.8552 | +4.5% |

### Comparison with Baselines

- **Qwen-only baseline**: Uses only Qwen2.5-VL-3B-Instruct with video frames
- **Prompt-based**: Injects VideoMAE predictions as text prompts
- **Deep Fusion**: This model (best performance)

## Intended Use

### Primary Use Cases

1. **Laboratory Documentation**: Automatic generation of protocol descriptions from video recordings
2. **Training Material**: Creating textual summaries of experimental procedures
3. **Quality Control**: Verification of performed actions in lab workflows


## Ethical Considerations

### Bias

- Dataset limited to specific laboratory protocols
- May not represent diverse laboratory practices across different institutions
- Potential bias towards common actions in training data

### Environmental Impact

- Training requires significant computational resources

### Privacy

- No personally identifiable information in training data
- Videos recorded in controlled laboratory environment

## Caveats and Recommendations

1. **Human Verification**: Generated descriptions should be reviewed by domain experts
2. **Fine-tuning**: Consider fine-tuning on institution-specific protocols
4. **Updates**: Model performance may degrade on protocols not seen during training

## Model Access

### Inference

```python
from scripts.evaluate_videomae_qwen import load_model, generate_protocol

model = load_model("path/to/checkpoint.ckpt")
description = generate_protocol(video_path="video.mp4", model=model)
```

### Checkpoints

Available at: ?


## Model Card Authors

Annaelle Benlamri, EPFL AI Team

## Model Card Contact

annaelle.benlamri@epfl.ch

## Version History

- **v1.0** (January 2025): Initial release

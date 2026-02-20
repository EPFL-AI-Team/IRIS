# VideoMAE-Qwen Fusion for Laboratory Protocol Documentation

Deep fusion architecture combining finetuned VideoMAE-v2 encoder and Qwen2.5-VL-3B-Instruct decoder for automatic generation of scientific protocol documentation from laboratory videos.

## Model Architecture

```
Video (16 frames) → VideoMAE-v2 (frozen) [1568 tokens × 768]
                  ↓
              Perceiver Resampler (trainable) [64 tokens × 768]
                  ↓
              MLP Projection (trainable) [64 tokens × 3584]
                  ↓
              Qwen2.5-VL (frozen)
                  ↓
              Text Description
```

**Key Components:**
- **VideoMAE-v2**: `AnnaelleMyriam/videomaev2-finetuned-finebio` (frozen, 32 lab actions)
- **Qwen2.5-VL**: `Qwen/Qwen2.5-VL-3B-Instruct` (frozen)
- **Perceiver Resampler**: 64 query tokens, 2 layers, 8 heads (trainable)
- **MLP Projector**: 3 layers, hidden dim 4096 (trainable)

**Training Strategy:**
- VideoMAE and Qwen remain frozen
- Only Perceiver Resampler + MLP are trained
- Uses contrastive loss for video-text alignment
- 20 epochs, lr=1e-4, batch_size=1, gradient accumulation=32

## Repository Structure

```
configs/              Configuration files
scripts/              Main Python scripts (train, evaluate)
outputs/              Model predictions
analysis/             Analysis and visualization tools
```

## Documentation

- [DATASET.md](DATASET.md) - Dataset format and preparation
- [MODEL_CARD.md](MODEL_CARD.md) - Model specifications
- [USAGE.md](USAGE.md) - Usage guide



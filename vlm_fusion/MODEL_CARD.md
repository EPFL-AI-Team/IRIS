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
   - Model: [videomaev2-finetuned-finebio](AnnaelleMyriam/videomaev2-finetuned-finebio)
   - Status: Frozen during fusion training

2. **Perceiver Resampler**
   - Compresses 1568 VideoMAE tokens to 64 tokens
   - 2 layers, 8 attention heads
   - Learnable query tokens
   - Status: Trainable

3. **MLP Projector**
   - Projects VideoMAE embeddings (768D) to Qwen space (2048D)
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

### Comparison with Baselines

- **Qwen-only baseline**: Uses only Qwen2.5-VL-3B-Instruct with video frames
- **Prompt-based**: Injects VideoMAE predictions as text prompts
- **Deep Fusion**: This model


For detailed results and analysis, please refer to the full report:  
[Annaelle-Benlamri-IRIS-VLM-Report.pdf](Annaelle-Benlamri-IRIS-VLM-Report.pdf)


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


### Checkpoints

Available at: 
- [videomaev2-finetuned-finebio](AnnaelleMyriam/videomaev2-finetuned-finebio)
- [videomae-qwen-connectors](AnnaelleMyriam/videomae-qwen-connectors)


## Model Card Authors

Annaelle Myriam Benlamri, member of the EPFL AI Team


## Version History

- **v1.0** (January 2025): Initial release

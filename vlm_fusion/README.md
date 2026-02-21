# Action recognition and Multimodal fusion for Laboratory Protocol Documentation

Investigated specialized video action recognition models and two strategies for integrating them with a VLM, evaluated on the [FineBio Dataset](https://github.com/aistairc/FineBio).

- **Backbone**: VideoMAE V2 (ViT-Base distilled from ViT-Giant, https://huggingface.co/OpenGVLab/VideoMAE2/tree/main/distill), pretrained on 1.35M unlabeled clips and fine-tuned on the processed FineBio dataset.
- **Fusion strategy 1, Prompt injection**: Top-k predicted actions from VideoMAE V2 are formatted as structured context and injected into Qwen2.5-VL's prompt
- **Fusion strategy 2, Deep fusion**: VideoMAE V2 spatiotemporal tokens are compressed via a Perceiver Resampler and projected directly into Qwen2.5-VL's embedding space via a trainable MLP, with both backbones frozen

- **Report**: [`Annaelle-Benlamri-IRIS-VLM-Report.pdf`](Ahttps://github.com/EPFL-AI-Team/IRIS/blob/main/Annaelle-Benlamri-IRIS-VLM-Report.pdf)
- **Code**: [`vlm_fusion/`](vlm_fusion/)


## Strategy 1 : Prompt Injection


Prompt-based injection treats VideoMAE-v2 as an external action recognizer whose predictions guide the vision-language model (Qwen2.5-VL-3B-Instruct) through textual prompts.

### Approach

- VideoMAE-v2 produces action class probabilities: `p(a|x) = softmax(g(Zv))`
- Top-k predicted actions are extracted with their confidence scores
- Predictions are formatted as structured text and injected into the VLM prompt
- The VLM generates protocol documentation conditioned on these action predictions

### Prompt Template

```
SYSTEM: You are an assistant documenting laboratory procedures based on video analysis.

USER:
Segment metadata: start: <t_start>s, end: <t_end>s, duration: <dur>s

Action recognizer (VideoMAE-v2) predictions (top-k):
1) <action_1> (confidence: <p_1>%)
2) <action_2> (confidence: <p_2>%)
...

Task: Based on the action recognizer predictions, produce a structured description
of the laboratory step. Output a JSON object with fields:
(step_id, time, action, manipulated_object, affected_object, notes).

If the evidence is insufficient, inconsistent, or the confidence scores are low,
output "uncertain": true and explain why in notes.

Respond ONLY with the JSON object.
```


## Strategy 2 : Deep Fusion

Deep fusion architecture combining finetuned VideoMAE-v2 encoder and Qwen2.5-VL-3B-Instruct decoder for automatic generation of scientific protocol documentation from laboratory videos.

### Model Architecture

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


## Checkpoints

Available at:
- [videomaev2-finetuned-finebio](https://huggingface.co/AnnaelleMyriam/videomaev2-finetuned-finebio)
- [videomae-qwen-connectors](https://huggingface.co/AnnaelleMyriam/videomae-qwen-connectors)

## Repository Structure

```
configs/              Configuration files
notebooks/            Data processing pipeline (FinBio)
scripts/              Main Python scripts 
outputs/              Model predictions
```

## Documentation

- [DATASET.md](DATASET.md) - Dataset format and preparation (FineBio)
- [MODEL_CARD.md](MODEL_CARD.md) - Model (VideoMAE-Qwen Fusion) specifications


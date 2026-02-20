# Action Recognition

VideoMAE V2-based action recognition and multimodal fusion with Qwen2.5-VL, developed by Annaelle Myriam Benlamri.

**Report**: [`reports/Annaelle-Benlamri-IRIS-Action-Recognition-Report.pdf`](../reports/Annaelle-Benlamri-IRIS-Action-Recognition-Report.pdf)

## Approach

Two strategies were explored for integrating a specialized video action recognizer with Qwen2.5-VL:

- **Prompt injection**: top-k predictions from VideoMAE V2 are formatted as structured context and added to the VLM prompt
- **Deep latent fusion**: spatiotemporal tokens from VideoMAE V2 are compressed via a Perceiver Resampler and projected into Qwen2.5-VL's embedding space via a trainable MLP

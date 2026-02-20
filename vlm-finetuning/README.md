# VLM Fine-tuning

Fine-tuned Qwen2.5-VL (3B) on the FineBio dataset for laboratory action recognition.

**Model on HuggingFace**: [animarcus/iris-qwen2.5-vl-3b-finebio](https://huggingface.co/animarcus/iris-qwen2.5-vl-3b-finebio)

## Approach

- **Dataset**: FineBio — 9K stratified train samples drawn from 50K total atomic operations, split at participant level to avoid visual leakage
- **Method**: LoRA (r=16, alpha=32) on Qwen2.5-VL (3B)
- **Output format**: Structured JSON `{verb, tool, target, context}`

## Code

Training and evaluation live in [`src/iris/vlm/`](../src/iris/vlm/) and [`src/iris/dataset/`](../src/iris/dataset/). CLI entry points are in [`src/iris/cli/finetune/`](../src/iris/cli/finetune/).

See the [training guide](../docs/rcp-guide.md) for how to run fine-tuning on EPFL clusters.

## Results

The fine-tuned model improved substantially on in-distribution FineBio test samples compared to the base model, but showed vocabulary overfitting when tested on out-of-distribution colony counting videos. The base model's generalization proved more robust in that setting. See the [technical report](../reports/Marcus-Hamelink-IRIS-VLM-Report.pdf) for full evaluation and confusion matrices.

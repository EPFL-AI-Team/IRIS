# VLM Fine-tuning

Fine-tuned Qwen2.5-VL (3B) on FineBio dataset for laboratory action recognition.

## What I Did

- **Dataset**: FineBio (9K train samples from 50K total, stratified)
- **Method**: LoRA fine-tuning (r=16, α=32)
- **Output**: Structured JSON with `{verb, tool, target, context}`

**Model**: [animarcus/iris-qwen2.5-vl-3b-finebio](https://huggingface.co/animarcus/iris-qwen2.5-vl-3b-finebio)

---

## Code

All training/evaluation code is in [`src/iris/cli/finetune/`](../src/iris/cli/finetune/):
- `train.py` - Training script
- `evaluate.py` - Test set evaluation
- `inference.py` - Inference on videos

See [training guide](../docs/rcp-guide.md) for detailed usage.

---

## Key Lesson Learned

**Always test your baseline first.** 

The base Qwen2.5-VL already handles JSON output well - fine-tuning taught it laboratory vocabulary but overfit to specific procedures. On out-of-distribution videos (colony counting), the base model performed better than the fine-tuned version.

**Takeaway**: Measure a model's existing capabilities before investing compute in fine-tuning. Sometimes it's not needed.

---

## Results

See [full technical report](../reports/IRIS-report-IN-BA5-Marcus-Hamelink.pdf) for evaluation details, confusion matrices, and analysis.

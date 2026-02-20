# SFT VLM Fine-tuning

LoRA fine-tuning of Qwen2.5-VL (3B) on the FineBio dataset for laboratory action recognition.

- **Model on HuggingFace**: [animarcus/iris-qwen2.5-vl-3b-finebio](https://huggingface.co/animarcus/iris-qwen2.5-vl-3b-finebio)
- **Report**: [`IRIS-report-Marcus-Hamelink-IN-BA5.pdf`](../IRIS-report-Marcus-Hamelink-IN-BA5.pdf)

## Structure

```
dataset/    Dataset preparation — FineBio annotations → JSONL training splits
vlm/        Model utilities — config, trainer, data collator, LoRA setup
scripts/    CLI scripts — training, evaluation, inference
```

## Setup

Dependencies are managed from the repo root:
```bash
cd ..                  # repo root
uv sync
source .venv/bin/activate
```

Then return here and run commands with `python -m ...` as shown below.

## Quick start

All commands run from inside this folder:

```bash
cd sft-vlm-finetune

# 1. Prepare data
python -m dataset.process_dataset
python -m dataset.create_training_data

# 2. Train
python -m scripts.train --config train_a100 --hardware a100

# 3. Evaluate
python -m scripts.evaluate --checkpoint_dir <path> --val_path <path>
```

See [`dataset/README.md`](dataset/README.md) for data prep details and [`scripts/README.md`](scripts/README.md) for training details.

## Results

The fine-tuned model improved substantially on in-distribution FineBio test samples vs. the base model, but showed vocabulary overfitting on out-of-distribution colony counting videos. See the technical report for full evaluation and confusion matrices.

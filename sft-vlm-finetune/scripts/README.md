# Training Scripts

CLI scripts for fine-tuning, evaluating, and running inference with Qwen2.5-VL.

All commands run from the `sft-vlm-finetune/` directory using `python -m scripts.<name>`.

## Training

```bash
python -m scripts.train --config train_a100 --hardware a100 --wandb-project iris-qwen-training
```

Config files live in `../configs/vlm/`. Hardware profiles (`a100`, `v100`, `mac`) override dtype and attention settings. See the [training guide](../docs/rcp-guide.md) for running on EPFL clusters.

## Evaluation

```bash
python -m scripts.evaluate \
  --checkpoint_dir /path/to/checkpoint \
  --val_path /path/to/finebio_test.jsonl \
  --max_samples 0 \
  --compare-base \
  --eval-name my_run
```

`--max_samples 0` evaluates all samples. `--compare-base` also runs the untuned base model for comparison.

Outputs to `evaluation/{eval-name}/`:
- `metrics.json` - exact match, per-field accuracy, token F1
- `predictions.csv` - per-sample results
- `summary.txt`, `errors.txt`, `examples.txt`
- Confusion matrix and accuracy plots

## Inference on video

```bash
python -m scripts.inference \
  --video /path/to/video.mp4 \
  --checkpoint /path/to/checkpoint \
  --output-dir ./results \
  --segment-duration 2.0 \
  --num-frames 4
```

Omit `--checkpoint` to run the base model only. Outputs `{video_name}_base.jsonl` and/or `{video_name}_finetuned.jsonl`.

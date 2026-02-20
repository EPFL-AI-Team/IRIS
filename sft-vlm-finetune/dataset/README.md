# Dataset Preparation

Pipeline for converting raw FineBio annotations and videos into JSONL training splits for Qwen2.5-VL fine-tuning.

## Assumptions

- **FineBio annotations**: tab-separated `.txt` files per video in `finebio_action_annotations/`
- **Videos**: `.mp4` files in `videos/w640/` (640px wide)
- **Participant split** (hardcoded):
  - Train: P01–P31 (excluding val/test participants)
  - Val: P05, P09, P15, P24, P32
  - Test: P03, P08, P13, P20, P28
- **Quotas**: 1000 train samples / verb, 200 val/test samples / verb (~9K total train)
- **Frame format**: 4 frames per segment, sampled from a canonical 16-frame grid (slots 0, 5, 10, 15)

## Configuration

Edit `dataset_config.yaml` to set paths for your environment:

```yaml
default_profile: mac     # or rcp, izar
profiles:
  mac:
    annotations_dir: /path/to/finebio_action_annotations
    videos_dir: /path/to/videos/w640
    output_dir: /path/to/output
```

## Pipeline

Run all commands from the `sft-vlm-finetune/` directory.

### Step 1 — Process annotations

```bash
python -m dataset.process_dataset
```

Reads raw FineBio `.txt` annotation files, filters by duration (0.5–3.0s) and valid verb/object pairs, fills task context, and outputs:
- `{output_dir}/csv_annotations/{video_id}.csv` — per-video filtered annotations
- `{output_dir}/all_annotations.csv` — consolidated annotation table

### Step 2 — Extract frames and generate splits

```bash
python -m dataset.create_training_data
```

Applies participant-based stratified splits, extracts frames from video at canonical slots, and outputs:
- `{output_dir}/frames/{segment_id}/frame_{00,05,10,15}.jpg`
- `{splits_dir}/finebio_train.jsonl`
- `{splits_dir}/finebio_val.jsonl`
- `{splits_dir}/finebio_test.jsonl`

Each JSONL entry follows the Qwen chat format:
```json
{
  "id": "segment_id",
  "messages": [
    {"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "{\"verb\": ..., \"tool\": ..., \"target\": ..., \"context\": ...}"}]}
  ]
}
```

### Step 3 — Validate (recommended before training)

```bash
python -m scripts.debug_data_pipeline
```

CPU-only check that images load correctly, the data collator tokenizes as expected, and labels are masked properly (prompt tokens = -100, response tokens = unmasked).

## Other scripts

- `visualize_sampling.py` — debug visualization of frame sampling strategies across segment durations

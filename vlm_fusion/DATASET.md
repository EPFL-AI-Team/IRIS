# Dataset Documentation

This document describes the processed dataset derived from the [FineBio Dataset](https://github.com/aistairc/FineBio), including its format, structure, and statistics, as used in the VideoMAEv2–Qwen fusion project.

## Overview

The dataset consists of laboratory protocol videos with action annotations (action_label) and textual descriptions (action_name). Videos are decomposed into frame sequences (frames) with corresponding captions describing the experimental procedures (caption).

## Dataset Structure

```
data/
├── annotations/
│   ├── train_annotations.json      # Training set annotations
│   ├── val_annotations.json        # Validation set annotations
│   └── test_annotations.json       # Test set annotations
├── splits/
│   ├── train.csv                   # Training video IDs
│   ├── val.csv                     # Validation video IDs
│   └── test.csv                    # Test video IDs
└── class_mappings/
    └── class_mapping_finebio.json  # Action class mapping for the FineBio dataset (32 classes)
```

## Annotation Format

### JSON Structure

Each annotation file contains a list of video segments with the following format:

```json
[
  {
    "video_path": "/path/to/video/frames/",
    "frames": [
      "frame_0000.jpg",
      "frame_0001.jpg",
      ...
      "frame_0015.jpg"
    ],
    "caption": "Removing cell culture medium from wells.",
    "action_label": 17,
    "action_name": "remove_culture_medium",
    "video_id": "P01_01_01_seg000"
  }
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `video_path` | string | Absolute or relative path to frame directory |
| `frames` | list[string] | List of 16 frame filenames (ordered temporally) |
| `caption` | string | Natural language description of the action/protocol |
| `action_label` | int | Numeric class ID (0-31 for FineBio) |
| `action_name` | string | Human-readable action name |
| `video_id` | string | Unique identifier for the video segment |

## Action Classes

### FineBio Taxonomy (32 classes)

The dataset uses a specialized taxonomy for laboratory actions:

```
0:  add_70pct_ethanol
1:  add_binding_buffer
2:  add_cell_lystate
3:  add_extract
4:  add_magnetic_beads
5:  add_pbs
6:  add_sterile_water
7:  add_wash_buffer
8:  aspirate_pbs
9:  aspirate_supernatant
10: close_8_tube_stripes_lid
11: detach_spin_column_and_insert_to_new_tube
12: dispense_solution
13: dispense_spin_column
14: load_8_tube_stripes_to_pcr_machine
15: pipetting
16: place_in_magnetic_rack
17: remove_culture_medium
18: shake_plate
19: spindown
20: spindown_8_tube_stripes
21: transfer_cell_lystate_to_tube
22: transfer_forward_primer_to_8_tube_stripes
23: transfer_pcrmix_to_8_tube_stripes
24: transfer_reverse_primer_to_8_tube_stripes
25: transfer_sample_to_8_tube_stripes
26: transfer_sample_tube_to_spin_column_tube
27: transfer_supernatant_to_empty_tube
28: transfer_template_dna_to_8_tube_stripes
29: transfer_water_to_8_tube_stripes
30: vortex
31: vortex_8_tube_stripes
```


## Dataset Statistics

### Split Distribution

| Split | Number of Videos | Description |
|-------|------------------|-------------|
| Train | 2371 | Training set |
| Val | 574 | Validation set for hyperparameter tuning |
| Test | 520 | Held-out test set for final evaluation |

### Video Specifications

- **Format**: Sequential frames (JPG)
- **Frames per video**: 16 frames
- **Frame resolution**: Variable (typically 224x224)
- **Temporal sampling**: Uniform sampling from video segments
- **Duration**: Variable segment length (to have 1 action per segment)

### Caption Statistics

- **Average caption length**: ~10-15 words
- **Vocabulary size**: Domain-specific laboratory terminology
- **Language**: English
- **Style**: Imperative/descriptive sentences

## Data Preprocessing

### Frame Extraction

Videos are preprocessed into frame sequences:

```bash
python scripts/create_lab_annotations.py \
    --video_dir /path/to/videos \
    --output_dir data/annotations
```

### Frame Processing

During training, frames are processed using `VideoMAEImageProcessor`:
- Resize to 224x224
- Normalize with ImageNet statistics
- Temporal sampling: 16 frames uniformly sampled

### Text Processing

Captions are tokenized using Qwen tokenizer:
- Max length: 256 tokens
- Instruction format: Prompt + Caption
- Teacher forcing: Only caption tokens have loss

## Usage

### Loading Annotations

```python
import json

with open('data/annotations/train_annotations.json', 'r') as f:
    annotations = json.load(f)

for sample in annotations:
    video_path = sample['video_path']
    frames = sample['frames']
    caption = sample['caption']
    # Process sample...
```

### Custom Dataset Class

See `scripts/train_videomae_qwen_contrastive_new_resampler.py` for the `VideoTextDataset` implementation.

## Data Availability

### Source

The dataset is based on:
- **FineBio**: Laboratory action recognition dataset https://github.com/aistairc/FineBio
- **Custom annotations**: Textual descriptions for protocol documentation

### Access

Contact the authors or institution for dataset access.

## Known Issues

- Some video segments may have temporal discontinuities
- Caption quality varies (human-annotated)
- Class imbalance: Some actions are more frequent than others

## Updates

- **v1.0** (January 2025): Initial release with 3020 annotated video segments

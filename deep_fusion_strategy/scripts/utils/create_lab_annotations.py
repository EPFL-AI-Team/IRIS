"""
Create detailed annotations for laboratory protocol actions
From CSV files with action labels
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

ACTION_CLASSES = {
    0: 'add_70pct_ethanol',
    1: 'add_binding_buffer',
    2: 'add_cell_lystate',
    3: 'add_extract',
    4: 'add_magnetic_beads',
    5: 'add_pbs',
    6: 'add_sterile_water',
    7: 'add_wash_buffer',
    8: 'aspirate_pbs',
    9: 'aspirate_supernatant',
    10: 'close_8_tube_stripes_lid',
    11: 'detach_spin_column_and_insert_to_new_tube',
    12: 'dispense_solution',
    13: 'dispense_spin_column',
    14: 'load_8_tube_stripes_to_pcr_machine',
    15: 'pipetting',
    16: 'place_in_magnetic_rack',
    17: 'remove_culture_medium',
    18: 'shake_plate',
    19: 'spindown',
    20: 'spindown_8_tube_stripes',
    21: 'transfer_cell_lystate_to_tube',
    22: 'transfer_forward_primer_to_8_tube_stripes',
    23: 'transfer_pcrmix_to_8_tube_stripes',
    24: 'transfer_reverse_primer_to_8_tube_stripes',
    25: 'transfer_sample_to_8_tube_stripes',
    26: 'transfer_sample_tube_to_spin_column_tube',
    27: 'transfer_supernatant_to_empty_tube',
    28: 'transfer_template_dna_to_8_tube_stripes',
    29: 'transfer_water_to_8_tube_stripes',
    30: 'vortex',
    31: 'vortex_8_tube_stripes',
}

# Concise, precise descriptions for each laboratory action
ACTION_DESCRIPTIONS = {
    'add_70pct_ethanol': 'Adding 70% ethanol for DNA purification and washing.',
    'add_binding_buffer': 'Adding binding buffer to enable DNA attachment to silica membrane.',
    'add_cell_lystate': 'Adding cell lysate containing extracted genetic material.',
    'add_extract': 'Adding purified DNA extract to the tube.',
    'add_magnetic_beads': 'Adding magnetic beads suspension for DNA purification.',
    'add_pbs': 'Adding phosphate buffered saline for cell washing.',
    'add_sterile_water': 'Adding sterile nuclease-free water for DNA elution.',
    'add_wash_buffer': 'Adding wash buffer to remove contaminants from DNA sample.',
    'aspirate_pbs': 'Aspirating PBS from cell culture wells.',
    'aspirate_supernatant': 'Aspirating supernatant while preserving the pellet.',
    'close_8_tube_stripes_lid': 'Closing the lid of 8-tube PCR strips.',
    'detach_spin_column_and_insert_to_new_tube': 'Transferring spin column to new collection tube.',
    'dispense_solution': 'Dispensing solution into multiple tubes.',
    'dispense_spin_column': 'Placing spin columns into collection tubes.',
    'load_8_tube_stripes_to_pcr_machine': 'Loading PCR tube strips into thermocycler.',
    'pipetting': 'Pipetting liquid with precision micropipette.',
    'place_in_magnetic_rack': 'Placing tubes in magnetic rack for bead separation.',
    'remove_culture_medium': 'Removing cell culture medium from wells.',
    'shake_plate': 'Shaking plate on orbital shaker for mixing.',
    'spindown': 'Centrifuging tubes to pellet cells or DNA.',
    'spindown_8_tube_stripes': 'Centrifuging PCR tube strips briefly.',
    'transfer_cell_lystate_to_tube': 'Transferring cell lysate to purification tube.',
    'transfer_forward_primer_to_8_tube_stripes': 'Adding forward primer to PCR tubes.',
    'transfer_pcrmix_to_8_tube_stripes': 'Adding PCR master mix to reaction tubes.',
    'transfer_reverse_primer_to_8_tube_stripes': 'Adding reverse primer to PCR tubes.',
    'transfer_sample_to_8_tube_stripes': 'Transferring DNA samples to PCR tubes.',
    'transfer_sample_tube_to_spin_column_tube': 'Loading sample into spin column for purification.',
    'transfer_supernatant_to_empty_tube': 'Transferring supernatant to new tube.',
    'transfer_template_dna_to_8_tube_stripes': 'Adding template DNA to PCR reaction tubes.',
    'transfer_water_to_8_tube_stripes': 'Adding sterile water to PCR tubes.',
    'vortex': 'Vortexing tube to mix solution thoroughly.',
    'vortex_8_tube_stripes': 'Vortexing 8-tube strips for mixing.',
}


def get_action_description(action_name: str) -> str:
    """
    Get concise description for a laboratory action

    Args:
        action_name: Action name (e.g., 'add_70pct_ethanol')

    Returns:
        Concise textual description
    """
    return ACTION_DESCRIPTIONS.get(
        action_name,
        f"Laboratory action: {action_name.replace('_', ' ')}"
    )


def find_video_frames(video_id: str, frames_root: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Find the folder containing video frames

    Args:
        video_id: Video ID (e.g., 'P01_01_01_seg000')
        frames_root: Root folder containing frames

    Returns:
        (video_path, frames_list) or (None, None) if not found
    """
    parts = video_id.split('_seg')
    base_id = parts[0]
    seg_num = parts[1] if len(parts) > 1 else '000'

    possible_paths = [
        os.path.join(frames_root, video_id),
        os.path.join(frames_root, base_id, f"seg{seg_num}"),
        os.path.join(frames_root, base_id.replace('_', '/'), f"seg{seg_num}"),
    ]

    for video_path in possible_paths:
        if os.path.exists(video_path):
            frames = sorted([
                f for f in os.listdir(video_path)
                if f.endswith(('.jpg', '.png', '.jpeg'))
            ])
            if frames:
                return video_path, frames

    # If not found, return default path
    return None, None


def create_annotations_from_csv(
    csv_path: str,
    frames_root: str,
    output_json: str,
    split_name: str = "train"
):
    """
    Create JSON annotations from CSV file

    Args:
        csv_path: Path to CSV file (e.g., train.csv)
        frames_root: Root folder containing video frames
        output_json: Output path for JSON file
        split_name: Split name (train/val/test)
    """
    print(f"Creating annotations for {split_name}")

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, sep=' ', header=None, names=['video_id', 'num_frames', 'label'])

    print(f"Found {len(df)} videos in CSV")

    annotations = []
    missing_videos = []

    for idx, row in df.iterrows():
        video_id = row['video_id']
        num_frames = int(row['num_frames'])
        label = int(row['label'])

        action_name = ACTION_CLASSES.get(label, 'unknown_action')
        video_path, frames_list = find_video_frames(video_id, frames_root)

        if video_path is None:
            missing_videos.append(video_id)
            video_path = os.path.join(frames_root, video_id)
            frames_list = [f"frame_{i:06d}.jpg" for i in range(num_frames)]

        caption = get_action_description(action_name)

        annotation = {
            'video_path': video_path,
            'frames': frames_list[:num_frames],
            'caption': caption,
            'action_label': int(label),
            'action_name': action_name,
            'video_id': video_id,
        }

        annotations.append(annotation)


    print(f"\nSaving {len(annotations)} annotations to {output_json}...")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"Annotations saved: {output_json}")

    # Statistics
    print(f"Statistics for {split_name}:")
    print(f"Total annotations: {len(annotations)}")
    print(f"Missing videos: {len(missing_videos)}")

    if missing_videos:
        print(f"\nFirst missing videos:")
        for vid in missing_videos[:5]:
            print(f"  - {vid}")
        if len(missing_videos) > 5:
            print(f"  ... and {len(missing_videos) - 5} more")

    # Action distribution
    action_counts = df['label'].value_counts().sort_index()
    print(f"\nAction distribution:")
    for label, count in action_counts.head(10).items():
        action_name = ACTION_CLASSES.get(label, 'unknown')
        print(f"  {label:2d} - {action_name:45s}: {count:4d} videos")

    if len(action_counts) > 10:
        print(f"  ... and {len(action_counts) - 10} more actions")

    return annotations


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create annotations for laboratory protocol actions"
    )
    parser.add_argument(
        '--csv-dir',
        required=True,
        help='Directory containing train.csv, val.csv, test.csv'
    )
    parser.add_argument(
        '--frames-root',
        required=True,
        help='Root directory containing video frames'
    )
    parser.add_argument(
        '--output-dir',
        default='./data',
        help='Output directory for JSON files'
    )

    args = parser.parse_args()

    # Create annotations for each split
    splits = ['train', 'val', 'test']

    for split in splits:
        csv_path = os.path.join(args.csv_dir, f'{split}.csv')

        if not os.path.exists(csv_path):
            print(f" CSV not found: {csv_path}, skipping")
            continue

        output_json = os.path.join(args.output_dir, f'{split}_annotations.json')

        create_annotations_from_csv(
            csv_path=csv_path,
            frames_root=args.frames_root,
            output_json=output_json,
            split_name=split
        )

    print("Annotation creation completed!")
    print(f"\nFiles created in: {args.output_dir}")
    print("  - train_annotations.json")
    print("  - val_annotations.json")
    print("  - test_annotations.json")


if __name__ == '__main__':
    main()

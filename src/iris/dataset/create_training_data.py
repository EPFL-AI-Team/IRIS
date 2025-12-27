import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dataset_config import (
    configure_logging,
    ensure_output_dirs,
    load_dataset_config,
    resolve_paths,
)

from iris.dataset.dataset_utils import filter_data
from iris.dataset.training_format import (
    chat_jsonl_entry,
    expected_output_json,
    pick_prompt,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test JSONL splits")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("dataset_config.yaml"),
        help="Path to dataset_config.yaml (defaults to scripts/dataset/dataset_config.yaml).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile name (e.g. mac, rcp, izar). Defaults to config's default_profile.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Split output subfolder name. Defaults to config's default_split_name.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


TOTAL_SAMPLES_PER_VERB = 1000
# VIDEO TO HOLD OUT (For Demo)
# P25_02_01 = Participant 25, Protocol 2, take 1
HOLDOUT_VIDEO_IDS = ["P25_02_01"]


def get_stratified_splits(
    annotations_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data using the OFFICIAL FineBio participant splits (Subject-Independent).
    Ref: FineBio-Dataset-essentials.pdf (Section C.1 Dataset split details)
    """
    logger.info("Splitting by Official Participant IDs...")

    # Extract Participant ID (PXX from PXX_YY_ZZ)
    annotations_df = annotations_df.copy()
    annotations_df["participant_id"] = annotations_df["video_id"].apply(
        lambda x: x.split("_")[0]
    )

    # Official Splits
    TRAIN_IDS = {
        "P01",
        "P02",
        "P04",
        "P06",
        "P07",
        "P10",
        "P11",
        "P12",
        "P14",
        "P16",
        "P17",
        "P18",
        "P19",
        "P21",
        "P22",
        "P23",
        "P25",
        "P26",
        "P27",
        "P29",
        "P30",
        "P31",
    }
    VAL_IDS = {"P05", "P09", "P15", "P24", "P32"}
    TEST_IDS = {"P03", "P08", "P13", "P20", "P28"}

    # Filter into pools
    train_pool = annotations_df[annotations_df["participant_id"].isin(TRAIN_IDS)].copy()
    val_df = annotations_df[annotations_df["participant_id"].isin(VAL_IDS)].copy()
    test_df = annotations_df[annotations_df["participant_id"].isin(TEST_IDS)].copy()

    # Sanity Check
    total_assigned = len(train_pool) + len(val_df) + len(test_df)
    if total_assigned != len(annotations_df):
        logger.warning(
            f"Dropping {len(annotations_df) - total_assigned} samples with unknown IDs."
        )

    # --- DOWNSAMPLING (TRAIN ONLY) ---
    logger.info(
        f"Downsampling Training Set (Target: {TOTAL_SAMPLES_PER_VERB} per verb)..."
    )
    train_downsampled_dfs = []
    all_verbs = train_pool["verb"].unique()
    for verb in all_verbs:
        v_df = train_pool[train_pool["verb"] == verb]
        count = len(v_df)
        if count <= TOTAL_SAMPLES_PER_VERB:
            train_downsampled_dfs.append(v_df)
        else:
            tool_counts = v_df["manipulated_object"].value_counts()
            weights = 1.0 / v_df["manipulated_object"].map(tool_counts)
            selected = v_df.sample(
                n=TOTAL_SAMPLES_PER_VERB, weights=weights, random_state=42
            )
            train_downsampled_dfs.append(selected)
    train_final = pd.concat(train_downsampled_dfs)

    logger.info("--- Official Split Statistics ---")
    logger.info(f"Train: {len(train_final)} (Downsampled from {len(train_pool)})")
    logger.info(f"Val:   {len(val_df)} (Natural Distribution)")
    logger.info(f"Test:  {len(test_df)} (Natural Distribution)")

    return train_final, val_df, test_df


def create_jsonl(
    annotations_df: pd.DataFrame,
    *,
    frame_base_dir: Path,
    output_path: Path,
    canonical_max_frames: int,
    frames_per_segment: int,
) -> None:
    """Generate a JSONL file for a given DataFrame."""
    entries: list[str] = []

    missing_files_total = 0
    missing_segments_total = 0
    warned_segments = 0
    max_segment_warnings = 25

    if canonical_max_frames <= 0:
        raise ValueError("canonical_max_frames must be > 0")
    if frames_per_segment <= 0:
        raise ValueError("frames_per_segment must be > 0")
    if frames_per_segment > canonical_max_frames:
        raise ValueError("frames_per_segment cannot exceed canonical_max_frames")

    # Canonical-slot strategy (matches process_dataset.py):
    # Example: canonical_max_frames=16, frames_per_segment=4 -> [0, 5, 10, 15]
    frame_slots = np.linspace(
        0, canonical_max_frames - 1, frames_per_segment, dtype=int
    )
    frame_slots = np.unique(frame_slots)

    for _, row in annotations_df.iterrows():
        segment_id = f"{row['video_id']}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"
        segment_dir = frame_base_dir / segment_id

        # Build absolute paths for frames
        image_paths = [str(segment_dir / f"frame_{i:02d}.jpg") for i in frame_slots]

        # Quick sanity logging: warn when expected frames are missing on disk.
        # This usually means process_dataset.py wasn't run with --extract-frames,
        # or the frames YAML settings differ from what was used during extraction.
        missing_paths = [p for p in image_paths if not Path(p).exists()]
        if missing_paths:
            missing_segments_total += 1
            missing_files_total += len(missing_paths)

            if warned_segments < max_segment_warnings:
                warned_segments += 1
                shown = missing_paths[:8]
                extra = len(missing_paths) - len(shown)
                suffix = f" (+{extra} more)" if extra > 0 else ""
                logger.warning(
                    "Missing %d/%d frame files for segment_id=%s (dir=%s). Missing: %s%s. "
                    "Config: per_segment=%d canonical=%d. Fix: rerun process_dataset.py --extract-frames with the same config.",
                    len(missing_paths),
                    len(image_paths),
                    segment_id,
                    segment_dir,
                    shown,
                    suffix,
                    frames_per_segment,
                    canonical_max_frames,
                )

        prompt = pick_prompt()
        response_json = expected_output_json(row)
        entries.append(
            chat_jsonl_entry(
                entry_id=segment_id,
                image_paths=image_paths,
                prompt=prompt,
                expected_json=response_json,
            )
        )

    # Shuffle final list to mix verbs together
    random.shuffle(entries)

    with open(output_path, "w") as f:
        f.write("\n".join(entries))
    logger.info("Saved %d samples to %s", len(entries), output_path)

    if missing_segments_total > 0:
        logger.warning(
            "Frame check: %d missing files across %d/%d segments (warnings shown=%d/%d).",
            missing_files_total,
            missing_segments_total,
            len(annotations_df),
            warned_segments,
            max_segment_warnings,
        )


if __name__ == "__main__":
    args = parse_arguments()
    configure_logging(args.log_level)

    config = load_dataset_config(args.config)
    resolved, effective_split_name = resolve_paths(
        config, profile_name=args.profile, split_name=args.split_name
    )
    ensure_output_dirs(resolved)

    input_csv = resolved.consolidated_csv
    frame_base_dir = resolved.frames_dir
    out_dir = resolved.splits_dir

    out_train = out_dir / "finebio_train.jsonl"
    out_val = out_dir / "finebio_val.jsonl"
    out_test = out_dir / "finebio_test.jsonl"

    logger.info("Dataset profile: %s", resolved.profile.name)
    logger.info("Using split_name: %s", effective_split_name)
    logger.info("Input CSV: %s", input_csv)
    logger.info("Frame dir: %s", frame_base_dir)
    logger.info("Split output dir: %s", out_dir)
    logger.info(
        "Frames: per_segment=%d (canonical grid=%d)",
        config.frames_per_segment,
        config.canonical_max_frames,
    )

    # Filtering parameters from config (add these to your config if not present)
    min_duration = getattr(config, "min_duration", 0.5)
    max_duration = getattr(config, "max_duration", 3.0)

    # Load
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing consolidated CSV: {input_csv}. Run process_dataset.py first (or update output_dir)."
        )
    annotations_df = pd.read_csv(input_csv)

    # Quality filter (duration/object)
    filtered_df = filter_data(annotations_df, min_duration, max_duration)

    # Official split
    train_df, val_df, test_df = get_stratified_splits(filtered_df)

    logger.info("--- Final File Statistics ---")
    logger.info("Train: %d samples", len(train_df))
    logger.info("Val:   %d samples", len(val_df))
    logger.info("Test:  %d samples", len(test_df))

    # Generate Files
    create_jsonl(
        train_df,
        frame_base_dir=frame_base_dir,
        output_path=out_train,
        canonical_max_frames=config.canonical_max_frames,
        frames_per_segment=config.frames_per_segment,
    )
    create_jsonl(
        val_df,
        frame_base_dir=frame_base_dir,
        output_path=out_val,
        canonical_max_frames=config.canonical_max_frames,
        frames_per_segment=config.frames_per_segment,
    )
    create_jsonl(
        test_df,
        frame_base_dir=frame_base_dir,
        output_path=out_test,
        canonical_max_frames=config.canonical_max_frames,
        frames_per_segment=config.frames_per_segment,
    )

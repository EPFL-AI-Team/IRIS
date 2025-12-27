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
from sklearn.model_selection import train_test_split
from training_format import chat_jsonl_entry, expected_output_json, pick_prompt

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


# Parameters
TOTAL_SAMPLES_PER_VERB = 1000
NUM_FRAMES_TO_SAMPLE = 4  # Default; overridden from dataset_config.yaml at runtime.
MIN_DURATION = 0.5
MAX_DURATION = 3.0

# VIDEO TO HOLD OUT (For Demo)
# P25_02_01 = Participant 25, Protocol 2, take 1
HOLDOUT_VIDEO_IDS = ["P25_02_01"]


def filter_data(annotations_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply quality filters AND separate holdout videos.

    Returns:
        (filtered_training_pool, holdout_demo_pool)
    """
    logger.info("Initial pool: %d", len(annotations_df))
    logger.info(
        "Filter parameters: NUM_FRAMES=%d, MIN_DURATION=%.1f, MAX_DURATION=%.1f",
        NUM_FRAMES_TO_SAMPLE,
        MIN_DURATION,
        MAX_DURATION,
    )

    # 1. Duration & Object Filters
    annotations_df["duration"] = annotations_df["end_sec"] - annotations_df["start_sec"]

    # Log rows with duration issues
    too_short = annotations_df[annotations_df["duration"] < MIN_DURATION]
    too_long = annotations_df[annotations_df["duration"] > MAX_DURATION]

    if len(too_short) > 0:
        logger.info(
            "Rows with duration < MIN_DURATION (%.1f): %d", MIN_DURATION, len(too_short)
        )
        logger.debug(
            "Too short rows:\n%s",
            too_short[
                ["video_id", "start_sec", "end_sec", "duration", "verb"]
            ].to_string(),
        )

    if len(too_long) > 0:
        logger.info(
            "Rows with duration > MAX_DURATION (%.1f): %d", MAX_DURATION, len(too_long)
        )
        logger.debug(
            "Too long rows:\n%s",
            too_long[
                ["video_id", "start_sec", "end_sec", "duration", "verb"]
            ].to_string(),
        )

    valid_df = annotations_df[
        (annotations_df["duration"] >= MIN_DURATION)
        & (annotations_df["duration"] <= MAX_DURATION)
        & (annotations_df["manipulated_object"].notna())
        & (annotations_df["manipulated_object"] != "nan")
    ].copy()

    logger.info("Pool after quality filtering: %d", len(valid_df))

    # 2. Holdout Logic
    # Identify rows belonging to holdout videos
    is_holdout = valid_df["video_id"].isin(HOLDOUT_VIDEO_IDS)

    demo_df = valid_df[is_holdout].copy()
    train_pool_df = valid_df[~is_holdout].copy()

    if len(demo_df) > 0:
        logger.info(f"Held out {len(demo_df)} samples from videos: {HOLDOUT_VIDEO_IDS}")
    else:
        logger.warning(f"No samples found for holdout videos: {HOLDOUT_VIDEO_IDS}")

    return train_pool_df, demo_df


def get_stratified_splits(
    annotations_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create per-verb 80/10/10 train/val/test splits with downsampling."""
    train_dfs = []
    val_dfs = []
    test_dfs = []

    all_verbs = annotations_df["verb"].unique()

    logger.info("Processing splits (target per verb: %d)...", TOTAL_SAMPLES_PER_VERB)

    for verb in all_verbs:
        v_df = annotations_df[annotations_df["verb"] == verb]
        count = len(v_df)

        if count <= TOTAL_SAMPLES_PER_VERB:
            # Take everything (Rare case)
            selected_df = v_df
            logger.info("[%s] Rare: keeping all %d", verb, count)
        else:
            # Weighted downsampling: weight = 1 / tool frequency
            tool_counts = v_df["manipulated_object"].value_counts()
            weights = 1.0 / v_df["manipulated_object"].map(tool_counts)

            selected_df = v_df.sample(
                n=TOTAL_SAMPLES_PER_VERB, weights=weights, random_state=42
            )
            logger.info(
                "[%s] Common: downsampled %d -> %d (weighted)",
                verb,
                count,
                TOTAL_SAMPLES_PER_VERB,
            )

        # 80/10/10 split
        train, temp = train_test_split(
            selected_df, test_size=0.2, random_state=42, shuffle=True
        )

        val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)


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


if __name__ == "__main__":
    args = parse_arguments()
    configure_logging(args.log_level)

    config = load_dataset_config(args.config)
    resolved, effective_split_name = resolve_paths(
        config, profile_name=args.profile, split_name=args.split_name
    )
    ensure_output_dirs(resolved)

    # Keep producer/consumer aligned via dataset_config.yaml
    NUM_FRAMES_TO_SAMPLE = config.frames_per_segment

    input_csv = resolved.consolidated_csv
    frame_base_dir = resolved.frames_dir
    out_dir = resolved.splits_dir

    out_train = out_dir / "finebio_train.jsonl"
    out_val = out_dir / "finebio_val.jsonl"
    out_test = out_dir / "finebio_test.jsonl"
    out_demo = out_dir / "finebio_demo.jsonl"  # New output file

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

    # Load
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing consolidated CSV: {input_csv}. Run process_dataset.py first (or update output_dir)."
        )
    annotations_df = pd.read_csv(input_csv)

    # Filter & Holdout
    train_pool_df, demo_df = filter_data(annotations_df)

    # Split the main pool
    train_df, val_df, test_df = get_stratified_splits(train_pool_df)

    logger.info("--- Final File Statistics ---")
    logger.info("Train: %d samples", len(train_df))
    logger.info("Val:   %d samples", len(val_df))
    logger.info("Test:  %d samples", len(test_df))
    logger.info("Demo:  %d samples (from %s)", len(demo_df), HOLDOUT_VIDEO_IDS)

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

    # Save Demo Set
    if not demo_df.empty:
        create_jsonl(
            demo_df,
            frame_base_dir=frame_base_dir,
            output_path=out_demo,
            canonical_max_frames=config.canonical_max_frames,
            frames_per_segment=config.frames_per_segment,
        )

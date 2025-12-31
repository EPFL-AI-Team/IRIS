import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dataset_config import (
    DatasetPaths,
    configure_logging,
    ensure_output_dirs,
    load_dataset_config,
    resolve_paths,
    validate_inputs,
)
from tqdm import tqdm

from iris.dataset.logic import _target_frame_slots, fill_task_column, is_valid_action

logger = logging.getLogger(__name__)

# Configuration & args


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process FineBio dataset: Annotations & Frames"
    )
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
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    # Worker count is internal; not exposed on CLI for this script.
    return parser.parse_args()


def _build_paths(profile_paths: DatasetPaths) -> dict[str, Path]:
    return {
        "videos": profile_paths.profile.videos_dir,
        "annotations": profile_paths.profile.annotations_dir,
        "output_frames": profile_paths.frames_dir,
        "csv_out": profile_paths.csv_per_video_dir,
        "consolidated_csv": profile_paths.consolidated_csv,
    }


# Helpers


# Task/slot logic lives in iris.dataset.logic; frame extraction moved to create_training_data.


# Frame extraction logic moved to create_training_data.py per "filter early, extract late".


# Worker


def process_video_annotations(
    video_id: str,
    txt_path: Path,
    paths: dict[str, Path],
    canonical_max_frames: int,
    frames_per_segment: int,
    min_duration: float = 0.5,
    max_duration: float = 3.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    1. Loads raw txt.
    2. Processes/Fills tasks.
    3. Generates CSV.
    4. (Optional) Extracts frames.
    """
    try:
        raw_df = pd.read_csv(txt_path)
    except Exception:
        return [], []

    # Process Data
    processed_df = fill_task_column(raw_df)
    processed_df["video_id"] = video_id

    # Early filter: drop invalid actions now so the consolidated CSV is clean.
    processed_df = processed_df[
        processed_df.apply(
            lambda r: is_valid_action(r, min_duration, max_duration), axis=1
        )
    ]

    # Clean up columns (we want: video_id, start, end, verb, tool, target, task)
    # Mapping dataset columns to standard names
    # Dataset: verb, manipulated_object, affected_object, context_task

    final_rows: list[dict[str, Any]] = []
    target_slots = _target_frame_slots(
        canonical_max_frames=canonical_max_frames,
        frames_per_segment=frames_per_segment,
    ).tolist()

    for _, row in processed_df.iterrows():
        # Data Row
        segment_id = f"{video_id}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"

        item = {
            "video_id": video_id,
            "segment_id": segment_id,
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "verb": str(row.get("verb", "unknown")),
            "manipulated_object": str(row.get("manipulated_object", "none")),
            "affected_object": str(row.get("affected_object", "none")),
            "task_step": str(row.get("context_task", "unknown")),
            "hand": str(row.get("hand_side", "unknown")),
            # CSV metadata to keep downstream consumers aligned with YAML-driven frame policy.
            "frame_slots": str(target_slots),
        }
        final_rows.append(item)

        # Frame extraction is deferred to create_training_data.py (just-in-time).

    # Save Individual Files
    if final_rows:
        # CSV
        df_out = pd.DataFrame(final_rows)
        csv_path = paths["csv_out"] / f"{video_id}.csv"
        df_out.to_csv(csv_path, index=False)

    return final_rows, []


# Main


def main() -> None:
    args = parse_arguments()

    # Frame extraction has been removed from this script (extract late in create_training_data)

    configure_logging(args.log_level)

    config = load_dataset_config(args.config)
    resolved, _split_name = resolve_paths(
        config, profile_name=args.profile, split_name=None
    )
    validate_inputs(resolved)
    ensure_output_dirs(resolved)

    paths = _build_paths(resolved)

    logger.info("Dataset profile: %s", resolved.profile.name)
    logger.info("annotations_dir: %s", resolved.profile.annotations_dir)
    logger.info("videos_dir: %s", resolved.profile.videos_dir)
    logger.info("output_dir: %s", resolved.profile.output_dir)
    logger.info(
        "Frames: per_segment=%d (canonical grid=%d)",
        config.frames_per_segment,
        config.canonical_max_frames,
    )

    # Create Directories
    for k, p in paths.items():
        if k not in ["videos", "annotations", "consolidated_csv"]:
            p.mkdir(parents=True, exist_ok=True)

    # internal worker count for parallel processing (not part of CLI)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    # List all txt files
    txt_files = list(paths["annotations"].glob("*.txt"))
    if not txt_files:
        logger.warning("No annotation files found in %s", paths["annotations"])
        return

    logger.info("Found %d annotation files.", len(txt_files))

    all_csv_rows = []
    start_time = time.perf_counter()
    ok_count = 0
    err_count = 0

    # Parallel Processing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit tasks (pass min/max duration so worker can filter consistently)
        min_duration = getattr(config, "min_duration", 0.5)
        max_duration = getattr(config, "max_duration", 3.0)
        future_map = {
            executor.submit(
                process_video_annotations,
                f.stem,
                f,
                paths,
                config.canonical_max_frames,
                config.frames_per_segment,
                min_duration,
                max_duration,
            ): f.stem
            for f in txt_files
        }

        for future in tqdm(
            as_completed(future_map), total=len(txt_files), desc="Processing Videos"
        ):
            vid_id = future_map[future]
            try:
                rows, _ = future.result()
                if rows:
                    all_csv_rows.extend(rows)
                ok_count += 1
            except Exception as e:
                err_count += 1
                logger.warning("Error processing %s: %s", vid_id, e)

            done = ok_count + err_count
            if done % 50 == 0:
                elapsed = time.perf_counter() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Progress: %d/%d videos (ok=%d, err=%d, %.2f vids/s)",
                    done,
                    len(txt_files),
                    ok_count,
                    err_count,
                    rate,
                )

    # --- CONSOLIDATION & METRICS ---

    logger.info("Generating consolidated files...")

    # 1. Consolidated CSV
    full_df = pd.DataFrame(all_csv_rows)
    if not full_df.empty:
        full_df.to_csv(paths["consolidated_csv"], index=False)
        logger.info("Saved consolidated CSV to: %s", paths["consolidated_csv"])

        # 2. Metrics
        logger.info("Total Atomic Actions: %d", len(full_df))
        logger.info("Total Videos Processed: %d", full_df["video_id"].nunique())

        # Top Distributions
        def print_dist(col_name: str, top_n: int = 10) -> None:
            # Full distribution (not truncated)
            full_counts = full_df[col_name].value_counts(dropna=False)
            distinct = full_counts.size

            # Top-N for display
            counts = full_counts.head(top_n)
            shown = len(counts)

            logger.info("Top %d %s (out of %d distinct):", shown, col_name, distinct)
            total = 0
            for name, count in counts.items():
                logger.info("  - %s: %s", name, count)
                total += count
            logger.info("Total in top %d: %d", shown, total)

        print_dist("verb")
        print_dist("manipulated_object")
        print_dist("affected_object")  # Included as per discussion
        print_dist("task_step")

    else:
        logger.warning("No data processed.")

    total_elapsed = time.perf_counter() - start_time
    logger.info(
        "Done. videos=%d (ok=%d, err=%d) elapsed=%.1fs",
        len(txt_files),
        ok_count,
        err_count,
        total_elapsed,
    )


if __name__ == "__main__":
    main()

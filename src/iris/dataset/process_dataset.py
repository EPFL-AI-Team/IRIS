import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dataset_config import (
    DatasetPaths,
    configure_logging,
    ensure_output_dirs,
    load_dataset_config,
    resolve_paths,
    validate_inputs,
)
from PIL import Image
from tqdm import tqdm

# Make OpenCV optional - only needed if extracting frames
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore

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
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        default=False,
        help="If set, extracts frames from videos. Default is False (annotation processing only).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
        help="Number of parallel workers.",
    )
    return parser.parse_args()


def _build_paths(profile_paths: DatasetPaths) -> dict[str, Path]:
    return {
        "videos": profile_paths.profile.videos_dir,
        "annotations": profile_paths.profile.annotations_dir,
        "output_frames": profile_paths.frames_dir,
        "csv_out": profile_paths.csv_per_video_dir,
        "consolidated_csv": profile_paths.consolidated_csv,
    }


def _target_frame_slots(
    *, canonical_max_frames: int, frames_per_segment: int
) -> np.ndarray:
    """Return the canonical slot indices to materialize on disk.

    Example: canonical_max_frames=16, frames_per_segment=4 -> [0, 5, 10, 15]
    """
    if canonical_max_frames <= 0:
        raise ValueError("canonical_max_frames must be > 0")
    if frames_per_segment <= 0:
        raise ValueError("frames_per_segment must be > 0")
    if frames_per_segment > canonical_max_frames:
        raise ValueError("frames_per_segment cannot exceed canonical_max_frames")

    slots = np.linspace(0, canonical_max_frames - 1, frames_per_segment, dtype=int)
    return np.unique(slots)


# Helpers


def fill_task_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    FineBio txt files mix 'tasks' (high-level steps) and 'atomic actions'
    in the same list. Tasks usually have empty verbs.
    This maps the 'task' label to all atomic actions that fall within its time window.
    """
    # Identify rows that define a Task (Step) vs Atomic Actions
    # In FineBio, task rows have a 'task' string, action rows usually have empty 'task' but have 'verb'
    task_rows = df[df["task"].notna() & (df["task"] != "")].copy()
    action_rows = df[df["verb"].notna() & (df["verb"] != "")].copy()

    # If no tasks are defined, return actions as is (context will be NaN)
    if task_rows.empty:
        return action_rows

    # Sort to ensure logical time progression
    task_rows = task_rows.sort_values("start_sec")

    # Efficiently assign tasks to actions
    # We create a new column 'context_task' to avoid overwriting original structure if needed
    action_rows["context_task"] = None

    for _, t_row in task_rows.iterrows():
        t_start, t_end = t_row["start_sec"], t_row["end_sec"]
        t_label = t_row["task"]

        # Find actions fully or partially inside this task window
        mask = (action_rows["start_sec"] >= t_start) & (
            action_rows["start_sec"] < t_end
        )
        action_rows.loc[mask, "context_task"] = t_label

    return action_rows


def extract_frames_for_segment(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    segment_dir: Path,
    *,
    canonical_max_frames: int,
    frames_per_segment: int,
) -> list[str]:
    """Handles the OpenCV logic to extract frames."""
    if not CV2_AVAILABLE:
        logger.error("OpenCV (cv2) not available. Cannot extract frames.")
        return []

    if not video_path.exists():
        return []

    target_slots = _target_frame_slots(
        canonical_max_frames=canonical_max_frames, frames_per_segment=frames_per_segment
    )

    # Fast path: if all target files exist, skip any OpenCV work.
    segment_dir.mkdir(parents=True, exist_ok=True)
    target_paths = [segment_dir / f"frame_{slot:02d}.jpg" for slot in target_slots]
    missing = [p for p in target_paths if not p.exists()]
    if not missing:
        return [str(p) for p in target_paths]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        return []

    vid_start_f = int(start_sec * fps)
    vid_end_f = int(end_sec * fps)
    if vid_end_f <= vid_start_f:
        vid_end_f = vid_start_f + 1

    duration_f = max(vid_end_f - vid_start_f, 1)

    for out_path in missing:
        # Map the canonical slot to a position within the segment's frame window.
        # fraction: slot 0 -> start, slot (N-1) -> end
        slot_idx = int(out_path.stem.split("_")[-1])
        fraction = (
            slot_idx / float(canonical_max_frames - 1)
            if canonical_max_frames > 1
            else 0.0
        )
        video_frame_idx = vid_start_f + int(round(fraction * duration_f))

        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
            out_path, quality=85
        )

    cap.release()
    # Return the canonical ordering (even if a few couldn't be decoded).
    return [str(p) for p in target_paths]


# Worker


def process_video_annotations(
    video_id: str,
    txt_path: Path,
    extract_frames: bool,
    paths: dict[str, Path],
    canonical_max_frames: int,
    frames_per_segment: int,
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

    # Clean up columns (we want: video_id, start, end, verb, tool, target, task)
    # Mapping dataset columns to standard names
    # Dataset: verb, manipulated_object, affected_object, context_task

    final_rows: list[dict[str, Any]] = []
    target_slots = _target_frame_slots(
        canonical_max_frames=canonical_max_frames,
        frames_per_segment=frames_per_segment,
    ).tolist()
    video_file_path = paths["videos"] / f"{video_id}.mp4"

    for _, row in processed_df.iterrows():
        # Skip if no verb (not an atomic action)
        if pd.isna(row.get("verb")):
            continue

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

        # JSONL / Frame Logic
        # We define where frames SHOULD be
        segment_dir = paths["output_frames"] / segment_id

        # Extract if requested (extract_frames_for_segment will instantly no-op if files exist)
        if extract_frames:
            extracted = extract_frames_for_segment(
                video_file_path,
                row["start_sec"],
                row["end_sec"],
                segment_dir,
                canonical_max_frames=canonical_max_frames,
                frames_per_segment=frames_per_segment,
            )
            _ = extracted

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

    # Validate OpenCV availability if frame extraction is requested
    if args.extract_frames and not CV2_AVAILABLE:
        logger.error(
            "Frame extraction requested but OpenCV (cv2) is not installed. "
            "Install with: pip install opencv-python-headless"
        )
        return

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

    logger.info("Extract frames: %s", args.extract_frames)
    logger.info("Workers: %d", args.workers)

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
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        future_map = {
            executor.submit(
                process_video_annotations,
                f.stem,
                f,
                args.extract_frames,
                paths,
                config.canonical_max_frames,
                config.frames_per_segment,
            ): f.stem
            for f in txt_files
        }

        for future in tqdm(
            as_completed(future_map), total=len(txt_files), desc="Processing Videos"
        ):
            vid_id = future_map[future]
            try:
                rows, lines = future.result()
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

import argparse
import json
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
        "jsonl_out": profile_paths.jsonl_per_video_dir,
        "manifest_file": profile_paths.consolidated_jsonl,
        "consolidated_csv": profile_paths.consolidated_csv,
    }


MAX_FRAMES = 16

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
    video_path: Path, start_sec: float, end_sec: float, segment_dir: Path
) -> list[str]:
    """Handles the OpenCV logic to extract frames."""
    if not CV2_AVAILABLE:
        logger.error("OpenCV (cv2) not available. Cannot extract frames.")
        return []

    if not video_path.exists():
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f = int(start_sec * fps)
    end_f = int(end_sec * fps)
    if end_f <= start_f:
        end_f = start_f + 1

    indices = np.linspace(start_f, end_f, MAX_FRAMES, dtype=int)
    indices = np.unique(indices)

    frame_paths: list[str] = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Save frame
            out_path = segment_dir / f"frame_{i:02d}.jpg"
            # Only save if we strictly need to (though function is called only when needed usually)
            if not out_path.exists():
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
                    out_path, quality=85
                )
            frame_paths.append(str(out_path))

    cap.release()
    return frame_paths


# Worker


def process_video_annotations(
    video_id: str,
    txt_path: Path,
    extract_frames: bool,
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    1. Loads raw txt.
    2. Processes/Fills tasks.
    3. Generates CSV.
    4. Generates JSONL entries.
    5. (Optional) Extracts frames.
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
    jsonl_entries: list[str] = []

    video_file_path = paths["videos"] / f"{video_id}.mp4"

    for _, row in processed_df.iterrows():
        # Skip if no verb (not an atomic action)
        if pd.isna(row.get("verb")):
            continue

        # Data Row
        item = {
            "video_id": video_id,
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "verb": str(row.get("verb", "unknown")),
            "manipulated_object": str(row.get("manipulated_object", "none")),
            "affected_object": str(row.get("affected_object", "none")),
            "task_step": str(row.get("context_task", "unknown")),
            "hand": str(row.get("hand_side", "unknown")),
        }
        final_rows.append(item)

        # JSONL / Frame Logic
        # We define where frames SHOULD be
        segment_id = f"{video_id}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"
        segment_dir = paths["output_frames"] / segment_id

        frame_paths = []

        # Check existing frames
        if segment_dir.exists():
            found = sorted(segment_dir.glob("frame_*.jpg"))
            if found:
                frame_paths = [str(p) for p in found]

        # Extract if requested AND frames are missing
        if extract_frames and (not frame_paths or len(frame_paths) < MAX_FRAMES):
            segment_dir.mkdir(parents=True, exist_ok=True)
            extracted = extract_frames_for_segment(
                video_file_path, row["start_sec"], row["end_sec"], segment_dir
            )
            if extracted:
                frame_paths = extracted

        # Create JSONL Entry (Even if frames are missing, we create the metadata entry)
        # Note: If frames are missing and extraction is False, path list might be empty.
        # Ideally, we construct the paths assuming they exist if we plan to generate them later.

        if not frame_paths:
            # Construct theoretical paths if we assume they will be generated later
            frame_paths = [
                str(segment_dir / f"frame_{i:02d}.jpg") for i in range(MAX_FRAMES)
            ]

        prompt_text = (
            "Analyze this video segment of a biological experiment. "
            "Output a valid JSON object with keys: action, tool, target, context."
        )

        model_response = {
            "action": item["verb"],
            "tool": item["manipulated_object"],
            "target": item["affected_object"],
            "context": item["task_step"],
        }

        json_entry = {
            "id": segment_id,
            "image_paths": frame_paths,  # Custom field for reference
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": fp} for fp in frame_paths],
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(model_response, indent=2)}
                    ],
                },
            ],
        }
        jsonl_entries.append(json.dumps(json_entry))

    # Save Individual Files
    if final_rows:
        # CSV
        df_out = pd.DataFrame(final_rows)
        csv_path = paths["csv_out"] / f"{video_id}.csv"
        df_out.to_csv(csv_path, index=False)

        # JSONL
        jsonl_path = paths["jsonl_out"] / f"{video_id}.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("\n".join(jsonl_entries))

    return final_rows, jsonl_entries


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

    # Create Directories
    for k, p in paths.items():
        if k not in ["videos", "annotations", "manifest_file", "consolidated_csv"]:
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
    all_jsonl_lines = []

    start_time = time.perf_counter()
    ok_count = 0
    err_count = 0

    # Parallel Processing
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        future_map = {
            executor.submit(
                process_video_annotations, f.stem, f, args.extract_frames, paths
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
                    all_jsonl_lines.extend(lines)
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

        # 2. Consolidated JSONL
        with open(paths["manifest_file"], "w") as f:
            f.write("\n".join(all_jsonl_lines))
        logger.info("Saved consolidated JSONL to: %s", paths["manifest_file"])

        # 3. Metrics
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

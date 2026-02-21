import argparse
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from .dataset_config import (
    configure_logging,
    ensure_output_dirs,
    load_dataset_config,
    resolve_paths,
)
from PIL import Image
from tqdm import tqdm

from .logic import _target_frame_slots, is_valid_action
from .training_format import (
    chat_jsonl_entry,
    expected_output_json,
    pick_prompt,
)

logger = logging.getLogger(__name__)


# Optional OpenCV - required for frame extraction
try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore


def extract_frames_for_segment(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    segment_dir: Path,
    *,
    canonical_max_frames: int,
    frames_per_segment: int,
) -> list[str]:
    """Extract canonical frames for a segment using OpenCV."""
    if not CV2_AVAILABLE:
        logger.error("OpenCV (cv2) not available. Cannot extract frames.")
        return []

    if not video_path.exists():
        return []

    target_slots = _target_frame_slots(
        canonical_max_frames=canonical_max_frames,
        frames_per_segment=frames_per_segment,
    )

    target_paths = [segment_dir / f"frame_{slot:02d}.jpg" for slot in target_slots]

    if segment_dir.exists():
        missing_paths = [p for p in target_paths if not p.exists()]
        if not missing_paths:
            return [str(p) for p in target_paths]
    else:
        segment_dir.mkdir(parents=True, exist_ok=True)
        missing_paths = target_paths

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        cap.release()
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    vid_start_f = int(start_sec * fps)
    vid_end_f = int(end_sec * fps)
    if vid_end_f <= vid_start_f:
        vid_end_f = vid_start_f + 1

    duration_f = max(vid_end_f - vid_start_f, 1)

    for out_path in missing_paths:
        slot_idx = int(out_path.stem.split("_")[-1])
        fraction = (
            slot_idx / float(canonical_max_frames - 1)
            if canonical_max_frames > 1
            else 0.0
        )
        video_frame_idx = vid_start_f + round(fraction * duration_f)
        if total_frames > 0:
            video_frame_idx = max(0, min(video_frame_idx, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()

        if not ret and total_frames > 0:
            last_frame_idx = total_frames - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
            ret, frame = cap.read()

        if ret:
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
                out_path, quality=85
            )
        else:
            logger.warning("Failed to extract %s even after clamping.", out_path)

    cap.release()
    return [str(p) for p in target_paths]


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
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
        help="Number of parallel workers for frame extraction.",
    )
    return parser.parse_args()


def get_stratified_splits(
    annotations_df: pd.DataFrame,
    *,
    train_quota_per_verb: int = 1000,
    val_test_quota: int = 200,
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

    # --- DOWNSAMPLING (Apply quotas to all splits) ---
    logger.info(
        f"Applying quotas per-verb: train={train_quota_per_verb}, val/test={val_test_quota}"
    )

    def quota_downsample(pool_df: pd.DataFrame, quota: int) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for verb in pool_df["verb"].unique():
            v_df = pool_df[pool_df["verb"] == verb]
            count = len(v_df)
            if count <= quota:
                parts.append(v_df)
            else:
                tool_counts = v_df["manipulated_object"].value_counts()
                weights = 1.0 / v_df["manipulated_object"].map(tool_counts)
                selected = v_df.sample(n=quota, weights=weights, random_state=42)
                parts.append(selected)
        return pd.concat(parts) if parts else pool_df

    train_final = quota_downsample(train_pool, train_quota_per_verb)
    val_final = quota_downsample(val_df, val_test_quota)
    test_final = quota_downsample(test_df, val_test_quota)

    logger.info("--- Official Split Statistics ---")
    logger.info(f"Train: {len(train_final)} (Downsampled from {len(train_pool)})")
    logger.info(f"Val:   {len(val_final)}")
    logger.info(f"Test:  {len(test_final)}")

    return train_final, val_final, test_final


def create_jsonl(
    annotations_df: pd.DataFrame,
    *,
    frame_base_dir: Path,
    output_path: Path,
    videos_base_dir: Path,
    canonical_max_frames: int,
    frames_per_segment: int,
    workers: int = 8,
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

    # Canonical-slot strategy (matches process_dataset.py)
    frame_slots = _target_frame_slots(
        canonical_max_frames=canonical_max_frames, frames_per_segment=frames_per_segment
    )

    # First pass: identify missing segments and prepare items
    items: list[dict] = []
    missing_jobs: list[tuple[Path, float, float, Path]] = []

    for _, row in annotations_df.iterrows():
        segment_id = f"{row['video_id']}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"
        segment_dir = frame_base_dir / segment_id
        image_paths = [segment_dir / f"frame_{i:02d}.jpg" for i in frame_slots]
        missing_paths = [p for p in image_paths if not p.exists()]
        if missing_paths:
            missing_jobs.append((
                videos_base_dir / f"{row['video_id']}.mp4",
                row["start_sec"],
                row["end_sec"],
                segment_dir,
            ))

        items.append({
            "row": row,
            "segment_dir": segment_dir,
            "segment_id": segment_id,
            "frame_slots": frame_slots,
        })

    # Extract missing frames in parallel (just-in-time)
    if missing_jobs:
        max_workers = min(workers or 8, (os.cpu_count() or 4))
        logger.info(
            "Extracting frames for %d missing segments using %d workers...",
            len(missing_jobs),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_map = {
                ex.submit(
                    extract_frames_for_segment,
                    video_path,
                    start,
                    end,
                    segdir,
                    canonical_max_frames=canonical_max_frames,
                    frames_per_segment=frames_per_segment,
                ): (video_path, start, end, segdir)
                for (video_path, start, end, segdir) in missing_jobs
            }

            completed = 0
            for fut in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc="Extracting frames",
                unit="seg",
            ):
                try:
                    _ = fut.result()
                except Exception:
                    logger.exception(
                        "Frame extraction task failed for %s", future_map.get(fut)
                    )
                completed += 1
                if completed % 100 == 0:
                    logger.info(
                        "Frame extraction progress: %d/%d segments",
                        completed,
                        len(future_map),
                    )
        logger.info(
            "Frame extraction completed: %d segments processed", len(missing_jobs)
        )
    else:
        logger.info("All required frames already exist on disk. Skipping extraction.")

    # Second pass: build entries (frames should now exist for most segments)
    for it in items:
        row = it["row"]
        segment_dir = it["segment_dir"]
        segment_id = it["segment_id"]
        frame_slots = it["frame_slots"]

        image_paths = [str(segment_dir / f"frame_{i:02d}.jpg") for i in frame_slots]

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
                    "Config: per_segment=%d canonical=%d. Fix: ensure videos are accessible and rerun.",
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
                image_paths=[str(p) for p in image_paths],
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
    logger.info("Workers: %d", args.workers)

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
    filtered_df = annotations_df[
        annotations_df.apply(
            lambda r: is_valid_action(r, min_duration, max_duration), axis=1
        )
    ]

    # Official split (use quotas from config)
    train_df, val_df, test_df = get_stratified_splits(
        filtered_df,
        train_quota_per_verb=config.train_per_verb,
        val_test_quota=config.val_test_per_verb,
    )

    logger.info("--- Final File Statistics ---")
    logger.info("Train: %d samples", len(train_df))
    logger.info("Val:   %d samples", len(val_df))
    logger.info("Test:  %d samples", len(test_df))

    # Generate Files
    create_jsonl(
        train_df,
        frame_base_dir=frame_base_dir,
        videos_base_dir=resolved.profile.videos_dir,
        output_path=out_train,
        canonical_max_frames=config.canonical_max_frames,
        frames_per_segment=config.frames_per_segment,
        workers=args.workers,
    )
    create_jsonl(
        val_df,
        frame_base_dir=frame_base_dir,
        videos_base_dir=resolved.profile.videos_dir,
        output_path=out_val,
        canonical_max_frames=config.canonical_max_frames,
        frames_per_segment=config.frames_per_segment,
        workers=args.workers,
    )
    create_jsonl(
        test_df,
        frame_base_dir=frame_base_dir,
        videos_base_dir=resolved.profile.videos_dir,
        output_path=out_test,
        canonical_max_frames=config.canonical_max_frames,
        frames_per_segment=config.frames_per_segment,
        workers=args.workers,
    )

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# READ-ONLY input data (The slow, big storage)
SOURCE_BASE = Path("/work/team-ai/IRIS_vlm")
# SOURCE_BASE = Path("/Users/marcushamelink/Developer/ml/IRIS-semester-project/data")

# READ-WRITE output (The fast scratch space)
# SCRATCH_BASE = Path(
#     "/Users/marcushamelink/Developer/ml/IRIS-semester-project/data/dataset_gen_output"
# )
SCRATCH_BASE = Path(
    "/scratch/izar/mhamelin/finebio_data"
)

PATHS = {
    # Inputs
    "videos": SOURCE_BASE / "finebio/videos/w640",
    "annotations": SOURCE_BASE / "finebio/annotations/finebio_action_annotations",
    # Outputs
    "output_frames": SCRATCH_BASE / "frames",
    "manifest_file": SCRATCH_BASE / "train.jsonl",
}

MAX_FRAMES = 16  # Optimal for Qwen 3B

def fill_task_column(details_df: pd.DataFrame, tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Fill task column for all rows based on time overlap with task annotations"""

    result_df: pd.DataFrame = details_df.copy()

    for _, task_row in tasks_df.iterrows():
        task_start: float = task_row["start_sec"]
        task_end: float = task_row["end_sec"]
        task: str = task_row["task"]

        mask: pd.Series = (
            (result_df["start_sec"] < task_end)
            & (result_df["start_sec"] > task_start)
            & (result_df["end_sec"] < task_end)
            & (result_df["end_sec"] > task_start)
            & (result_df["start_sec"] < result_df["end_sec"])
            & (result_df["task"].isna())
        )

        result_df.loc[mask, "task"] = task

    return result_df


def extract_frames_uniform(
    video_path: Path, start_sec: float, end_sec: float, max_frames: int
) -> list[Any]:
    """
    Opens video, jumps to specific timestamps, and extracts exactly 'max_frames'.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_f = int(start_sec * fps)
    end_f = int(end_sec * fps)

    # Handle edge case where segment is < 1 frame
    if end_f <= start_f:
        end_f = start_f + 1

    # Uniformly pick frame indices
    indices = np.linspace(start_f, end_f, max_frames, dtype=int)
    # Deduplicate indices if video segment is too short for 16 distinct frames
    indices = np.unique(indices)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # OpenCV BGR -> PIL RGB
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()
    return frames


def main() -> None:
    # --- Configuration ---
    MAX_FRAMES = 16

    # Setup output dirs
    print(f"Setting up directories at {PATHS['output_frames']}...")
    PATHS["output_frames"].mkdir(parents=True, exist_ok=True)

    # --- Load Annotations ---
    print("Loading annotation files...")
    txt_files = list(PATHS["annotations"].glob("*.txt"))

    all_dfs = []
    for txt_file in txt_files:
        video_id = txt_file.stem
        try:
            df = pd.read_csv(txt_file)
            task_mask = df["task"].notna()
            task_df = df[task_mask]
            action_df = df[~task_mask]

            processed_df = fill_task_column(action_df, task_df)
            processed_df["video_id"] = video_id
            all_dfs.append(processed_df)
        except Exception as e:
            print(f"Error reading {txt_file.name}: {e}")

    if not all_dfs:
        print("No valid annotations found.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df[final_df["task"].notna()]

    print(f"Total samples to process: {len(final_df)}")

    # --- OPTIMIZATION START: Group by Video ID ---
    # We group here so we only open each video file ONCE
    grouped = final_df.groupby("video_id")

    jsonl_lines = []

    # Create a progress bar for the total number of segments
    with tqdm(total=len(final_df), desc="Processing Segments") as pbar:
        for video_id, group_df in grouped:
            video_path = PATHS["videos"] / f"{video_id}.mp4"

            if not video_path.exists():
                pbar.update(len(group_df))  # Skip these counts
                continue

            # Open Video ONCE per file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                pbar.update(len(group_df))
                continue

            video_fps = cap.get(cv2.CAP_PROP_FPS)

            # Sort segments by start time to minimize seeking back and forth
            group_df = group_df.sort_values("start_sec")

            for _, row in group_df.iterrows():
                # --- Inline Extraction Logic (Reusing 'cap') ---
                start_f = int(row["start_sec"] * video_fps)
                end_f = int(row["end_sec"] * video_fps)

                if end_f <= start_f:
                    end_f = start_f + 1

                # Uniform sampling
                indices = np.linspace(start_f, end_f, MAX_FRAMES, dtype=int)
                indices = np.unique(indices)

                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )

                # If extraction failed (e.g. end of video), skip
                if not frames:
                    pbar.update(1)
                    continue

                # --- Save Images ---
                segment_id = f"{video_id}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"
                segment_dir = PATHS["output_frames"] / segment_id
                segment_dir.mkdir(exist_ok=True)

                frame_paths = []
                for j, frame in enumerate(frames):
                    p = segment_dir / f"frame_{j:02d}.jpg"
                    frame.save(p, quality=85)
                    frame_paths.append(str(p))

                # --- Construct JSONL ---
                prompt_text = (
                    "Analyze this video segment of a biological experiment. "
                    "Output a valid JSON object with the following keys:\n"
                    "- action: The specific movement (e.g., streaking, inspecting).\n"
                    "- tool: The active instrument.\n"
                    "- target: The object being acted upon.\n"
                    "- context: The protocol step."
                )

                model_response = {
                    "action": row.get("verb", "unknown"),
                    "tool": row.get("noun", "unknown"),
                    "target": row.get("target", "unknown"),
                    "context": row.get("task", "unknown"),
                }

                entry = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                *[
                                    {"type": "image", "image": str(fp)}
                                    for fp in frame_paths
                                ],
                                {"type": "text", "text": prompt_text},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(model_response, indent=2),
                                }
                            ],
                        },
                    ]
                }
                jsonl_lines.append(json.dumps(entry))
                pbar.update(1)

            # Close video after processing all its segments
            cap.release()

    # --- SAVE ---
    print(f"Saving {len(jsonl_lines)} samples to {PATHS['manifest_file']}...")
    with open(PATHS["manifest_file"], "w") as f:
        f.write("\n".join(jsonl_lines))

    print("Done!")


if __name__ == "__main__":
    main()

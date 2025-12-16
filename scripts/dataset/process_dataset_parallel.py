import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

# READ-ONLY input data
SOURCE_BASE = Path("/work/team-ai/IRIS_vlm")

# READ-WRITE output (Scratch)
SCRATCH_BASE = Path("/scratch/izar/mhamelin/finebio_data")

PATHS = {
    "videos": SOURCE_BASE / "finebio/videos/w640",
    "annotations": SOURCE_BASE / "finebio/annotations/finebio_action_annotations",
    "output_frames": SCRATCH_BASE / "frames",
    "manifest_file": SCRATCH_BASE / "raw_data_v2.jsonl"
}

MAX_FRAMES = 16

# ==========================================
# 2. WORKER FUNCTION
# ==========================================


def process_single_video(
    video_id: str, group_df: pd.DataFrame, video_dir: Path, output_dir: Path
) -> list[Any]:
    """
    Processes segments for ONE video. 
    SKIPS video reading if frames already exist on disk.
    """
    video_path = video_dir / f"{video_id}.mp4"
    jsonl_entries = []
    
    # We initialize 'cap' as None. We only open the video if we absolutely need to extract frames.
    cap = None 
    video_fps = None

    # Sort segments
    group_df = group_df.sort_values("start_sec")

    for _, row in group_df.iterrows():
        # 1. Basic Filtering: Skip if Action (verb) is missing
        action = row.get("verb")
        if pd.isna(action) or str(action).lower() in ["nan", "none", "", "unknown"]:
            continue

        # 2. Construct Segment ID and Path
        segment_id = f"{video_id}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"
        segment_dir = output_dir / segment_id
        
        frame_paths = []
        frames_extracted = False

        # 3. CHECK: Do frames already exist?
        if segment_dir.exists():
            # Get existing frames sorted by name
            existing_frames = sorted(list(segment_dir.glob("frame_*.jpg")))
            
            # If we have enough frames, just use them (Skip CV2!)
            if len(existing_frames) > 0: 
                frame_paths = [str(p) for p in existing_frames]
                frames_extracted = True

        # 4. IF NOT EXIST: Perform Extraction
        if not frames_extracted:
            continue
            
            # Open video only now if not already open
            if cap is None:
                if not video_path.exists():
                    break # Video file missing, can't extract
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    break
                video_fps = cap.get(cv2.CAP_PROP_FPS)

            segment_dir.mkdir(parents=True, exist_ok=True)
            
            start_f = int(row["start_sec"] * video_fps)
            end_f = int(row["end_sec"] * video_fps)
            if end_f <= start_f: end_f = start_f + 1
            
            indices = np.linspace(start_f, end_f, MAX_FRAMES, dtype=int)
            indices = np.unique(indices)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            if not frames:
                continue

            # Save to disk
            for j, frame in enumerate(frames):
                p = segment_dir / f"frame_{j:02d}.jpg"
                # frame.save(p, quality=85)
                frame_paths.append(str(p))

        # 5. Construct JSON Entry (With Correct Columns)
        prompt_text = (
            "Analyze this video segment of a biological experiment. "
            "Output a valid JSON object with the following keys:\n"
            "- action: The specific movement (e.g., streaking, inspecting).\n"
            "- tool: The active instrument.\n"
            "- target: The object being acted upon.\n"
            "- context: The protocol step."
        )

        # Handle NaNs gracefully for the JSON output
        def clean(val):
            s = str(val)
            if s.lower() == "nan": return "unknown"
            return s

        model_response = {
            "action": clean(row.get("verb")),
            "tool": clean(row.get("manipulated_object")),
            "target": clean(row.get("affected_object")),
            "context": clean(row.get("task"))
        }

        entry = {
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
            ]
        }
        jsonl_entries.append(json.dumps(entry))

    if cap is not None:
        cap.release()
        
    return jsonl_entries


# ==========================================
# 3. MAIN (Orchestrator)
# ==========================================


def fill_task_column(details_df: pd.DataFrame, tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Map tasks to actions based on timestamps."""
    result_df = details_df.copy()
    for _, task_row in tasks_df.iterrows():
        mask = (
            (result_df["start_sec"] >= task_row["start_sec"])
            & (result_df["end_sec"] <= task_row["end_sec"])
            & (result_df["task"].isna())
        )
        result_df.loc[mask, "task"] = task_row["task"]
    return result_df

def main():
    PATHS["output_frames"].mkdir(parents=True, exist_ok=True)
    
    print("Loading annotations...")
    all_dfs = []
    for txt_file in PATHS["annotations"].glob("*.txt"):
        try:
            df = pd.read_csv(txt_file)
            task_df = df[df["task"].notna()]
            action_df = df[df["task"].isna()]
            
            processed_df = fill_task_column(action_df, task_df)
            processed_df["video_id"] = txt_file.stem
            all_dfs.append(processed_df)
        except Exception:
            continue

    if not all_dfs:
        print("No annotations found.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Filtering: We only want rows where we actually have a task AND an action
    # (Though logic above ensures task is filled, action might be missing)
    final_df = final_df[final_df["task"].notna()] 
    
    print(f"Total samples to process: {len(final_df)}")
    
    grouped = final_df.groupby("video_id")
    max_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 8)) # Increased default slightly
    print(f"Starting processing with {max_workers} workers...")
    print("(Existing frames will be reused to speed up generation)")

    all_jsonl_lines = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(
                process_single_video,
                vid_id,
                group,
                PATHS["videos"],
                PATHS["output_frames"],
            ): vid_id
            for vid_id, group in grouped
        }
        
        for future in tqdm(as_completed(future_to_video), total=len(grouped)):
            try:
                result_lines = future.result()
                all_jsonl_lines.extend(result_lines)
            except Exception as e:
                print(f"Worker failed: {e}")

    print(f"Saving {len(all_jsonl_lines)} samples to {PATHS['manifest_file']}...")
    with open(PATHS["manifest_file"], "w") as f:
        f.write("\n".join(all_jsonl_lines))

    print("Done!")


if __name__ == "__main__":
    main()

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

# READ-ONLY input data
SOURCE_BASE = Path('/work/team-ai/IRIS_vlm')

# READ-WRITE output (Scratch)
SCRATCH_BASE = Path('/scratch/izar/mhamelin/finebio_data')

PATHS = {
    "videos": SOURCE_BASE / "finebio/videos/w640",
    "annotations": SOURCE_BASE / "finebio/annotations/finebio_action_annotations",
    "output_frames": SCRATCH_BASE / "frames",
    "manifest_file": SCRATCH_BASE / "train.jsonl"
}

MAX_FRAMES = 16

# ==========================================
# 2. WORKER FUNCTION (Runs on multiple cores)
# ==========================================

def process_single_video(video_id, group_df, video_dir, output_dir):
    """
    Processes all segments for ONE video file.
    Returns a list of JSONL strings (one per segment).
    """
    video_path = video_dir / f"{video_id}.mp4"
    jsonl_entries = []
    
    if not video_path.exists():
        return []

    # Open Video ONCE
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sort segments to minimize seeking
    group_df = group_df.sort_values("start_sec")

    for _, row in group_df.iterrows():
        start_f = int(row["start_sec"] * video_fps)
        end_f = int(row["end_sec"] * video_fps)
        
        if end_f <= start_f: end_f = start_f + 1
        
        # Uniform sampling
        indices = np.linspace(start_f, end_f, MAX_FRAMES, dtype=int)
        indices = np.unique(indices)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR -> RGB
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # Skip if extraction failed
        if not frames:
            continue

        # Save Images to Scratch
        segment_id = f"{video_id}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"
        segment_dir = output_dir / segment_id
        segment_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        for j, frame in enumerate(frames):
            p = segment_dir / f"frame_{j:02d}.jpg"
            frame.save(p, quality=85)
            frame_paths.append(str(p))

        # Construct JSON Entry
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
            "context": row.get("task", "unknown")
        }

        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": str(fp)} for fp in frame_paths],
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

    cap.release()
    return jsonl_entries

# ==========================================
# 3. MAIN (Orchestrator)
# ==========================================

def fill_task_column(details_df, tasks_df):
    """Map tasks to actions based on timestamps."""
    result_df = details_df.copy()
    for _, task_row in tasks_df.iterrows():
        mask = (
            (result_df["start_sec"] >= task_row["start_sec"]) &
            (result_df["end_sec"] <= task_row["end_sec"]) &
            (result_df["task"].isna())
        )
        result_df.loc[mask, "task"] = task_row["task"]
    return result_df

def main():
    # 1. Setup
    PATHS["output_frames"].mkdir(parents=True, exist_ok=True)
    
    # 2. Load Annotations
    print("Loading annotations...")
    all_dfs = []
    for txt_file in PATHS["annotations"].glob("*.txt"):
        try:
            df = pd.read_csv(txt_file)
            task_df = df[df["task"].notna()]
            action_df = df[df["task"].isna()] # Use rows where task IS None
            
            processed_df = fill_task_column(action_df, task_df)
            processed_df["video_id"] = txt_file.stem
            all_dfs.append(processed_df)
        except Exception:
            continue

    if not all_dfs:
        print("No annotations found.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df[final_df["task"].notna()] # Only keep valid tasks
    
    print(f"Total samples to process: {len(final_df)}")
    
    # 3. Group by Video for Parallel Processing
    grouped = final_df.groupby("video_id")
    
    # Use as many CPUs as SLURM gives us, or default to 4
    max_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    print(f"Starting parallel processing with {max_workers} workers...")

    all_jsonl_lines = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_video = {
            executor.submit(
                process_single_video, 
                vid_id, 
                group, 
                PATHS["videos"], 
                PATHS["output_frames"]
            ): vid_id 
            for vid_id, group in grouped
        }
        
        # Progress Bar
        for future in tqdm(as_completed(future_to_video), total=len(grouped)):
            try:
                result_lines = future.result()
                all_jsonl_lines.extend(result_lines)
            except Exception as e:
                print(f"Worker failed: {e}")

    # 4. Save Manifest
    print(f"Saving {len(all_jsonl_lines)} samples to {PATHS['manifest_file']}...")
    with open(PATHS['manifest_file'], 'w') as f:
        f.write('\n'.join(all_jsonl_lines))
        
    print("Done!")

if __name__ == "__main__":
    main()

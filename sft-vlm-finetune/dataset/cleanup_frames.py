import json
import shutil
import os
from pathlib import Path
from tqdm import tqdm

# Config
BASE_DIR = Path("/scratch/iris/finebio_processed")
FRAME_DIR = BASE_DIR / "frames"
SPLIT_DIR = BASE_DIR / "splits/train_10k_4_frames"
KEEP_SLOTS = {0, 5, 10, 15}
DRY_RUN = False  # Set to False to actually delete files


def cleanup():
    # 1. Collect IDs
    keep_ids = set()
    for split in ["finebio_train.jsonl", "finebio_val.jsonl", "finebio_test.jsonl"]:
        path = SPLIT_DIR / split
        if not path.exists():
            print(f"Warning: {path} not found.")
            continue
        with open(path) as f:
            for line in f:
                keep_ids.add(json.loads(line)["id"])

    print(f"Targeting {len(keep_ids)} segments for retention.")

    # 2. Count folders for progress bar (fast)
    print("Scanning directory structure...")
    all_folders = [f.path for f in os.scandir(FRAME_DIR) if f.is_dir()]

    deleted_folders = 0
    cleaned_files = 0

    # 3. Process with Progress Bar
    for folder_path in tqdm(all_folders, desc="Cleaning frames", unit="folder"):
        folder_name = os.path.basename(folder_path)

        if folder_name not in keep_ids:
            # Case 1: Unused segment
            if not DRY_RUN:
                shutil.rmtree(folder_path)
            deleted_folders += 1
        else:
            # Case 2: Used segment -> Clean inner files
            for entry in os.scandir(folder_path):
                if entry.name.endswith(".jpg"):
                    try:
                        # 'frame_05.jpg' -> 5
                        slot_idx = int(entry.name.split("_")[-1].split(".")[0])
                        if slot_idx not in KEEP_SLOTS:
                            if not DRY_RUN:
                                os.remove(entry.path)
                            cleaned_files += 1
                    except (ValueError, IndexError):
                        continue

    status = "[DRY RUN - No files deleted]" if DRY_RUN else "[LIVE RUN]"
    print(f"\nCleanup complete {status}")
    print(f"Unused folders removed: {deleted_folders}")
    print(f"Redundant frames removed: {cleaned_files}")


if __name__ == "__main__":
    cleanup()

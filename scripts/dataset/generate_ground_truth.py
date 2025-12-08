#!/usr/bin/env python3
"""
LabLog: One-Pass Data Generation Script

NOT USED ANYMORE, BUT on sait jamais
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
    from tqdm import tqdm
except ImportError:
    print("Please run: pip install google-genai tqdm ffmpeg-python")
    exit(1)

# === CONFIGURATION ===
DEFAULT_MODEL = "gemini-2.5-pro"
CHUNK_DURATION = 60
CHUNK_OVERLAP = 0

# Hardcoded paths for the --test_prompt mode
TEST_VIDEO_PATH = Path("data/CHUV-videos/720p/GOPR_01.mp4")
TEST_START_TIME = 450.0
TEST_DURATION = 30

PROMPT_TEMPLATE = """You are a laboratory state classifier. Analyze this video segment at 1 FPS.

TASK: Identify the active interaction state and the bounding box of the main interaction zone.

CLASSES (Mutually Exclusive for the main action):
- "INSPECTING": Researcher is holding a red agar plate up (usually to read label/check growth).
- "COUNTING": Researcher is tapping the agar plate with a pen/marker.
- "LOGGING": Researcher is writing on the paper log sheet.
- "IDLE": No active interaction with dishes or logs.

OUTPUT SCHEMA (Strict JSON):
[
  {{
    "timestamp": <float seconds>,
    "state": "INSPECTING|COUNTING|LOGGING|IDLE",
    "main_object_box_2d": [ymin, xmin, ymax, xmax],
    "plate_label": "<text visible on dish or null>",  // Add this
    "reasoning": "brief description"
  }}
]

RULES:
- Return ONLY valid JSON.
- If IDLE, box_2d should be null.
- Extract any visible text/labels on petri dishes when readable.
- Focus ONLY on the primary action hand/object.
"""

OUTPUT_DIR = Path("data/dataset_gen_output")


def get_video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def extract_chunk(
    video_path: Path, start: float, duration: int, output_path: Path
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        str(video_path),
        "-vf",
        "scale=-1:720",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def calculate_chunks(duration: float) -> list[tuple[int, float]]:
    chunks = []
    chunk_idx = 0
    current_start = 0.0
    while current_start < duration:
        chunks.append((chunk_idx, current_start))
        chunk_idx += 1
        current_start += CHUNK_DURATION - CHUNK_OVERLAP
    return chunks


def test_prompt_mode(args: argparse.Namespace) -> dict | None:
    print(f"\n{'=' * 60}\n🧪 TEST PROMPT MODE\n{'=' * 60}")
    print(f"Target Video: {TEST_VIDEO_PATH}")
    print(f"Segment: {TEST_START_TIME}s - {TEST_START_TIME + TEST_DURATION}s")

    if not TEST_VIDEO_PATH.exists():
        print(f"❌ Error: Test video not found at {TEST_VIDEO_PATH}")
        return

    # 1. Extract the test chunk
    test_chunk_path = Path("test_chunk_120s.mp4")
    print("✂️  Extracting chunk...")
    extract_chunk(TEST_VIDEO_PATH, TEST_START_TIME, TEST_DURATION, test_chunk_path)

    # 2. Upload (FIXED)
    print("📤 Uploading to Gemini...")
    client = genai.Client(api_key=args.api_key)
    video_file = client.files.upload(file=str(test_chunk_path))

    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(1)
        if video_file.name is None:
            break
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        print("❌ Upload failed.")
        return
    print(" Ready.")

    # 3. Generate (FIXED)
    print("🧠 Running Inference (this may take 30-60s)...")

    if video_file.uri is None:
        print("❌ Video file URI is None.")
        return

    response = client.models.generate_content(
        model=args.model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=video_file.uri, mime_type=video_file.mime_type
                    ),
                    types.Part.from_text(text=PROMPT_TEMPLATE),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0
        ),
    )

    # 4. Show Results
    if response.text is None:
        print("❌ Model returned no response.")
        return

    output = json.loads(response.text)

    result = {
        "video": str(TEST_VIDEO_PATH.name),
        "chunk_start": TEST_START_TIME,
        "chunk_duration": TEST_DURATION,
        "model": args.model,
        "annotations": output,
    }

    # 5. Cleanup
    if video_file.name is not None:
        client.files.delete(name=video_file.name)
    test_chunk_path.unlink()

    return result


def batch_mode(args: argparse.Namespace) -> dict | None:
    print(f"\n{'=' * 60}\n🔄 BATCH MODE (50% cost reduction)\n{'=' * 60}")

    temp_dir = Path(f"temp_batch_{args.video.stem}_{int(time.time())}")
    duration = get_video_duration(args.video)
    chunks = calculate_chunks(duration)
    chunk_files = []

    if args.dry_run:
        print(f"🧪 DRY RUN: Would extract {len(chunks)} chunks from {args.video.name}")
        print(f"   Model: {args.model}")
        print(f"   API Key: {args.api_key[:5]}... (Checked)")
        return

    print(f"📹 Extracting {len(chunks)} chunks...")
    temp_dir.mkdir(exist_ok=True)

    for idx, start in tqdm(chunks):
        p = temp_dir / f"chunk_{idx:03d}.mp4"
        try:
            extract_chunk(args.video, start, CHUNK_DURATION, p)
            chunk_files.append({"idx": idx, "start": start, "path": p})
        except Exception:
            continue

    # Initialize Client
    client = genai.Client(api_key=args.api_key)

    # Upload (FIXED)
    print(f"\n📤 Uploading {len(chunk_files)} chunks...")
    batch_requests = []

    for item in tqdm(chunk_files):
        video_file = client.files.upload(file=str(item["path"]))

        while video_file.state.name == "PROCESSING":
            time.sleep(1)
            if video_file.name is None:
                print(f"⚠️ Failed upload: chunk {item['idx']} has no name")
                break
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name != "ACTIVE":
            print(f"⚠️ Failed upload: chunk {item['idx']}")
            continue

        # Add to batch request list (FIXED STRUCTURE)
        batch_requests.append({
            "contents": [
                {
                    "parts": [
                        {
                            "file_data": {
                                "file_uri": video_file.uri,
                                "mime_type": video_file.mime_type,
                            }
                        },
                        {"text": PROMPT_TEMPLATE},
                    ],
                    "role": "user",
                }
            ],
            "config": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        })

    # Submit Batch (FIXED)
    print("\n🚀 Submitting Batch Job...")
    batch_job = client.batches.create(model=args.model, src=batch_requests)

    print(f"✅ Job Created: {batch_job.name}")
    print(f"   Status: {batch_job.state}")
    print("   The job runs on Google's servers. You can exit this script.")

    result = {
        "video": str(args.video.name),
        "batch_job_name": batch_job.name,
        "status": batch_job.state,
        "num_chunks": len(chunk_files),
        "model": args.model,
    }

    # Cleanup temp files
    for item in chunk_files:
        item["path"].unlink(missing_ok=True)
    temp_dir.rmdir()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=Path, help="Path to video file (required for batch mode)"
    )
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--test_prompt",
        action="store_true",
        help="Run hardcoded 60s test on GOPR_01.mp4",
    )

    args = parser.parse_args()

    result = None
    if args.test_prompt:
        result = test_prompt_mode(args)
    elif args.batch and args.video:
        result = batch_mode(args)
    else:
        print("Usage:")
        print("  Test:  python generate_ground_truth.py --test_prompt --api_key ...")
        print(
            "  Batch: python generate_ground_truth.py --video ... --batch --api_key ..."
        )
        return

    if result is not None:
        # Save output
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        output_file = OUTPUT_DIR / f"output_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n✅ Output saved to: {output_file}")
    else:
        print("error, result was none")


if __name__ == "__main__":
    main()

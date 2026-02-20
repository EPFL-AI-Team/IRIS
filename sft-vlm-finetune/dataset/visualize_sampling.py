import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .dataset_config import load_dataset_config, resolve_paths


def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize sampling + VLM Inference")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("dataset_config.yaml")
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument(
        "--video-id", type=str, help="Specific video ID (e.g. P03_06_01)"
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


# --- MODEL HANDLING ---
def load_model(model_id, device):
    print(f"Loading model: {model_id} on {device}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation="sdpa",  # if device == "cuda" else "sdpa",
    )
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
    )
    return model, processor


def run_inference(model, processor, frames_pil, prompt_text, device):
    """Runs Qwen inference on a list of PIL images."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames_pil,
                },  # Qwen utils handles PIL list as video
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True
    )[0]
    return output_text


# --- DATA HANDLING ---
def extract_frames(video_path, start_sec, duration, num_frames):
    if not video_path.exists():
        return [], []

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_f = int(start_sec * fps)
    end_f = int((start_sec + duration) * fps)

    # Sample indices
    indices = np.linspace(start_f, end_f, num_frames, dtype=int)
    indices = np.clip(indices, 0, total_frames - 1)

    vis_frames = []  # For Matplotlib (numpy)
    pil_frames = []  # For Model

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # RGB conversion
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. For Viz: Resize height to 200px
            h, w, _ = frame_rgb.shape
            scale = 200 / h
            vis_frame = cv2.resize(frame_rgb, (int(w * scale), 200))
            vis_frames.append(vis_frame)

            # 2. For Model: Keep original resolution (or reasonable scale)
            pil_frames.append(Image.fromarray(frame_rgb))
        else:
            # Pad black if fail
            black = np.zeros((200, 300, 3), dtype=np.uint8)
            vis_frames.append(black)
            pil_frames.append(Image.fromarray(black))

    cap.release()
    return vis_frames, pil_frames


def main():
    args = parse_arguments()

    # 1. Setup Config & Paths
    config = load_dataset_config(args.config)
    resolved, _ = resolve_paths(config, profile_name=args.profile, split_name=None)
    csv_path = resolved.consolidated_csv
    videos_dir = resolved.profile.videos_dir

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    # 2. Pick Sample
    df = pd.read_csv(csv_path)
    if args.video_id:
        subset = df[df["video_id"] == args.video_id]
        if subset.empty:
            print("Video ID not found.")
            return
        sample = subset.sample(1).iloc[0]
    else:
        # Pick interesting action
        sample = (
            df[
                (df["verb"].isin(["insert", "detach", "open"]))
                & (df["manipulated_object"] != "nan")
            ]
            .sample(1)
            .iloc[0]
        )

    vid_id = sample["video_id"]
    start = sample["start_sec"]
    verb = sample["verb"]
    obj = sample["manipulated_object"]
    video_path = videos_dir / f"{vid_id}.mp4"

    print(f"\nAnalyzing: {vid_id}")
    print(f"Ground Truth: {verb} {obj} (Start: {start}s)")

    # 3. Load Model
    model, processor = load_model(args.model_id, args.device)

    # 4. Scenarios
    scenarios = [
        (1.0, 4, "Short (1s) - 4 Frames"),
        (1.0, 8, "Short (1s) - 8 Frames"),
        (2.5, 4, "Long (2.5s) - 4 Frames"),
        (2.5, 8, "Long (2.5s) - 8 Frames"),
    ]

    # Prompt for Base Model
    prompt = (
        "Describe the specific atomic action performed by the hands in this video clip."
    )

    # 5. Run Loop
    plt.figure(figsize=(24, 14))
    plt.suptitle(
        f"Base Model Prediction Check: {vid_id}\nGT: {verb} {obj}", fontsize=16
    )

    for i, (dur, n_frames, label) in enumerate(scenarios):
        print(f"Processing scenario: {label}...")

        # Extract
        vis_frames, pil_frames = extract_frames(video_path, start, dur, n_frames)

        # Infer
        output = run_inference(model, processor, pil_frames, prompt, args.device)
        print(f" -> Prediction: {output}")

        # Visualize
        strip = np.hstack(vis_frames)

        ax = plt.subplot(4, 1, i + 1)
        ax.imshow(strip)
        ax.set_title(
            f"{label}\nModel Output: {output}", fontsize=10, loc="left", wrap=True
        )
        ax.axis("off")

    # Save
    out_file = resolved.profile.output_dir / f"inference_viz_{vid_id}.jpg"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved analysis to: {out_file}")


if __name__ == "__main__":
    main()

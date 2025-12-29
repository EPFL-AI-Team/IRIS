import json
import logging
import os
from typing import Any

import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --- CONFIGURATION ---
ADAPTER_PATH = "/scratch/iris/checkpoints/qwen3b_finebio_run4"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_PATH = "/scratch/iris/test_videos/colony-counting-test-video.mp4"

SEGMENT_DURATION = 2.0  # T in seconds
NUM_FRAMES = 4  # s samples
FRAME_OVERLAP = 1  # k frames overlap between segments
MAX_NEW_TOKENS = 2048
# ----------------------

PROMPT = "Analyze this laboratory procedure clip. The context is colony counting, and the researcher is counting colonies on a petri dish. You need to analyze the specifics in these frames. Return JSON with: visual_analysis (describe what you see), verb (action type), tool (manipulated object), target (affected object or null), context (protocol step)."

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def get_inference_segments(
    video_path: str, t_sec: float, s_samples: int, k_overlap: int
) -> list[dict[str, Any]]:
    """Extracts segments from video and converts frames to PIL Images."""
    logging.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_segment = int(t_sec * fps)

    step = frames_per_segment - k_overlap
    segments = []

    logging.info(
        f"Video FPS: {fps}, Total frames: {total_frames}, Frames per segment: {frames_per_segment}, Step: {step}"
    )

    for seg_idx, start_idx in enumerate(
        range(0, total_frames - frames_per_segment + 1, step)
    ):
        segment_frames = []
        indices = [
            start_idx + int(i * (frames_per_segment - 1) / (s_samples - 1))
            for i in range(s_samples)
        ]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert NumPy array to PIL Image (Critical for Qwen Processor)
                pil_image = Image.fromarray(rgb_frame)
                segment_frames.append(pil_image)
            else:
                logging.warning(f"Failed to read frame at index {idx}")

        if len(segment_frames) == s_samples:
            segments.append({"start_sec": start_idx / fps, "frames": segment_frames})
        else:
            logging.debug(f"Segment {seg_idx} skipped due to insufficient frames.")

    cap.release()
    logging.info(
        f"Generated {len(segments)} segments with {s_samples} frames each (T={t_sec}s, k={k_overlap})."
    )
    return segments


def inference_single_segment(
    model, processor, frames: list[Image.Image], prompt: str
) -> str:
    """
    Runs inference on a single list of PIL images (one video segment).
    Mimics VideoJob._inference_qwen logic.
    """
    # Calculate effective FPS for the model context
    sample_fps = NUM_FRAMES / SEGMENT_DURATION

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "sample_fps": sample_fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs using Qwen utils
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Move inputs to device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Decode (trimming the input tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def run_eye_test() -> None:
    logging.info("Loading model and adapters...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="auto"
    )
    logging.info("Base model loaded.")

    # model.load_adapter(ADAPTER_PATH)
    # logging.info(f"Adapter loaded from {ADAPTER_PATH}.")

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    logging.info("Processor loaded.")

    # Prepare output file paths in adapter path under 'chuv_video_evaluation'
    video_base = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    output_dir = os.path.join(ADAPTER_PATH, "chuv_video_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{video_base}_inference_output.jsonl")
    txt_path = os.path.join(output_dir, f"{video_base}_inference_output.txt")

    logging.info(f"Output files will be: {jsonl_path}, {txt_path}")

    # 1. Get all segments (images loaded into memory)
    segments = get_inference_segments(
        VIDEO_PATH, SEGMENT_DURATION, NUM_FRAMES, FRAME_OVERLAP
    )

    logging.info(f"Starting sequential inference on {len(segments)} segments...")

    prompt = PROMPT

    # Clear files to start fresh
    with open(jsonl_path, "w", encoding="utf-8") as f:
        pass
    with open(txt_path, "w", encoding="utf-8") as f:
        pass

    # 2. Iterate and process sequentially
    for i, seg in enumerate(segments):
        start_sec = seg["start_sec"]
        frames = seg["frames"]

        logging.info(
            f"Processing segment {i + 1}/{len(segments)} at {start_sec:.2f}s..."
        )

        try:
            response = inference_single_segment(model, processor, frames, prompt)
        except Exception as e:
            logging.error(f"Error processing segment {i}: {e}")
            response = json.dumps({"error": str(e)})

        # 3. Write results immediately (append mode)
        with open(jsonl_path, "a", encoding="utf-8") as jf:
            jsonl_obj = {"start_sec": start_sec, "response": response}
            jf.write(json.dumps(jsonl_obj, ensure_ascii=False) + "\n")

        with open(txt_path, "a", encoding="utf-8") as tf:
            tf.write(f"--- Segment Start: {start_sec:.2f}s ---\n{response}\n\n")

        print(f"--- Segment Start: {start_sec:.2f}s ---")
        print(response)

    logging.info("All segments processed successfully.")


if __name__ == "__main__":
    run_eye_test()

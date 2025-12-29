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
ADAPTER_PATH = "/scratch/iris/checkpoints/qwen3b_finebio_run2"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_PATH = "/scratch/iris/test_videos/colony-counting-test-video.mp4"

SEGMENT_DURATION = 2.0  # T in seconds
NUM_FRAMES = 4  # s samples
FRAME_OVERLAP = 1  # k frames overlap
MAX_NEW_TOKENS = 256
BATCH_SIZE = 4  # Process N segments at a time to avoid OOM
# ----------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def setup_model() -> tuple[Any, Any]:
    """Loads the model, applies adapter, and loads the processor."""
    logging.info("Loading model and adapters...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="auto"
    )
    logging.info("Base model loaded.")

    model.load_adapter(ADAPTER_PATH)
    logging.info(f"Adapter loaded from {ADAPTER_PATH}.")

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    logging.info("Processor loaded.")
    return model, processor


def extract_video_segments(
    video_path: str, t_sec: float, s_samples: int, k_overlap: int
) -> list[dict[str, Any]]:
    """Opens video, extracts frames for all segments, converts to PIL."""
    logging.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_segment = int(t_sec * fps)
    step = frames_per_segment - k_overlap

    segments = []
    logging.info(f"Video FPS: {fps:.2f}, Total frames: {total_frames}")

    # Generate indices
    for start_idx in range(0, total_frames - frames_per_segment + 1, step):
        segment_frames = []
        indices = [
            start_idx + int(i * (frames_per_segment - 1) / (s_samples - 1))
            for i in range(s_samples)
        ]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                segment_frames.append(Image.fromarray(rgb_frame))
            else:
                logging.warning(f"Failed to read frame at index {idx}")

        if len(segment_frames) == s_samples:
            segments.append({"start_sec": start_idx / fps, "frames": segment_frames})

    cap.release()
    logging.info(f"Generated {len(segments)} segments.")
    return segments


def prepare_batch_inputs(
    processor, segments: list[dict], prompt: str
) -> tuple[dict, list[float]]:
    """
    Prepares the inputs for the model (tokenization + vision processing).
    CRITICAL FIX: Handles None values in image_inputs for pure video batches.
    """
    conversations = []
    start_times = []

    # 1. Construct Conversation Objects
    for seg in segments:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": seg["frames"],
                        "sample_fps": NUM_FRAMES / SEGMENT_DURATION,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        conversations.append(messages)
        start_times.append(seg["start_sec"])

    # 2. Apply Chat Template (Text)
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in conversations
    ]

    # 3. Process Vision Info (Extract Tensors)
    image_inputs = []
    video_inputs = []
    video_kwargs_list = []

    for msg in conversations:
        img_in, vid_in, vkwargs = process_vision_info(msg, return_video_kwargs=True)
        image_inputs.append(img_in)
        video_inputs.append(vid_in)
        video_kwargs_list.append(vkwargs)

    # 4. FIX: Filter out None values.
    # If a list contains ONLY None (e.g. pure video), pass None to processor.
    # Otherwise, the processor crashes trying to batch [None, None].
    final_images = None if all(x is None for x in image_inputs) else image_inputs
    final_videos = None if all(x is None for x in video_inputs) else video_inputs

    # Merge kwargs (grid sizes) - assuming consistent args across batch
    merged_kwargs = video_kwargs_list[0] if video_kwargs_list else {}

    # 5. Tokenize and Batch
    inputs = processor(
        text=texts,
        images=final_images,
        videos=final_videos,
        padding=True,
        return_tensors="pt",
        **merged_kwargs,
    )

    return inputs, start_times


def run_batch_inference(model, processor, inputs: dict) -> list[str]:
    """Runs the forward pass on a prepared batch."""
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Trim input tokens from output
    generated_ids = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]

    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def save_results(
    results: list[tuple[float, str]], video_path: str, mode: str = "w"
) -> None:
    """Writes results to JSONL and TXT files."""
    video_dir = os.path.dirname(video_path)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    jsonl_path = os.path.join(video_dir, f"{video_base}_inference_output.jsonl")
    txt_path = os.path.join(video_dir, f"{video_base}_inference_output.txt")

    with (
        open(jsonl_path, mode, encoding="utf-8") as jf,
        open(txt_path, mode, encoding="utf-8") as tf,
    ):
        for start_sec, response in results:
            # JSONL
            obj = {"start_sec": start_sec, "response": response}
            jf.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # TXT
            tf.write(f"--- Segment Start: {start_sec:.2f}s ---\n{response}\n\n")

            # Console
            print(f"--- Segment Start: {start_sec:.2f}s ---")
            print(response)


def main():
    # 1. Setup
    model, processor = setup_model()

    # 2. Extract Data
    all_segments = extract_video_segments(
        VIDEO_PATH, SEGMENT_DURATION, NUM_FRAMES, FRAME_OVERLAP
    )

    # Initialize files (clear them)
    save_results([], VIDEO_PATH, mode="w")

    prompt = (
        "Describe the lab action in this segment in JSON format with visual reasoning."
    )

    # 3. Batch Processing Loop
    total_segments = len(all_segments)
    logging.info(f"Processing {total_segments} segments in batches of {BATCH_SIZE}...")

    for i in range(0, total_segments, BATCH_SIZE):
        batch_segments = all_segments[i : i + BATCH_SIZE]
        logging.info(
            f"Batch {i // BATCH_SIZE + 1}: Processing segments {i} to {i + len(batch_segments) - 1}"
        )

        try:
            # Prepare
            inputs, start_times = prepare_batch_inputs(
                processor, batch_segments, prompt
            )

            # Infer
            responses = run_batch_inference(model, processor, inputs)

            # Save (Append mode)
            batch_results = list(zip(start_times, responses, strict=True))
            save_results(batch_results, VIDEO_PATH, mode="a")

        except Exception as e:
            logging.error(f"Failed to process batch starting at index {i}: {e}")
            import traceback

            traceback.print_exc()

    logging.info("Inference complete.")


if __name__ == "__main__":
    main()

"""Benchmark VLM inference performance."""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import os
import cv2
import numpy as np
import torch
from PIL import Image

from iris.vlm.models import load_model_and_processor

os.environ["TRANSFORMERS_VIDEO_BACKEND"] = "torchvision"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    """Load all frames from video."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    logger.info(f"Loaded {len(frames)} frames from {video_path.name}")
    return frames


def sample_consecutive_frames(
    frames: list[np.ndarray], 
    num_frames: int,
) -> list[Image.Image]:
    """Sample N consecutive frames from random start point."""
    if num_frames >= len(frames):
        # Use all frames
        start_idx = 0
        sampled = frames
    else:
        # Random start, then consecutive
        start_idx = random.randint(0, len(frames) - num_frames)
        sampled = frames[start_idx:start_idx + num_frames]
    
    # Convert to PIL Images
    return [Image.fromarray(f) for f in sampled]


def run_qwen_inference(
    model: Any,
    processor: Any,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 128,
    fps: float = 5.0,
) -> str:
    """Qwen inference logic (from your VideoJob)."""
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "qwen_vl_utils required for Qwen models"
        ) from exc
    
    with torch.no_grad():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "sample_fps": fps},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_dtype = getattr(model, "dtype", None)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **(video_kwargs or {}),
        ).to(device=model.device, dtype=model_dtype)
        
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        result = processor.decode(generated_ids, skip_special_tokens=True)
    
    return result


def run_smolvlm_inference(
    model: Any,
    processor: Any,
    frames: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 128,
    fps: float = 5.0,
) -> str:
    """SmolVLM inference with VideoMetadata."""
    from transformers.video_utils import VideoMetadata
    
    # Create metadata for our pre-sampled frames
    video_metadata = VideoMetadata(
        total_num_frames=len(frames),
        fps=fps,
        duration=len(frames) / fps
    )
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": prompt},
        ],
    }]
    
    with torch.no_grad():
        inputs = processor.apply_chat_template(
            messages,
            video_metadata=[video_metadata],  # Add metadata
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device=model.device, dtype=torch.float16)
        
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return result


def is_qwen_model(model: Any) -> bool:
    """Check if model is Qwen."""
    model_type = getattr(model.config, "model_type", "").lower()
    return "qwen" in model_type


def run_inference(
    model: Any,
    processor: Any,
    frames: list[Image.Image],
    prompt: str,
    fps: float = 5.0,
) -> tuple[str, float]:
    """Run inference and return (result, time_taken)."""
    start = time.perf_counter()
    
    if is_qwen_model(model):
        result = run_qwen_inference(model, processor, frames, prompt, fps=fps)
    else:
        result = run_smolvlm_inference(model, processor, frames, prompt, fps=fps)
    
    elapsed = time.perf_counter() - start
    return result, elapsed


def benchmark_model(
    model_key: str,
    video_paths: list[Path],
    frame_counts: list[int],
    runs_per_video: int = 3,
    hardware: str | None = None,
    prompt: str = "Describe what is happening in this video.",
) -> list[dict]:
    """Benchmark a model across videos and frame counts."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_key}")
    logger.info(f"{'='*60}")
    
    # Load model once
    model, processor = load_model_and_processor(model_key, hardware=hardware)
    logger.info(f"Model loaded on {model.device}")
    
    # Load all videos
    all_video_frames = {}
    for video_path in video_paths:
        all_video_frames[video_path.name] = load_video_frames(video_path)
    
    logger.info("Running warmup...")
    warmup_frames = sample_consecutive_frames(
        list(all_video_frames.values())[0], 4
    )
    try:
        run_inference(model, processor, warmup_frames, prompt)
        logger.info("Warmup complete")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
    
    results = []
    
    for num_frames in frame_counts:
        logger.info(f"\n--- Testing buffer_size={num_frames} ---")
        
        frame_results = []
        
        # Run on each video
        for video_name, frames in all_video_frames.items():
            logger.info(f"  Video: {video_name} ({runs_per_video} runs)")
            
            for run_idx in range(runs_per_video):
                # Sample consecutive frames from random start
                sampled_frames = sample_consecutive_frames(frames, num_frames)
                
                try:
                    result, elapsed = run_inference(
                        model, processor, sampled_frames, prompt, fps=5.0
                    )
                    
                    frame_results.append({
                        "video": video_name,
                        "run": run_idx,
                        "time": elapsed,
                        "fps": num_frames / elapsed,
                    })
                    
                    logger.info(
                        f"    Run {run_idx+1}: {elapsed:.3f}s "
                        f"({num_frames/elapsed:.2f} fps)"
                    )
                    
                except Exception as e:
                    logger.error(f"    Run {run_idx+1} failed: {e}")
                    continue
        
        if not frame_results:
            logger.warning(f"  All runs failed for {num_frames} frames")
            continue
        
        # Aggregate stats
        times = [r["time"] for r in frame_results]
        result_dict = {
            "buffer_size": num_frames,
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "mean_fps": float(np.mean([r["fps"] for r in frame_results])),
            "total_runs": len(times),
            "raw_results": frame_results,
        }
        results.append(result_dict)
        
        logger.info(
            f"  Summary: {result_dict['mean_time']:.3f}s ± "
            f"{result_dict['std_time']:.3f}s ({result_dict['mean_fps']:.2f} fps)"
        )
    
    return results


def main():
    # Configuration
    VIDEO_DIR = Path("src/iris/client/web/static/videos")
    OUTPUT_DIR = Path("benchmarks")
    
    videos = [
        VIDEO_DIR / "chuv-test-video.mp4",
        VIDEO_DIR / "dataset-test-video.mp4",
    ]
    
    models = [
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", 
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
    ]
    frame_counts = [2, 4, 8, 12, 16, 32, 64]
    runs_per_video = 3
    hardware = "v100"  # or None, or "v100"
    
    # Validate videos exist
    for video in videos:
        if not video.exists():
            logger.error(f"Video not found: {video}")
            return
    
    # Create output dir
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    
    for model_key in models:
        try:
            results = benchmark_model(
                model_key=model_key,
                video_paths=videos,
                frame_counts=frame_counts,
                runs_per_video=runs_per_video,
                hardware=hardware,
            )
            all_results[model_key] = results
            
            # Save individual model results
            # Save individual model results
            safe_model_name = model_key.replace("/", "_")
            output_file = OUTPUT_DIR / f"{safe_model_name}_benchmark.json"
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "model": model_key,
                        "hardware": hardware,
                        "videos": [str(v) for v in videos],
                        "runs_per_video": runs_per_video,
                        "results": results,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"\nSaved: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to benchmark {model_key}: {e}", exc_info=True)
    
    # Save combined
    combined_file = OUTPUT_DIR / "combined_benchmark.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking complete! Results in {OUTPUT_DIR}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

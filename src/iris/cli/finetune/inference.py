"""Video inference pipeline with layered architecture.

This module provides three levels of abstraction:
- Layer 1: infer_segment() - Single segment inference (atomic)
- Layer 2: infer_video() - Whole video inference with one model
- Layer 3: compare_models() - Compare base vs fine-tuned models

Can be used as CLI or imported as module.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from iris.utils.logging import setup_logger

logger = setup_logger(__name__)

# --- DEFAULTS ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

DEFAULT_PROMPT = (
    "Analyze this laboratory procedure clip. Return JSON with: "
    "visual_analysis (describe what you see), verb (action type), "
    "tool (manipulated object), target (affected object or null), "
    "context (protocol step)."
)


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class VideoInferenceConfig:
    """Configuration for video inference."""

    segment_duration: float = 2.0
    num_frames: int = 4
    frame_overlap: int = 1
    max_new_tokens: int = 512
    prompt: str = DEFAULT_PROMPT


@dataclass
class ComparisonConfig:
    """Configuration for model comparison."""

    video_path: str | Path
    output_dir: str | Path
    checkpoint_dir: str | Path | None = None
    run_base: bool = True
    run_finetuned: bool = True
    backend: str = "qwen"  # "qwen", "gemini", "openai"
    api_key: str | None = None  # For external APIs
    video_config: VideoInferenceConfig = field(default_factory=VideoInferenceConfig)


# =============================================================================
# Model Loading
# =============================================================================


def load_model(
    model_path: str | Path,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load Qwen model and processor from path."""
    model_path = str(model_path)
    logger.info(f"Loading model from {model_path}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()

    logger.info("Model loaded successfully")
    return model, processor


# =============================================================================
# Video Processing
# =============================================================================


def extract_video_segments(
    video_path: str | Path,
    config: VideoInferenceConfig,
) -> list[dict[str, Any]]:
    """Extract segments from video as PIL Images.

    Returns:
        List of {"start_sec": float, "frames": list[Image.Image]}
    """
    video_path = str(video_path)
    logger.info(f"Extracting segments from {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_segment = int(config.segment_duration * fps)
    step = frames_per_segment - config.frame_overlap

    logger.info(
        f"Video: {fps:.1f} FPS, {total_frames} frames, "
        f"{frames_per_segment} frames/segment, step={step}"
    )

    segments = []
    for start_idx in range(0, total_frames - frames_per_segment + 1, step):
        # Sample frames evenly within segment
        indices = [
            start_idx + int(i * (frames_per_segment - 1) / (config.num_frames - 1))
            for i in range(config.num_frames)
        ]

        segment_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                segment_frames.append(pil_image)

        if len(segment_frames) == config.num_frames:
            segments.append({
                "start_sec": start_idx / fps,
                "frames": segment_frames,
            })

    cap.release()
    logger.info(f"Extracted {len(segments)} segments")
    return segments


# =============================================================================
# Layer 1: Segment-level Inference (Atomic)
# =============================================================================


def infer_segment(
    frames: list[Image.Image],
    model: Any,
    processor: Any,
    prompt: str,
    max_new_tokens: int = 512,
    num_frames: int = 4,
    segment_duration: float = 2.0,
) -> str:
    """Run inference on a single video segment (list of frames).

    This is the atomic inference unit. Used by infer_video().

    Args:
        frames: List of PIL Images for this segment
        model: Loaded Qwen model
        processor: Loaded processor
        prompt: Text prompt for the model
        max_new_tokens: Maximum tokens to generate
        num_frames: Number of frames (for FPS calculation)
        segment_duration: Duration in seconds (for FPS calculation)

    Returns:
        Model response as string
    """
    sample_fps = num_frames / segment_duration

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": sample_fps},
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

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only generated tokens
    generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, output_ids, strict=True)
    ]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# =============================================================================
# Layer 1 Alternative: External API Backends (Placeholders)
# =============================================================================


def infer_segment_gemini(
    frames: list[Image.Image],
    prompt: str,
    api_key: str,
    max_new_tokens: int = 512,
) -> str:
    """Gemini API inference (placeholder).

    TODO: Implement with google.generativeai
    """
    raise NotImplementedError(
        "Gemini backend not implemented. "
        "Install google-generativeai and implement with gemini-1.5-flash or gemini-1.5-pro"
    )


def infer_segment_openai(
    frames: list[Image.Image],
    prompt: str,
    api_key: str,
    max_new_tokens: int = 512,
) -> str:
    """OpenAI API inference (placeholder).

    TODO: Implement with openai client (gpt-4-vision)
    """
    raise NotImplementedError(
        "OpenAI backend not implemented. "
        "Install openai and implement with gpt-4-vision-preview"
    )


# =============================================================================
# Layer 2: Video-level Inference
# =============================================================================


def infer_video(
    video_path: str | Path,
    model: Any,
    processor: Any,
    config: VideoInferenceConfig | None = None,
) -> list[dict]:
    """Run inference on entire video, returning results per segment.

    Args:
        video_path: Path to video file
        model: Loaded model
        processor: Loaded processor
        config: Video inference configuration (uses defaults if None)

    Returns:
        List of {"start_sec": float, "response": str} dicts
    """
    config = config or VideoInferenceConfig()
    segments = extract_video_segments(video_path, config)

    results = []
    for i, seg in enumerate(segments):
        logger.info(f"Processing segment {i + 1}/{len(segments)} at {seg['start_sec']:.2f}s")

        try:
            response = infer_segment(
                frames=seg["frames"],
                model=model,
                processor=processor,
                prompt=config.prompt,
                max_new_tokens=config.max_new_tokens,
                num_frames=config.num_frames,
                segment_duration=config.segment_duration,
            )
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")
            response = json.dumps({"error": str(e)})

        results.append({
            "start_sec": seg["start_sec"],
            "response": response,
        })

    return results


# =============================================================================
# Layer 3: Model Comparison
# =============================================================================


def compare_models(config: ComparisonConfig) -> dict[str, list[dict]]:
    """Compare base vs fine-tuned model on a video.

    Args:
        config: Comparison configuration

    Returns:
        Dict with keys "base" and/or "finetuned", each containing
        list of {"start_sec": float, "response": str} dicts
    """
    results = {}

    # Run base model
    if config.run_base:
        logger.info("Running base model inference...")
        model, processor = load_model(BASE_MODEL_NAME)
        results["base"] = infer_video(
            config.video_path, model, processor, config.video_config
        )
        del model
        torch.cuda.empty_cache()
        logger.info("Base model complete, VRAM freed")

    # Run fine-tuned model
    if config.run_finetuned and config.checkpoint_dir:
        logger.info("Running fine-tuned model inference...")
        model, processor = load_model(config.checkpoint_dir)
        results["finetuned"] = infer_video(
            config.video_path, model, processor, config.video_config
        )
        del model
        torch.cuda.empty_cache()
        logger.info("Fine-tuned model complete")

    # Save outputs
    save_comparison_outputs(results, config)

    return results


# =============================================================================
# Output Saving
# =============================================================================


def save_comparison_outputs(
    results: dict[str, list[dict]],
    config: ComparisonConfig,
) -> None:
    """Save JSONL and TXT outputs for each model variant."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = Path(config.video_path).stem

    for model_name, segments in results.items():
        # JSONL (machine readable)
        jsonl_path = output_dir / f"{video_name}_{model_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

        # TXT (human readable)
        txt_path = output_dir / f"{video_name}_{model_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Video: {config.video_path}\n")
            f.write(f"Prompt: {config.video_config.prompt}\n")
            f.write("=" * 80 + "\n\n")

            for seg in segments:
                f.write(f"--- {seg['start_sec']:.2f}s ---\n")
                f.write(seg["response"] + "\n\n")

        logger.info(f"Saved {model_name}: {jsonl_path}, {txt_path}")

    logger.info(f"All outputs saved to {output_dir}")


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VLM inference on video (compare base vs fine-tuned)"
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to video file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned checkpoint path (required unless --base-only)",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only run base model",
    )
    parser.add_argument(
        "--finetuned-only",
        action="store_true",
        help="Only run fine-tuned model (requires --checkpoint)",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=2.0,
        help="Segment duration in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Frames per segment (default: 4)",
    )
    parser.add_argument(
        "--frame-overlap",
        type=int,
        default=1,
        help="Frame overlap between segments (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (uses default if not set)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Validate args
    if args.finetuned_only and not args.checkpoint:
        raise ValueError("--finetuned-only requires --checkpoint")

    # Build config
    video_config = VideoInferenceConfig(
        segment_duration=args.segment_duration,
        num_frames=args.num_frames,
        frame_overlap=args.frame_overlap,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt or DEFAULT_PROMPT,
    )

    config = ComparisonConfig(
        video_path=args.video,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint,
        run_base=not args.finetuned_only,
        run_finetuned=not args.base_only and args.checkpoint is not None,
        video_config=video_config,
    )

    # Run comparison
    results = compare_models(config)

    # Summary
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    for model_name, segments in results.items():
        print(f"  {model_name}: {len(segments)} segments")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

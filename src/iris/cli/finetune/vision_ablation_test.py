"""Vision Ablation Test: Verify if the model actually uses visual information.

This script runs inference twice:
1. With normal images (baseline)
2. With zeroed/blank pixel_values (ablation)

If predictions are similar, the model isn't using vision effectively.
If predictions differ significantly, the model IS using visual information.
"""

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from iris.utils.logging import setup_logger

logger = setup_logger(__name__)

# Use the same prompt as training
PROMPT = (
    "Analyze this laboratory procedure clip. Return JSON with: "
    "visual_analysis (describe what you see), verb (action type), "
    "tool (manipulated object), target (affected object or null), "
    "context (protocol step)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision ablation test")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("/scratch/iris/checkpoints/qwen3b_finebio_run7"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        default=Path(
            "/scratch/iris/finebio_processed/splits/train_10k_4_frames_v2/finebio_test.jsonl"
        ),
        help="Path to validation JSONL",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Number of samples to test (default 20 for quick test)",
    )
    parser.add_argument(
        "--ablation_mode",
        choices=["zero", "random", "shuffle"],
        default="zero",
        help="How to ablate vision: zero (blank), random (noise), shuffle (wrong images)",
    )
    return parser.parse_args()


def load_model_and_processor(checkpoint_dir: Path):
    """Load model and processor."""
    logger.info(f"Loading model from {checkpoint_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(str(checkpoint_dir))
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def load_samples(val_path: Path, max_samples: int) -> list[dict]:
    """Load validation samples."""
    samples = []
    with open(val_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def run_single_inference(
    model,
    processor,
    sample: dict,
    ablate_vision: bool = False,
    ablation_mode: str = "zero",
) -> str:
    """Run inference on a single sample, optionally ablating vision."""
    # Extract user message
    user_msg = next((m for m in sample["messages"] if m["role"] == "user"), None)
    if not user_msg:
        return ""

    # Modify message to use our standard prompt
    modified_msg = {"role": "user", "content": []}
    for item in user_msg["content"]:
        if item.get("type") == "image":
            modified_msg["content"].append(item)
        elif item.get("type") == "text":
            modified_msg["content"].append({"type": "text", "text": PROMPT})

    messages = [modified_msg]

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ABLATION: Modify pixel_values
    if ablate_vision and "pixel_values" in inputs:
        original_shape = inputs["pixel_values"].shape
        original_dtype = inputs["pixel_values"].dtype

        if ablation_mode == "zero":
            # Zero out all visual information
            inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        elif ablation_mode == "random":
            # Random noise (same distribution as normalized images)
            inputs["pixel_values"] = torch.randn_like(inputs["pixel_values"])
        elif ablation_mode == "shuffle":
            # Shuffle pixels within each image (destroys structure but keeps statistics)
            flat = inputs["pixel_values"].flatten()
            shuffled_indices = torch.randperm(flat.size(0), device=flat.device)
            inputs["pixel_values"] = flat[shuffled_indices].view(original_shape)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode only the generated part
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0, input_len:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)

    return response.strip()


def parse_json_response(response: str) -> dict:
    """Try to parse JSON from response."""
    try:
        # Find JSON in response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    return {}


def compare_predictions(normal: dict, ablated: dict) -> dict:
    """Compare normal vs ablated predictions."""
    fields = ["verb", "tool", "target", "context", "visual_analysis"]
    comparison = {}

    for field in fields:
        normal_val = str(normal.get(field, "")).lower().strip()
        ablated_val = str(ablated.get(field, "")).lower().strip()
        comparison[field] = {
            "normal": normal_val,
            "ablated": ablated_val,
            "same": normal_val == ablated_val,
        }

    return comparison


def main():
    args = parse_args()

    # Load model
    model, processor = load_model_and_processor(args.checkpoint_dir)

    # Load samples
    samples = load_samples(args.val_path, args.max_samples)
    logger.info(f"Loaded {len(samples)} samples for ablation test")

    # Run comparison
    results = []
    same_counts = {
        "verb": 0,
        "tool": 0,
        "target": 0,
        "context": 0,
        "visual_analysis": 0,
    }

    print("\n" + "=" * 80)
    print(f"VISION ABLATION TEST (mode: {args.ablation_mode})")
    print("=" * 80)
    print("\nRunning inference with normal images vs ablated (zeroed) images...\n")

    for i, sample in enumerate(tqdm(samples, desc="Testing")):
        # Run with normal images
        normal_response = run_single_inference(
            model, processor, sample, ablate_vision=False
        )
        normal_parsed = parse_json_response(normal_response)

        # Run with ablated images
        ablated_response = run_single_inference(
            model,
            processor,
            sample,
            ablate_vision=True,
            ablation_mode=args.ablation_mode,
        )
        ablated_parsed = parse_json_response(ablated_response)

        # Compare
        comparison = compare_predictions(normal_parsed, ablated_parsed)
        results.append({
            "sample_idx": i,
            "normal_response": normal_response[:200],  # Truncate for display
            "ablated_response": ablated_response[:200],
            "comparison": comparison,
        })

        # Count matches
        for field in same_counts:
            if comparison.get(field, {}).get("same", False):
                same_counts[field] += 1

    # Print detailed results for first few samples
    print("\n" + "-" * 80)
    print("SAMPLE-BY-SAMPLE COMPARISON (first 5)")
    print("-" * 80)

    for result in results[:5]:
        print(f"\n[Sample {result['sample_idx']}]")
        print(f"  Normal:  {result['normal_response'][:100]}...")
        print(f"  Ablated: {result['ablated_response'][:100]}...")
        print("  Field comparison:")
        for field, data in result["comparison"].items():
            status = "SAME" if data["same"] else "DIFF"
            print(
                f"    {field:18s}: [{status}] '{data['normal'][:30]}' vs '{data['ablated'][:30]}'"
            )

    # Summary statistics
    n = len(samples)
    print("\n" + "=" * 80)
    print("SUMMARY: How often did ablated predictions match normal predictions?")
    print("=" * 80)
    print(f"\n{'Field':<20} {'Same':>10} {'Different':>10} {'Same %':>10}")
    print("-" * 50)

    for field in ["verb", "tool", "target", "context", "visual_analysis"]:
        same = same_counts[field]
        diff = n - same
        pct = (same / n) * 100
        print(f"{field:<20} {same:>10} {diff:>10} {pct:>9.1f}%")

    # Overall interpretation
    avg_same = sum(same_counts.values()) / (len(same_counts) * n) * 100

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if avg_same > 80:
        print(f"""
Average field match: {avg_same:.1f}%

WARNING: Predictions are very similar with and without images!
This suggests the model is NOT effectively using visual information.

Possible causes:
1. Model is memorizing text patterns from prompts/training data
2. Visual features are not flowing properly to language model
3. Dataset has low visual diversity (similar images → similar outputs)

Recommended actions:
1. Check if visual tokens are actually being attended to
2. Try training with higher learning rate for vision encoder
3. Verify image preprocessing is correct
4. Consider unfreezing vision encoder layers
""")
    elif avg_same > 50:
        print(f"""
Average field match: {avg_same:.1f}%

MIXED: Model shows some visual dependence but also significant text bias.

The model is partially using visual information, but there's room for improvement.

Recommended actions:
1. Check which fields are most/least visually dependent
2. Consider targeted improvements for low-dependence fields
3. May benefit from more epochs or data augmentation
""")
    else:
        print(f"""
Average field match: {avg_same:.1f}%

GOOD: Predictions differ significantly with vs without images!
The model IS using visual information to make predictions.

This rules out the hypothesis that the model is purely pattern-matching.
Low accuracy may be due to task difficulty rather than vision neglect.
""")

    # Save detailed results
    output_path = args.checkpoint_dir / "vision_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "ablation_mode": args.ablation_mode,
                "n_samples": n,
                "same_counts": same_counts,
                "avg_same_pct": avg_same,
                "detailed_results": results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

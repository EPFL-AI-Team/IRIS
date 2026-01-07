"""Evaluate Qwen2.5-VL fine-tuning results on laboratory action recognition.

This script loads a trained model, runs inference on the validation/test set,
and generates comprehensive metrics and visualizations for thesis reporting.

Features:
- Compare fine-tuned model vs base model (--compare-base)
- Test prompts with/without visual_analysis (--prompt-mode)
- Comprehensive per-field and aggregate metrics

Can be used as CLI or imported as module.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from iris.utils.logging import setup_logger

logger = setup_logger(__name__)

# Set visualization style
sns.set_theme(style="whitegrid", palette="muted")

# Prompt templates
PROMPT_WITH_VISUAL = (
    "Analyze this laboratory procedure clip. Return JSON with: "
    "visual_analysis (describe what you see), verb (action type), "
    "tool (manipulated object), target (affected object or null), "
    "context (protocol step)."
)

PROMPT_WITHOUT_VISUAL = (
    "Analyze this laboratory procedure clip. Return JSON with: "
    "verb (action type), tool (manipulated object), "
    "target (affected object or null), context (protocol step)."
)

# Base model identifier
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


# =============================================================================
# Configuration Dataclass (for module import)
# =============================================================================


@dataclass
class EvaluationConfig:
    """Configuration for running evaluation.

    Use this when importing as a module instead of CLI.
    """

    checkpoint_dir: Path
    val_path: Path
    batch_size: int = 8
    max_samples: int = 100  # 0 for all
    compare_base: bool = False
    prompt_mode: str = "with_visual"  # "with_visual", "without_visual", "both"
    eval_name: str | None = None


# =============================================================================
# CLI Argument Parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VLM fine-tuning results")
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("/scratch/iris/checkpoints/qwen3b_finebio_run7/checkpoint-400"),
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        default=Path(
            "/scratch/iris/finebio_processed/splits/train_without_vis_analysis/finebio_test.jsonl"
        ),
        help="Path to validation/test JSONL file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate (0 for all)",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate base model for comparison",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["with_visual", "without_visual", "both"],
        default="with_visual",
        help="Prompt mode: with/without visual_analysis request",
    )
    parser.add_argument(
        "--eval-name",
        type=str,
        default=None,
        help="Name for this evaluation run (creates subdirectory). If not specified, uses 'evaluation'.",
    )
    return parser.parse_args()


def load_model_and_processor(
    checkpoint_dir: Path | str,
    is_base_model: bool = False,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load model and processor from checkpoint or base model."""
    model_path = (
        str(checkpoint_dir) if isinstance(checkpoint_dir, Path) else checkpoint_dir
    )
    logger.info(
        f"Loading {'base' if is_base_model else 'fine-tuned'} model from {model_path}"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Use left padding for decoder-only generation
    processor.tokenizer.padding_side = "left"

    model.eval()
    logger.info("Model loaded successfully")
    return model, processor


def load_base_model() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """Load the base Qwen model (not fine-tuned) for comparison."""
    return load_model_and_processor(BASE_MODEL_NAME, is_base_model=True)


def load_validation_data(val_path: Path, max_samples: int | None = None) -> list[dict]:
    """Load validation data from JSONL file."""
    logger.info(f"Loading validation data from {val_path}")

    samples = []
    with open(val_path) as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def normalize_value(val: str | None) -> str:
    """Normalize values for comparison (handles nan/null/none equivalence)."""
    if val is None:
        return "none"
    val = str(val).lower().strip()
    if val in ("nan", "null", "none", ""):
        return "none"
    return val


def token_f1(pred: str, gt: str) -> float:
    """Calculate F1 score based on underscore-separated tokens.

    Gives partial credit for predictions like "yellow_pipette" vs "pipette".
    Returns 1.0 for exact matches, 0.0 for no overlap.
    """
    pred_norm = normalize_value(pred)
    gt_norm = normalize_value(gt)

    # Handle null/none cases
    if pred_norm == "none" or gt_norm == "none":
        return 1.0 if pred_norm == gt_norm else 0.0

    pred_tokens = set(pred_norm.split("_"))
    gt_tokens = set(gt_norm.split("_"))

    if not pred_tokens or not gt_tokens:
        return 0.0

    intersection = pred_tokens & gt_tokens
    if not intersection:
        return 0.0

    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_ground_truth(sample: dict) -> dict[str, str]:
    """Extract ground truth JSON from sample messages."""
    # Find assistant message in the conversation
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            # Parse the JSON response
            content = msg["content"]
            if isinstance(content, list):
                # Extract text from content list
                text = next(
                    (item["text"] for item in content if item.get("type") == "text"),
                    None,
                )
            else:
                text = content

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse ground truth for {sample['id']}")
                return {}

    return {}


def run_inference(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    samples: list[dict],
    batch_size: int = 8,
    custom_prompt: str | None = None,
    desc: str = "Inference",
) -> list[dict]:
    """Run inference on all samples and return predictions.

    Args:
        model: The model to use for inference
        processor: The processor for tokenization
        samples: List of samples to evaluate
        batch_size: Batch size for inference
        custom_prompt: Optional custom prompt to replace the text in user messages
        desc: Description for progress bar
    """
    from qwen_vl_utils import process_vision_info

    logger.info(f"Running inference ({desc})...")
    predictions = []

    # Process in batches to handle OOM
    for i in tqdm(range(0, len(samples), batch_size), desc=desc):
        batch = samples[i : i + batch_size]

        try:
            # Prepare batch messages (user messages only for inference)
            batch_messages = []
            for sample in batch:
                # Extract only user message for inference
                user_msg = next(
                    (m for m in sample["messages"] if m["role"] == "user"), None
                )
                if user_msg:
                    # If custom prompt provided, replace the text content
                    if custom_prompt:
                        modified_msg = {"role": "user", "content": []}
                        for item in user_msg["content"]:
                            if item.get("type") == "image":
                                modified_msg["content"].append(item)
                            elif item.get("type") == "text":
                                modified_msg["content"].append({
                                    "type": "text",
                                    "text": custom_prompt,
                                })
                        batch_messages.append([modified_msg])
                    else:
                        batch_messages.append([user_msg])

            # Apply chat template
            texts = [
                processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in batch_messages
            ]

            # Process vision info
            image_inputs, video_inputs = process_vision_info(batch_messages)

            # Prepare inputs
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Generate with greedy decoding
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Greedy decoding (temperature warning can be ignored)
                    use_cache=True,
                )

            # Decode outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
            ]
            outputs = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Store predictions
            for sample, output in zip(batch, outputs, strict=True):
                predictions.append({"id": sample["id"], "raw_output": output})

        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            # Add empty predictions for failed batch
            for sample in batch:
                predictions.append({
                    "id": sample["id"],
                    "raw_output": "",
                    "error": str(e),
                })

    return predictions


def parse_predictions(predictions: list[dict]) -> list[dict]:
    """Parse JSON from raw model outputs, handling markdown fences."""
    logger.info("Parsing predictions...")

    parsed = []
    for pred in predictions:
        result = {
            "id": pred["id"],
            "raw_output": pred["raw_output"],
            "parse_success": False,
        }

        text = pred["raw_output"]

        # 1. Strip Markdown code blocks (```json ... ```)
        if "```" in text:
            pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                text = match.group(1)

        # 2. If no code blocks, look for the first '{' and last '}'
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]

        try:
            # Try to parse JSON
            parsed_json = json.loads(text)
            result.update({
                "parse_success": True,
                "visual_analysis": parsed_json.get("visual_analysis", ""),
                "verb": parsed_json.get("verb", ""),
                "tool": parsed_json.get("tool", ""),
                "target": parsed_json.get("target", ""),
                "context": parsed_json.get("context", ""),
            })
        except json.JSONDecodeError:
            # Failed to parse - use empty strings
            result.update({
                "visual_analysis": "",
                "verb": "",
                "tool": "",
                "target": "",
                "context": "",
            })

        parsed.append(result)

    parse_rate = sum(p["parse_success"] for p in parsed) / len(parsed) * 100
    logger.info(f"JSON parse success rate: {parse_rate:.1f}%")

    return parsed


def compute_metrics(
    predictions: list[dict], ground_truths: list[dict]
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    logger.info("Computing metrics...")

    # Build comparison dataframe
    data = []
    for pred, gt in zip(predictions, ground_truths, strict=True):
        if not gt:  # Skip if ground truth is empty
            continue

        row = {
            "id": pred["id"],
            "gt_verb": gt.get("verb", ""),
            "pred_verb": pred.get("verb", ""),
            "gt_tool": gt.get("tool", ""),
            "pred_tool": pred.get("tool", ""),
            "gt_target": gt.get("target", ""),
            "pred_target": pred.get("target", ""),
            "gt_context": gt.get("context", ""),
            "pred_context": pred.get("context", ""),
            "gt_visual": gt.get("visual_analysis", ""),
            "pred_visual": pred.get("visual_analysis", ""),
            "parse_success": pred.get("parse_success", False),
        }

        # Compute correctness for each field (with normalization for nan/null/none)
        row["verb_correct"] = normalize_value(row["gt_verb"]) == normalize_value(
            row["pred_verb"]
        )
        row["tool_correct"] = normalize_value(row["gt_tool"]) == normalize_value(
            row["pred_tool"]
        )
        row["target_correct"] = normalize_value(row["gt_target"]) == normalize_value(
            row["pred_target"]
        )
        row["context_correct"] = normalize_value(row["gt_context"]) == normalize_value(
            row["pred_context"]
        )

        # Exact match: all fields correct (including visual_analysis comparison)
        row["exact_match"] = all([
            row["verb_correct"],
            row["tool_correct"],
            row["target_correct"],
            row["context_correct"],
        ])

        # Structured exact match: only verb, tool, target, context (excludes visual_analysis)
        row["structured_exact_match"] = row[
            "exact_match"
        ]  # Same as exact_match for structured fields

        # Partial match score: count of correct fields (0-4)
        row["fields_correct"] = sum([
            row["verb_correct"],
            row["tool_correct"],
            row["target_correct"],
            row["context_correct"],
        ])

        # Visual Triplet: verb + tool + target only (excludes context)
        # Context requires temporal reasoning beyond single clip
        row["visual_triplet_match"] = all([
            row["verb_correct"],
            row["tool_correct"],
            row["target_correct"],
        ])
        row["visual_triplet_fields"] = sum([
            row["verb_correct"],
            row["tool_correct"],
            row["target_correct"],
        ])

        # Token-level F1 scores (partial credit for partial matches)
        row["verb_f1"] = token_f1(row["pred_verb"], row["gt_verb"])
        row["tool_f1"] = token_f1(row["pred_tool"], row["gt_tool"])
        row["target_f1"] = token_f1(row["pred_target"], row["gt_target"])
        row["context_f1"] = token_f1(row["pred_context"], row["gt_context"])
        row["avg_f1"] = (
            row["verb_f1"] + row["tool_f1"] + row["target_f1"] + row["context_f1"]
        ) / 4

        # Visual triplet F1 (excludes context)
        row["visual_triplet_f1"] = (
            row["verb_f1"] + row["tool_f1"] + row["target_f1"]
        ) / 3

        data.append(row)

    df = pd.DataFrame(data)

    # Overall metrics
    metrics = {
        "n_samples": len(df),
        "parse_success_rate": df["parse_success"].mean(),
        "exact_match_accuracy": df["exact_match"].mean(),
        "structured_exact_match": df["structured_exact_match"].mean(),
        "partial_match_score": df["fields_correct"].mean(),
        "verb_accuracy": df["verb_correct"].mean(),
        "tool_accuracy": df["tool_correct"].mean(),
        "target_accuracy": df["target_correct"].mean(),
        "context_accuracy": df["context_correct"].mean(),
        # Token F1 metrics (partial credit)
        "verb_f1_mean": df["verb_f1"].mean(),
        "tool_f1_mean": df["tool_f1"].mean(),
        "target_f1_mean": df["target_f1"].mean(),
        "context_f1_mean": df["context_f1"].mean(),
        "overall_f1": df["avg_f1"].mean(),
        # Visual Triplet metrics (perception only, no temporal reasoning)
        "visual_triplet_accuracy": df["visual_triplet_match"].mean(),
        "visual_triplet_partial": df["visual_triplet_fields"].mean(),
        "visual_triplet_f1": df["visual_triplet_f1"].mean(),
    }

    # Fields correct distribution (how many samples got 0/1/2/3/4 fields right)
    fields_dist = df["fields_correct"].value_counts().sort_index().to_dict()
    metrics["fields_correct_distribution"] = {
        f"{k}_correct": v for k, v in fields_dist.items()
    }

    # Per-verb accuracy
    per_verb = (
        df.groupby("gt_verb")
        .agg({"verb_correct": "mean", "gt_verb": "count"})
        .rename(columns={"verb_correct": "accuracy", "gt_verb": "count"})
    )
    metrics["per_verb_accuracy"] = per_verb.to_dict("index")

    # Per-tool accuracy (top 10)
    per_tool = (
        df.groupby("gt_tool")
        .agg({"tool_correct": "mean", "gt_tool": "count"})
        .rename(columns={"tool_correct": "accuracy", "gt_tool": "count"})
    )
    per_tool = per_tool.nlargest(10, "count")
    metrics["per_tool_accuracy_top10"] = per_tool.to_dict("index")

    # Vocabulary diversity
    metrics["vocab_diversity"] = {
        "verbs": {
            "gt_unique": df["gt_verb"].nunique(),
            "pred_unique": df["pred_verb"].nunique(),
            "ratio": df["pred_verb"].nunique() / df["gt_verb"].nunique(),
        },
        "tools": {
            "gt_unique": df["gt_tool"].nunique(),
            "pred_unique": df["pred_tool"].nunique(),
            "ratio": df["pred_tool"].nunique() / df["gt_tool"].nunique(),
        },
    }

    logger.info(f"Exact match accuracy: {metrics['exact_match_accuracy']:.3f}")
    logger.info(f"Verb accuracy: {metrics['verb_accuracy']:.3f}")
    logger.info(f"Tool accuracy: {metrics['tool_accuracy']:.3f}")

    return metrics, df


def create_visualizations(
    df: pd.DataFrame, output_dir: Path, eval_name: str | None = None
) -> None:
    """Create and save visualization plots."""
    logger.info("Creating visualizations...")
    if eval_name:
        viz_dir = output_dir / "evaluation" / eval_name
    else:
        viz_dir = output_dir / "evaluation"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-field accuracy bar plot
    _, ax = plt.subplots(figsize=(8, 5))
    field_accuracies = {
        "Verb": df["verb_correct"].mean(),
        "Tool": df["tool_correct"].mean(),
        "Target": df["target_correct"].mean(),
        "Context": df["context_correct"].mean(),
        "Exact Match": df["exact_match"].mean(),
    }

    bars = ax.bar(
        field_accuracies.keys(),
        field_accuracies.values(),
        color=sns.color_palette("muted"),
    )
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Field Accuracy")
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(viz_dir / "field_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Per-verb accuracy (top 15 by frequency)
    verb_stats = (
        df.groupby("gt_verb")
        .agg({"verb_correct": "mean", "gt_verb": "count"})
        .rename(columns={"verb_correct": "accuracy", "gt_verb": "count"})
    )
    verb_stats = verb_stats.nlargest(15, "count").sort_values("accuracy")

    _, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        verb_stats.index, verb_stats["accuracy"], color=sns.color_palette("muted")
    )
    ax.set_xlabel("Accuracy")
    ax.set_title("Per-Verb Accuracy (Top 15 by Frequency)")
    ax.set_xlim(0, 1)

    # Add count labels
    for i, (_, row) in enumerate(verb_stats.iterrows()):
        ax.text(
            row["accuracy"],
            i,
            f" n={int(row['count'])}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(viz_dir / "verb_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Confusion matrix for verbs (top 15)
    top_verbs = df["gt_verb"].value_counts().nlargest(15).index.tolist()
    df_top = df[df["gt_verb"].isin(top_verbs) & df["pred_verb"].isin(top_verbs)]

    if len(df_top) > 0:
        cm = confusion_matrix(df_top["gt_verb"], df_top["pred_verb"], labels=top_verbs)

        # Normalize by row (ground truth)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        _, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=top_verbs,
            yticklabels=top_verbs,
            ax=ax,
            cbar_kws={"label": "Normalized Frequency"},
        )
        ax.set_xlabel("Predicted Verb")
        ax.set_ylabel("Ground Truth Verb")
        ax.set_title("Verb Confusion Matrix (Top 15, Row-Normalized)")
        plt.tight_layout()
        plt.savefig(viz_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"Visualizations saved to {viz_dir}")


def save_outputs(
    metrics: dict[str, Any],
    df: pd.DataFrame,
    predictions: list[dict],
    output_dir: Path,
    eval_name: str | None = None,
) -> None:
    """Save all outputs to disk."""
    logger.info("Saving outputs...")
    if eval_name:
        eval_dir = output_dir / "evaluation" / eval_name
    else:
        eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics JSON
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 2. Predictions CSV
    df.to_csv(eval_dir / "predictions.csv", index=False)

    # 3. Errors file (failed parses and incorrect predictions)
    with open(eval_dir / "errors.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("JSON PARSING FAILURES\n")
        f.write("=" * 80 + "\n\n")

        for pred in predictions:
            if not pred.get("parse_success", False):
                f.write(f"ID: {pred['id']}\n")
                f.write(f"Raw output: {pred['raw_output']}\n")
                f.write("-" * 80 + "\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("INCORRECT PREDICTIONS (Sample)\n")
        f.write("=" * 80 + "\n\n")

        incorrect = df[~df["exact_match"]].head(20)
        for _, row in incorrect.iterrows():
            f.write(f"ID: {row['id']}\n")
            f.write(
                f"GT: verb={row['gt_verb']}, tool={row['gt_tool']}, target={row['gt_target']}\n"
            )
            f.write(
                f"Pred: verb={row['pred_verb']}, tool={row['pred_tool']}, target={row['pred_target']}\n"
            )
            f.write("-" * 80 + "\n")

    # 4. Summary text
    with open(eval_dir / "summary.txt", "w") as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total samples: {metrics['n_samples']}\n")
        f.write(f"Parse success rate: {metrics['parse_success_rate']:.1%}\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Exact match accuracy: {metrics['exact_match_accuracy']:.3f}\n")
        f.write(f"Verb accuracy: {metrics['verb_accuracy']:.3f}\n")
        f.write(f"Tool accuracy: {metrics['tool_accuracy']:.3f}\n")
        f.write(f"Target accuracy: {metrics['target_accuracy']:.3f}\n")
        f.write(f"Context accuracy: {metrics['context_accuracy']:.3f}\n\n")

        f.write("VOCABULARY DIVERSITY\n")
        f.write("-" * 80 + "\n")
        vd = metrics["vocab_diversity"]
        f.write(
            f"Verbs - GT unique: {vd['verbs']['gt_unique']}, "
            f"Pred unique: {vd['verbs']['pred_unique']}, "
            f"Ratio: {vd['verbs']['ratio']:.2f}\n"
        )
        f.write(
            f"Tools - GT unique: {vd['tools']['gt_unique']}, "
            f"Pred unique: {vd['tools']['pred_unique']}, "
            f"Ratio: {vd['tools']['ratio']:.2f}\n\n"
        )

        f.write("TOP 10 VERBS BY ACCURACY\n")
        f.write("-" * 80 + "\n")
        verb_acc = metrics["per_verb_accuracy"]
        sorted_verbs = sorted(
            verb_acc.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )[:10]
        for verb, stats in sorted_verbs:
            f.write(f"{verb:15s} - Acc: {stats['accuracy']:.3f} (n={stats['count']})\n")

        # Add F1 metrics
        if "overall_f1" in metrics:
            f.write("\nTOKEN F1 METRICS (partial credit)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Verb F1:     {metrics['verb_f1_mean']:.3f}\n")
            f.write(f"Tool F1:     {metrics['tool_f1_mean']:.3f}\n")
            f.write(f"Target F1:   {metrics['target_f1_mean']:.3f}\n")
            f.write(f"Context F1:  {metrics['context_f1_mean']:.3f}\n")
            f.write(f"Overall F1:  {metrics['overall_f1']:.3f}\n")

        # Add Visual Triplet metrics
        if "visual_triplet_accuracy" in metrics:
            f.write("\nVISUAL TRIPLET METRICS (perception only)\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Exact match (verb+tool+target): {metrics['visual_triplet_accuracy']:.1%}\n"
            )
            f.write(
                f"Partial match (0-3 scale):      {metrics['visual_triplet_partial']:.2f}\n"
            )
            f.write(
                f"Token F1:                       {metrics['visual_triplet_f1']:.3f}\n"
            )
            f.write(
                "\nNote: Context excluded - requires temporal reasoning beyond single clip.\n"
            )

    # 5. Example comparisons file
    with open(eval_dir / "examples.txt", "w") as f:
        f.write("EXAMPLE COMPARISONS\n")
        f.write("=" * 80 + "\n\n")

        # Get mix of correct and incorrect
        correct_samples = df[df["exact_match"]].head(5)
        incorrect_samples = df[~df["exact_match"]].head(5)
        samples = pd.concat([correct_samples, incorrect_samples])

        for _, row in samples.iterrows():
            status = "CORRECT" if row["exact_match"] else "INCORRECT"
            f.write(f"[{row['id']}] - {status}\n")
            f.write(
                f"{'Field':<10} {'Ground Truth':<30} {'Prediction':<30} {'F1':>6}\n"
            )
            f.write("-" * 80 + "\n")
            for field in ["verb", "tool", "target", "context"]:
                gt = str(row[f"gt_{field}"])[:28]
                pred = str(row[f"pred_{field}"])[:28]
                f1 = row[f"{field}_f1"]
                match = "=" if row[f"{field}_correct"] else "X"
                f.write(f"{field:<10} {gt:<30} {pred:<30} {f1:>5.2f} {match}\n")
            f.write("\n")

    logger.info(f"All outputs saved to {eval_dir}")


def print_comparison_table(
    ft_metrics: dict[str, Any],
    base_metrics: dict[str, Any] | None = None,
    prompt_comparison: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print comparison tables to console."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    if base_metrics:
        print("\nMODEL COMPARISON: Fine-tuned vs Base")
        print("-" * 70)
        print(f"{'Metric':<25} {'Base':>12} {'Fine-tuned':>12} {'Improvement':>12}")
        print("-" * 70)

        for metric, ft_val in [
            ("Verb Accuracy", ft_metrics["verb_accuracy"]),
            ("Tool Accuracy", ft_metrics["tool_accuracy"]),
            ("Target Accuracy", ft_metrics["target_accuracy"]),
            ("Context Accuracy", ft_metrics["context_accuracy"]),
            ("Exact Match", ft_metrics["exact_match_accuracy"]),
            ("Partial Score (0-4)", ft_metrics["partial_match_score"]),
        ]:
            base_key = metric.lower().replace(" ", "_").replace("(0-4)", "").strip()
            if "partial" in base_key:
                base_key = "partial_match_score"
            base_val = base_metrics.get(base_key, 0)

            if base_val > 0:
                improvement = ((ft_val - base_val) / base_val) * 100
                imp_str = (
                    f"+{improvement:.0f}%" if improvement > 0 else f"{improvement:.0f}%"
                )
            else:
                imp_str = "N/A"

            if "Partial" in metric:
                print(f"{metric:<25} {base_val:>12.2f} {ft_val:>12.2f} {imp_str:>12}")
            else:
                print(
                    f"{metric:<25} {base_val * 100:>11.1f}% {ft_val * 100:>11.1f}% {imp_str:>12}"
                )

    if prompt_comparison:
        print("\n" + "=" * 70)
        print("PROMPT ABLATION: With vs Without visual_analysis")
        print("-" * 70)
        print(f"{'Metric':<25} {'With Visual':>15} {'Without Visual':>15}")
        print("-" * 70)

        with_vis = prompt_comparison.get("with_visual", {})
        without_vis = prompt_comparison.get("without_visual", {})

        for metric in [
            "verb_accuracy",
            "tool_accuracy",
            "target_accuracy",
            "context_accuracy",
            "exact_match_accuracy",
        ]:
            display_name = metric.replace("_", " ").title().replace("Accuracy", "Acc")
            with_val = with_vis.get(metric, 0)
            without_val = without_vis.get(metric, 0)
            print(
                f"{display_name:<25} {with_val * 100:>14.1f}% {without_val * 100:>14.1f}%"
            )

    # Always print fine-tuned model results
    print("\n" + "=" * 70)
    print("FINE-TUNED MODEL DETAILED RESULTS")
    print("-" * 70)
    print(f"Samples evaluated: {ft_metrics['n_samples']}")
    print(f"JSON parse rate: {ft_metrics['parse_success_rate'] * 100:.1f}%")
    print("\nPer-field accuracy:")
    print(f"  Verb:    {ft_metrics['verb_accuracy'] * 100:>6.1f}%")
    print(f"  Tool:    {ft_metrics['tool_accuracy'] * 100:>6.1f}%")
    print(f"  Target:  {ft_metrics['target_accuracy'] * 100:>6.1f}%")
    print(f"  Context: {ft_metrics['context_accuracy'] * 100:>6.1f}%")
    print("\nAggregate metrics:")
    print(
        f"  Exact match (all 4 correct): {ft_metrics['exact_match_accuracy'] * 100:.1f}%"
    )
    print(f"  Partial match score (0-4):   {ft_metrics['partial_match_score']:.2f}")

    if "fields_correct_distribution" in ft_metrics:
        print("\nFields correct distribution:")
        for k, v in sorted(ft_metrics["fields_correct_distribution"].items()):
            print(f"  {k}: {v} samples")

    # Token F1 metrics (partial credit)
    if "overall_f1" in ft_metrics:
        print("\nSOFT METRICS (Token F1 - partial credit)")
        print("-" * 40)
        print(f"  Verb F1:     {ft_metrics['verb_f1_mean']:.3f}")
        print(f"  Tool F1:     {ft_metrics['tool_f1_mean']:.3f}")
        print(f"  Target F1:   {ft_metrics['target_f1_mean']:.3f}")
        print(f"  Context F1:  {ft_metrics['context_f1_mean']:.3f}")
        print(f"  Overall F1:  {ft_metrics['overall_f1']:.3f}")

    # Visual Triplet metrics (perception without context)
    if "visual_triplet_accuracy" in ft_metrics:
        print("\nVISUAL TRIPLET (verb + tool + target, no context)")
        print("-" * 40)
        print(f"  Exact match:     {ft_metrics['visual_triplet_accuracy']:.1%}")
        print(f"  Partial (0-3):   {ft_metrics['visual_triplet_partial']:.2f}")
        print(f"  Token F1:        {ft_metrics['visual_triplet_f1']:.3f}")

    print("=" * 70 + "\n")


def print_example_comparisons(df: pd.DataFrame, n: int = 10) -> None:
    """Print side-by-side GT vs Pred comparisons (mix of correct/incorrect)."""
    print("\n" + "=" * 80)
    print(f"EXAMPLE COMPARISONS ({n} samples)")
    print("=" * 80)

    # Get mix: half correct, half incorrect (or as many as available)
    correct_df = df[df["exact_match"]]
    incorrect_df = df[~df["exact_match"]]

    n_correct = min(len(correct_df), n // 2)
    n_incorrect = min(len(incorrect_df), n - n_correct)

    samples = pd.concat([
        correct_df.head(n_correct),
        incorrect_df.head(n_incorrect),
    ])

    for _, row in samples.iterrows():
        status = "CORRECT" if row["exact_match"] else "INCORRECT"
        print(f"\n[{row['id']}] - {status}")
        print(f"{'Field':<10} {'Ground Truth':<30} {'Prediction':<30} {'F1':>6}")
        print("-" * 80)
        for field in ["verb", "tool", "target", "context"]:
            gt = str(row[f"gt_{field}"])[:28]
            pred = str(row[f"pred_{field}"])[:28]
            f1 = row[f"{field}_f1"]
            match = "=" if row[f"{field}_correct"] else "X"
            print(f"{field:<10} {gt:<30} {pred:<30} {f1:>5.2f} {match}")

    print()


def run_evaluation(config: EvaluationConfig) -> dict[str, Any]:
    """Run evaluation pipeline.

    This is the main evaluation function, usable both from CLI and as module import.

    Args:
        config: Evaluation configuration

    Returns:
        Dict containing:
        - "metrics": Primary metrics dict
        - "predictions": List of prediction dicts
        - "df": DataFrame with all results
        - "base_metrics": Base model metrics (if compare_base=True)
    """
    # Create output directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Determine which prompts to test
    prompts_to_test = []
    if config.prompt_mode == "both":
        prompts_to_test = [
            ("with_visual", PROMPT_WITH_VISUAL),
            ("without_visual", PROMPT_WITHOUT_VISUAL),
        ]
    elif config.prompt_mode == "with_visual":
        prompts_to_test = [("with_visual", PROMPT_WITH_VISUAL)]
    else:
        prompts_to_test = [("without_visual", PROMPT_WITHOUT_VISUAL)]

    # 1. Load fine-tuned model
    model, processor = load_model_and_processor(config.checkpoint_dir)

    # 2. Load validation data
    max_samples = config.max_samples if config.max_samples > 0 else None
    samples = load_validation_data(config.val_path, max_samples)

    # 3. Extract ground truths
    ground_truths = [extract_ground_truth(s) for s in samples]

    # 4. Run inference with fine-tuned model (for each prompt mode)
    all_results = {}
    prompt_comparison = {}

    for prompt_name, prompt_text in prompts_to_test:
        logger.info(f"Evaluating with prompt mode: {prompt_name}")

        predictions = run_inference(
            model,
            processor,
            samples,
            config.batch_size,
            custom_prompt=prompt_text,
            desc=f"Fine-tuned ({prompt_name})",
        )
        parsed_predictions = parse_predictions(predictions)
        metrics, df = compute_metrics(parsed_predictions, ground_truths)

        all_results[prompt_name] = {
            "predictions": parsed_predictions,
            "metrics": metrics,
            "df": df,
        }
        prompt_comparison[prompt_name] = metrics

    # Use the first prompt mode as primary results
    primary_prompt = prompts_to_test[0][0]
    ft_metrics = all_results[primary_prompt]["metrics"]
    ft_predictions = all_results[primary_prompt]["predictions"]
    ft_df = all_results[primary_prompt]["df"]

    # 5. Optionally compare with base model
    base_metrics = None
    if config.compare_base:
        logger.info("Loading base model for comparison...")
        # Unload fine-tuned model to free memory
        del model
        torch.cuda.empty_cache()

        base_model, base_processor = load_base_model()
        base_predictions = run_inference(
            base_model,
            base_processor,
            samples,
            config.batch_size,
            custom_prompt=PROMPT_WITH_VISUAL,
            desc="Base model",
        )
        base_parsed = parse_predictions(base_predictions)
        base_metrics, _ = compute_metrics(base_parsed, ground_truths)

        # Clean up base model
        del base_model
        torch.cuda.empty_cache()

    # 6. Print comparison tables
    print_comparison_table(
        ft_metrics,
        base_metrics=base_metrics,
        prompt_comparison=prompt_comparison if len(prompts_to_test) > 1 else None,
    )

    # 6b. Print example comparisons
    print_example_comparisons(ft_df, n=10)

    # 7. Create visualizations
    create_visualizations(ft_df, config.checkpoint_dir, config.eval_name)

    # 8. Save outputs (include comparison data)
    # Make a copy to avoid circular references when adding nested metrics
    metrics_to_save = dict(ft_metrics)
    if base_metrics:
        metrics_to_save["base_model_comparison"] = base_metrics
    if len(prompts_to_test) > 1:
        # Copy each prompt's metrics to avoid circular reference
        metrics_to_save["prompt_comparison"] = {
            k: dict(v) for k, v in prompt_comparison.items()
        }

    save_outputs(
        metrics_to_save, ft_df, ft_predictions, config.checkpoint_dir, config.eval_name
    )

    logger.info("Evaluation complete!")
    eval_path = config.checkpoint_dir / "evaluation"
    if config.eval_name:
        eval_path = eval_path / config.eval_name
    logger.info(f"Results saved to {eval_path}")

    # Return results for programmatic use
    return {
        "metrics": ft_metrics,
        "predictions": ft_predictions,
        "df": ft_df,
        "base_metrics": base_metrics,
        "all_results": all_results,
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    config = EvaluationConfig(
        checkpoint_dir=args.checkpoint_dir,
        val_path=args.val_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        compare_base=args.compare_base,
        prompt_mode=args.prompt_mode,
        eval_name=args.eval_name,
    )

    run_evaluation(config)


if __name__ == "__main__":
    main()

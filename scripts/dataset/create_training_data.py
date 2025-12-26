import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from dataset_config import (
    configure_logging,
    ensure_output_dirs,
    load_dataset_config,
    resolve_paths,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test JSONL splits")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("dataset_config.yaml"),
        help="Path to dataset_config.yaml (defaults to scripts/dataset/dataset_config.yaml).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Config profile name (e.g. mac, rcp, izar). Defaults to config's default_profile.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Split output subfolder name. Defaults to config's default_split_name.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser.parse_args()


# Parameters
TOTAL_SAMPLES_PER_VERB = 1000
NUM_FRAMES_TO_SAMPLE = 8  # Configurable frame count (multiple of 2)
assert (
    NUM_FRAMES_TO_SAMPLE % 2 == 0 and NUM_FRAMES_TO_SAMPLE <= 16
)  # Since we have 16 frames extracted for everything
MIN_DURATION = 0.5
MAX_DURATION = 3.0

# Prompt templates
# all explicitly specify the 5 required JSON keys
# to prevent smaller models from hallucinating different field names

PROMPT_TEMPLATES = [
    # 1. Chain-of-thought (best practice from Wei et al.)
    "First, describe what you observe in the video frames. Then identify the action verb, the tool being manipulated, any target object, and the lab protocol context. Output as JSON with keys: visual_analysis, verb, tool, target, context.",
    # 2. Direct with explicit structure
    "Analyze this laboratory procedure clip. Return JSON with: visual_analysis (describe what you see), verb (action type), tool (manipulated object), target (affected object or null), context (protocol step).",
    # 3. Minimal/terse (tests robustness)
    "Annotate this lab video. Output JSON with keys: visual_analysis, verb, tool, target, context.",
    # 4. Role-playing with context (improves domain alignment)
    "You are a lab technician documenting procedures. Watch this clip and record: what action is performed (verb), which instrument is used (tool), what it interacts with (target), and which protocol step this is (context). Include a brief visual description. Format as JSON with keys: visual_analysis, verb, tool, target, context.",
    # 5. Question-based (natural language style)
    "What's happening in this video? Describe the visual scene, then answer: What action? Which tool? What target? What protocol step? Provide as JSON: visual_analysis, verb, tool, target, context.",
    # 6. Step-by-step explicit
    "Analyze this atomic lab operation in steps: (1) Describe the visual scene, (2) Identify the action verb, (3) Name the tool/object being manipulated, (4) Identify the target (if any), (5) Determine the protocol context. Output JSON with keys: visual_analysis, verb, tool, target, context.",
    # 7. Example-driven (few-shot style without actual examples)
    'Generate a structured annotation following this format: {"visual_analysis": "[description]", "verb": "[action]", "tool": "[instrument]", "target": "[object]", "context": "[protocol_step]"}. Analyze this clip and fill in the JSON.',
]

VERB_PREPOSITIONS = {
    "insert": "into",
    "put": "on",
    "take": "from",
    "press": "on",
    "release": "of",
    "detach": "from",
    "open": "of",
    "close": "of",
    "eject": "into",
    "shake": "of",
}


def generate_visual_analysis(row: pd.Series) -> str:
    """Generate a short natural-language description from an annotation row."""
    hand = row["hand"]  # if pd.notna(row["hand"]) else "scientist"
    # if hand == "scientist":
    #     logger.info("Scientist found! %s", row)
    verb = row["verb"]
    tool = str(row["manipulated_object"]).replace("_", " ")
    target = row["affected_object"]

    if pd.isna(target) or str(target) == "nan":
        return f"The {hand} hand is performing a '{verb}' action using the {tool}."

    target = str(target).replace("_", " ")
    prep = VERB_PREPOSITIONS.get(verb, "with")
    verb_ing = verb[:-1] + "ing" if verb.endswith("e") else verb + "ing"

    return f"The {hand} hand is {verb_ing} the {tool} {prep} the {target}."


def filter_data(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to the annotation DataFrame."""
    logger.info("Initial pool: %d", len(annotations_df))
    annotations_df["duration"] = annotations_df["end_sec"] - annotations_df["start_sec"]
    filtered_df = annotations_df[
        (annotations_df["duration"] >= MIN_DURATION)
        & (annotations_df["duration"] <= MAX_DURATION)
    ]
    filtered_df = filtered_df[
        filtered_df["manipulated_object"].notna()
        & (filtered_df["manipulated_object"] != "nan")
    ]
    logger.info("Valid pool after filtering: %d", len(filtered_df))
    return filtered_df


def get_stratified_splits(
    annotations_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create per-verb 80/10/10 train/val/test splits with downsampling."""
    train_dfs = []
    val_dfs = []
    test_dfs = []

    all_verbs = annotations_df["verb"].unique()

    logger.info("Processing splits (target per verb: %d)...", TOTAL_SAMPLES_PER_VERB)

    for verb in all_verbs:
        v_df = annotations_df[annotations_df["verb"] == verb]
        count = len(v_df)

        if count <= TOTAL_SAMPLES_PER_VERB:
            # Take everything (Rare case)
            selected_df = v_df
            logger.info("[%s] Rare: keeping all %d", verb, count)
        else:
            # Weighted downsampling: weight = 1 / tool frequency
            tool_counts = v_df["manipulated_object"].value_counts()
            weights = 1.0 / v_df["manipulated_object"].map(tool_counts)

            selected_df = v_df.sample(
                n=TOTAL_SAMPLES_PER_VERB, weights=weights, random_state=42
            )
            logger.info(
                "[%s] Common: downsampled %d -> %d (weighted)",
                verb,
                count,
                TOTAL_SAMPLES_PER_VERB,
            )

        # 80/10/10 split
        train, temp = train_test_split(
            selected_df, test_size=0.2, random_state=42, shuffle=True
        )

        val, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)


def create_jsonl(
    annotations_df: pd.DataFrame, *, frame_base_dir: Path, output_path: Path
) -> None:
    """Generate a JSONL file for a given DataFrame."""
    entries: list[str] = []

    # Calculate frame indices (e.g., 0, 2, 4...)
    frame_indices = np.linspace(0, 15, NUM_FRAMES_TO_SAMPLE, dtype=int)

    for _, row in annotations_df.iterrows():
        segment_id = f"{row['video_id']}_{row['start_sec']:.1f}_{row['end_sec']:.1f}"
        segment_dir = frame_base_dir / segment_id

        # Build absolute paths for frames
        image_paths = [str(segment_dir / f"frame_{i:02d}.jpg") for i in frame_indices]

        prompt = random.choice(PROMPT_TEMPLATES)

        response_json = {
            "visual_analysis": generate_visual_analysis(row),
            "verb": row["verb"],
            "tool": str(row["manipulated_object"]),
            "target": str(row["affected_object"]),
            "context": str(row["task_step"]),
        }

        entry = {
            "id": segment_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": p} for p in image_paths],
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(response_json)}],
                },
            ],
        }
        entries.append(json.dumps(entry))

    # Shuffle final list to mix verbs together
    random.shuffle(entries)

    with open(output_path, "w") as f:
        f.write("\n".join(entries))
    logger.info("Saved %d samples to %s", len(entries), output_path)


if __name__ == "__main__":
    args = parse_arguments()
    configure_logging(args.log_level)

    config = load_dataset_config(args.config)
    resolved, effective_split_name = resolve_paths(
        config, profile_name=args.profile, split_name=args.split_name
    )
    ensure_output_dirs(resolved)

    input_csv = resolved.consolidated_csv
    frame_base_dir = resolved.frames_dir
    out_dir = resolved.splits_dir

    out_train = out_dir / "finebio_train.jsonl"
    out_val = out_dir / "finebio_val.jsonl"
    out_test = out_dir / "finebio_test.jsonl"

    logger.info("Dataset profile: %s", resolved.profile.name)
    logger.info("Using split_name: %s", effective_split_name)
    logger.info("Input CSV: %s", input_csv)
    logger.info("Frame dir: %s", frame_base_dir)
    logger.info("Split output dir: %s", out_dir)

    # Load
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing consolidated CSV: {input_csv}. Run process_dataset.py first (or update output_dir)."
        )
    annotations_df = pd.read_csv(input_csv)

    # Filter
    filtered_df = filter_data(annotations_df)

    # Split
    train_df, val_df, test_df = get_stratified_splits(filtered_df)

    logger.info("--- Split Statistics ---")
    logger.info("Train Size: %d", len(train_df))
    logger.info("Val Size:   %d", len(val_df))
    logger.info("Test Size:  %d", len(test_df))

    # Generate Files
    create_jsonl(train_df, frame_base_dir=frame_base_dir, output_path=out_train)
    create_jsonl(val_df, frame_base_dir=frame_base_dir, output_path=out_val)
    create_jsonl(test_df, frame_base_dir=frame_base_dir, output_path=out_test)

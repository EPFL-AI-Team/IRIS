from __future__ import annotations

import json
import random
from collections.abc import Mapping
from typing import Any

import pandas as pd

# Prompt templates
# All explicitly specify the required JSON keys to reduce field-name drift.

PROMPT_TEMPLATES: list[str] = [
    "First, describe what you observe in the video frames. Then identify the action verb, the tool being manipulated, any target object, and the lab protocol context. Output as JSON with keys: visual_analysis, verb, tool, target, context.",
    "Analyze this laboratory procedure clip. Return JSON with: visual_analysis (describe what you see), verb (action type), tool (manipulated object), target (affected object or null), context (protocol step).",
    "Annotate this lab video. Output JSON with keys: visual_analysis, verb, tool, target, context.",
    "You are a lab technician documenting procedures. Watch this clip and record: what action is performed (verb), which instrument is used (tool), what it interacts with (target), and which protocol step this is (context). Include a brief visual description. Format as JSON with keys: visual_analysis, verb, tool, target, context.",
    "What's happening in this video? Describe the visual scene, then answer: What action? Which tool? What target? What protocol step? Provide as JSON: visual_analysis, verb, tool, target, context.",
    "Analyze this atomic lab operation in steps: (1) Describe the visual scene, (2) Identify the action verb, (3) Name the tool/object being manipulated, (4) Identify the target (if any), (5) Determine the protocol context. Output JSON with keys: visual_analysis, verb, tool, target, context.",
    'Generate a structured annotation following this format: {"visual_analysis": "[description]", "verb": "[action]", "tool": "[instrument]", "target": "[object]", "context": "[protocol_step]"}. Analyze this clip and fill in the JSON.',
]

VERB_PREPOSITIONS: dict[str, str] = {
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


def pick_prompt(*, rng: random.Random | None = None) -> str:
    """Return a prompt template (randomly chosen)."""
    chooser = rng or random
    return chooser.choice(PROMPT_TEMPLATES)


def generate_visual_analysis(
    row: pd.Series | Mapping[str, Any],
    *,
    verb_prepositions: Mapping[str, str] = VERB_PREPOSITIONS,
) -> str:
    """Generate a short natural-language description from an annotation row."""
    # Support both pandas Series and dict-like input.
    hand = row.get("hand", "unknown")  # type: ignore[call-arg]
    verb = str(row.get("verb", "unknown"))  # type: ignore[call-arg]
    tool = str(row.get("manipulated_object", "unknown")).replace("_", " ")  # type: ignore[call-arg]
    target = row.get("affected_object")  # type: ignore[call-arg]

    if (
        target is None
        or (isinstance(target, float) and pd.isna(target))
        or str(target) == "nan"
    ):
        return f"The {hand} hand is performing a '{verb}' action using the {tool}."

    target_str = str(target).replace("_", " ")
    prep = str(verb_prepositions.get(verb, "with"))
    verb_ing = verb[:-1] + "ing" if verb.endswith("e") else verb + "ing"

    return f"The {hand} hand is {verb_ing} the {tool} {prep} the {target_str}."


def expected_output_json(row: pd.Series | Mapping[str, Any]) -> dict[str, str]:
    """Build the target/expected JSON fields from a CSV row."""
    return {
        "visual_analysis": generate_visual_analysis(row),
        "verb": str(row.get("verb", "unknown")),  # type: ignore[call-arg]
        "tool": str(row.get("manipulated_object", "unknown")),  # type: ignore[call-arg]
        "target": str(row.get("affected_object", "unknown")),  # type: ignore[call-arg]
        "context": str(row.get("task_step", "unknown")),  # type: ignore[call-arg]
    }


def segment_id_from_row(row: pd.Series | Mapping[str, Any]) -> str:
    video_id = str(row.get("video_id"))  # type: ignore[call-arg]
    start_sec = float(row.get("start_sec"))  # type: ignore[call-arg]
    end_sec = float(row.get("end_sec"))  # type: ignore[call-arg]
    return f"{video_id}_{start_sec:.1f}_{end_sec:.1f}"


def chat_jsonl_entry(
    *,
    entry_id: str,
    image_paths: list[str],
    prompt: str,
    expected_json: Mapping[str, Any],
) -> str:
    """Create a single JSONL line in the 'messages' format used for VLM fine-tuning."""
    entry = {
        "id": entry_id,
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
                "content": [{"type": "text", "text": json.dumps(dict(expected_json))}],
            },
        ],
    }
    return json.dumps(entry)

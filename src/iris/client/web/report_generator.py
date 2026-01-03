"""LLM-powered report generation using Gemini 2.5 Flash."""

import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Summary data for report generation."""

    session_id: str
    status: str
    duration_sec: float
    video_file: str | None
    annotation_file: str | None
    config: dict[str, Any]
    total_results: int
    sample_results: list[dict[str, Any]]
    ground_truth_count: int
    annotations_sample: list[dict[str, Any]]


def build_report_context(
    session: dict[str, Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
) -> SessionSummary:
    """Build context for LLM report generation.

    Args:
        session: Session data from database.
        results: List of inference results.
        annotations: Optional ground truth annotations.

    Returns:
        SessionSummary with data for the report.
    """
    started_at = session.get("started_at", 0)
    completed_at = session.get("completed_at", time.time())
    duration_sec = (completed_at - started_at) if started_at else 0

    # Take sample of results for context (avoid huge prompts)
    # sample_results = results[:5] if len(results) > 5 else results
    sample_results = results
    
    # Simplify results for prompt
    simplified_results = []
    for r in sample_results:
        simplified_results.append({
            "job_id": r.get("job_id"),
            "video_time_ms": r.get("video_time_ms"),
            "inference_duration_ms": r.get("inference_duration_ms"),
            "result": r.get("result"),
        })

    return SessionSummary(
        session_id=session.get("id", "unknown"),
        status=session.get("status", "unknown"),
        duration_sec=duration_sec,
        video_file=session.get("video_file"),
        annotation_file=session.get("annotation_file"),
        config=session.get("config", {}),
        total_results=len(results),
        sample_results=simplified_results,
        ground_truth_count=len(annotations) if annotations else 0,
        annotations_sample=annotations[:5] if annotations else [],
    )


def build_report_prompt(summary: SessionSummary) -> str:
    """Build prompt for documentation-style narrative generation.

    Args:
        summary: Session summary data.

    Returns:
        Formatted prompt string optimized for chronological narrative.
    """
    context = {
        "session_id": summary.session_id,
        "status": summary.status,
        "duration_sec": round(summary.duration_sec, 2),
        "video_file": summary.video_file,
        "total_results": summary.total_results,
        "sample_results": summary.sample_results,  # All inference outputs in order
        "ground_truth_count": summary.ground_truth_count,
    }

    return f"""You are a technical documentation writer creating a human-readable report documenting what happened during a laboratory first-person view video analysis session.

## Session Data

```json
{json.dumps(context, indent=2, default=str)}
```

## Task

Create a **flowing narrative documentation** (300-400 words) that describes the sequential flow of events detected in the laboratory video. The sample_results contain the inference outputs in chronological order - each entry shows what actions and tools were detected at specific timestamps from a first-person perspective.

## Instructions

Write in complete paragraphs (not bullet points) documenting the chronological flow. Structure:

1. **Session Overview** (1 paragraph): Session details, video file, duration, configuration

2. **Procedure Timeline** (2-3 paragraphs): Chronologically describe what was detected from the first-person laboratory view:
   - What actions and tools appear at the start
   - How the procedure progresses (transitions between actions/tools)
   - Key moments or changes in the detected activities
   - What was detected toward the end
   - Note: The model may have made errors, so use phrases like "the model detected," "appears to show," "suggests" when describing detections

3. **Summary** (1 paragraph): Overview of the complete procedure flow, any notable patterns

Focus on creating a readable timeline of events from the inference outputs. Use professional laboratory/technical language where appropriate.
"""


def get_gemini_api_key() -> str | None:
    """Get Gemini API key from stored value or environment variables.

    Checks in order:
    1. Stored API key (set via UI)
    2. GOOGLE_API_KEY environment variable
    3. GEMINI_API_KEY environment variable

    Returns:
        API key string or None if not found
    """
    # Import here to avoid circular dependency
    try:
        from iris.client.web.routes import gemini_api_key_store
        if gemini_api_key_store:
            return gemini_api_key_store
    except ImportError:
        pass

    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


async def generate_report_stream_gemini(
    summary: SessionSummary,
) -> AsyncGenerator[str, None]:
    """Generate report using Gemini 2.5 Flash with streaming.

    Args:
        summary: Session summary data.

    Yields:
        Markdown text chunks as they're generated.
    """
    try:
        from google import genai
    except ImportError:
        yield "# Report Generation Error\n\n"
        yield "The `google-genai` package is not installed. "
        yield "Install it with: `pip install google-genai`\n"
        return

    api_key = get_gemini_api_key()
    if not api_key:
        yield "# Report Generation Error\n\n"
        yield "Gemini API key is not configured. Set it via the TopBar settings or environment variables (GOOGLE_API_KEY or GEMINI_API_KEY).\n"
        return

    prompt = build_report_prompt(summary)

    try:
        client = genai.Client(api_key=api_key)

        async for chunk in await client.aio.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
        ):
            if chunk.text:
                yield chunk.text

    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        yield f"\n\n---\n\n**Error generating report:** {e}\n"


async def generate_report_stream(
    session: dict[str, Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a streaming report using Gemini 2.5 Flash.

    Args:
        session: Session data from database.
        results: List of inference results.
        annotations: Optional ground truth annotations.

    Yields:
        Markdown text chunks as they're generated.
    """
    summary = build_report_context(session, results, annotations)
    async for chunk in generate_report_stream_gemini(summary):
        yield chunk


def generate_fallback_report(
    session: dict[str, Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a basic statistics report without LLM.

    Used as fallback when no LLM API key is available.

    Args:
        session: Session data.
        results: Inference results.
        annotations: Optional annotations.

    Returns:
        Markdown report string.
    """
    summary = build_report_context(session, results, annotations)

    # Calculate basic statistics
    if results:
        inference_times = [r.get("inference_duration_ms", 0) for r in results]
        avg_inference = sum(inference_times) / len(inference_times)
        min_inference = min(inference_times)
        max_inference = max(inference_times)
    else:
        avg_inference = min_inference = max_inference = 0

    report = f"""# Analysis Report

## Session Summary

- **Session ID:** {summary.session_id}
- **Status:** {summary.status}
- **Duration:** {summary.duration_sec:.1f} seconds
- **Video File:** {summary.video_file or 'N/A'}
- **Annotation File:** {summary.annotation_file or 'N/A'}

## Inference Statistics

- **Total Inferences:** {summary.total_results}
- **Ground Truth Annotations:** {summary.ground_truth_count}
- **Average Inference Time:** {avg_inference:.1f} ms
- **Min Inference Time:** {min_inference:.1f} ms
- **Max Inference Time:** {max_inference:.1f} ms

## Configuration

```json
{json.dumps(summary.config, indent=2)}
```

---

*This is a basic statistics report. For detailed AI-powered analysis, configure an LLM API key.*
"""

    return report

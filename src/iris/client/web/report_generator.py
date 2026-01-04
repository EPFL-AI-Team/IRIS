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
    logs: list[dict[str, Any]]


def build_report_context(
    session: dict[str, Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
    logs: list[dict[str, Any]] | None = None,
) -> SessionSummary:
    """Build context for LLM report generation.

    Args:
        session: Session data from database.
        results: List of inference results.
        annotations: Optional ground truth annotations (unused, kept for compatibility).
        logs: Optional session logs with timestamps.

    Returns:
        SessionSummary with data for the report.
    """
    logger.info("=" * 80)
    logger.info("BUILDING REPORT CONTEXT")
    logger.info("=" * 80)

    started_at = session.get("started_at", 0)
    completed_at = session.get("completed_at", time.time())
    duration_sec = (completed_at - started_at) if started_at else 0

    logger.info(f"Session ID: {session.get('id', 'unknown')}")
    logger.info(f"Session status: {session.get('status', 'unknown')}")
    logger.info(f"Started at: {started_at}")
    logger.info(f"Completed at: {completed_at}")
    logger.info(f"Duration: {duration_sec:.2f}s")
    logger.info(f"Video file: {session.get('video_file')}")
    logger.info(f"Annotation file: {session.get('annotation_file')}")
    logger.info(f"Total results: {len(results)}")
    logger.info(f"Total logs: {len(logs) if logs else 0}")

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

    logger.info(f"Simplified {len(simplified_results)} results for prompt")

    # Simplify logs for prompt (include timestamp and message)
    simplified_logs = []
    if logs:
        for log in logs:
            simplified_logs.append({
                "timestamp": log.get("timestamp"),
                "level": log.get("level"),
                "message": log.get("message"),
            })
        logger.info(f"Simplified {len(simplified_logs)} logs for prompt")

    summary = SessionSummary(
        session_id=session.get("id", "unknown"),
        status=session.get("status", "unknown"),
        duration_sec=duration_sec,
        video_file=session.get("video_file"),
        annotation_file=session.get("annotation_file"),
        config=session.get("config", {}),
        total_results=len(results),
        sample_results=simplified_results,
        logs=simplified_logs,
    )

    logger.info("SessionSummary created:")
    logger.info(f"  - session_id: {summary.session_id}")
    logger.info(f"  - status: {summary.status}")
    logger.info(f"  - duration_sec: {summary.duration_sec:.2f}")
    logger.info(f"  - video_file: {summary.video_file}")
    logger.info(f"  - total_results: {summary.total_results}")
    logger.info(f"  - total logs: {len(summary.logs)}")
    logger.info("=" * 80)

    return summary


def build_report_prompt(summary: SessionSummary) -> str:
    """Build prompt for concise bullet-point protocol generation.

    Args:
        summary: Session summary data.

    Returns:
        Formatted prompt string optimized for protocol steps.
    """
    logger.info("BUILDING REPORT PROMPT")
    logger.info(f"Creating context dict for {summary.total_results} results and {len(summary.logs)} logs")

    context = {
        "session_id": summary.session_id,
        "duration_sec": round(summary.duration_sec, 2),
        "video_file": summary.video_file,
        "total_results": summary.total_results,
        "sample_results": summary.sample_results,  # All inference outputs in order
        "logs": summary.logs,  # Session logs with timestamps
    }

    context_json = json.dumps(context, indent=2, default=str)
    logger.info(f"Context JSON length: {len(context_json)} characters")
    logger.info(f"Context preview (first 500 chars):\n{context_json[:500]}")

    prompt = f"""You are a laboratory scientist writing a concise procedure protocol.

## Session Data

```json
{context_json}
```

## Task

Generate a brief protocol summary (100-150 words) documenting the key steps observed in this laboratory procedure. Format as a numbered list of protocol steps.

## Output Format

1. **Session Overview** (1 sentence): Video file, duration, number of observations

2. **Procedure Steps** (numbered list, 5-8 steps maximum):
   - Each step should be 1-2 sentences
   - Focus on chronological order of major actions
   - Use passive voice (e.g., "The pipette was inserted into the tube")
   - Mention tools and targets clearly
   - Describe what was observed, not what the model "detected"

3. **Summary** (1 sentence): Overall characterization of the procedure

Write in professional laboratory protocol style. Be concise and focus on observable actions.
"""

    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.info(f"Prompt preview (first 300 chars):\n{prompt[:300]}")

    return prompt


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
    logger.info("GENERATING STREAMING GEMINI REPORT")

    try:
        from google import genai
        logger.info("Google genai package imported successfully")
    except ImportError:
        logger.error("google-genai package not installed")
        yield "# Report Generation Error\n\n"
        yield "The `google-genai` package is not installed. "
        yield "Install it with: `pip install google-genai`\n"
        return

    api_key = get_gemini_api_key()
    if not api_key:
        logger.warning("No Gemini API key found")
        yield "# Report Generation Error\n\n"
        yield "Gemini API key is not configured. Set it via the TopBar settings or environment variables (GOOGLE_API_KEY or GEMINI_API_KEY).\n"
        return

    logger.info("Gemini API key found, building prompt")
    prompt = build_report_prompt(summary)

    try:
        logger.info("Creating Gemini client")
        client = genai.Client(api_key=api_key)

        logger.info("Starting streaming content generation with model: gemini-2.5-flash")
        chunk_count = 0
        total_chars = 0

        async for chunk in await client.aio.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
        ):
            if chunk.text:
                chunk_count += 1
                total_chars += len(chunk.text)
                if chunk_count % 10 == 0:
                    logger.info(f"Received chunk {chunk_count}, total chars: {total_chars}")
                yield chunk.text

        logger.info(f"Streaming complete: {chunk_count} chunks, {total_chars} total characters")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        yield f"\n\n---\n\n**Error generating report:** {e}\n"


async def generate_report_stream(
    session: dict[str, Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
    logs: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[str, None]:
    """Generate a streaming report using Gemini 2.5 Flash.

    Args:
        session: Session data from database.
        results: List of inference results.
        annotations: Optional ground truth annotations.
        logs: Optional session logs with timestamps.

    Yields:
        Markdown text chunks as they're generated.
    """
    summary = build_report_context(session, results, annotations, logs)
    async for chunk in generate_report_stream_gemini(summary):
        yield chunk


def generate_fallback_report(
    session: dict[str, Any],
    results: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None = None,
    logs: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a basic statistics report without LLM.

    Used as fallback when no LLM API key is available.

    Args:
        session: Session data.
        results: Inference results.
        annotations: Optional annotations (unused, kept for compatibility).
        logs: Optional session logs with timestamps.

    Returns:
        Markdown report string.
    """
    logger.info("GENERATING FALLBACK REPORT")
    logger.info(f"Input: {len(results)} results, {len(logs) if logs else 0} logs")

    summary = build_report_context(session, results, annotations, logs)

    # Calculate basic statistics
    if results:
        inference_times = [r.get("inference_duration_ms", 0) for r in results]
        avg_inference = sum(inference_times) / len(inference_times)
        min_inference = min(inference_times)
        max_inference = max(inference_times)
        logger.info(f"Inference stats: avg={avg_inference:.1f}ms, min={min_inference:.1f}ms, max={max_inference:.1f}ms")
    else:
        avg_inference = min_inference = max_inference = 0
        logger.warning("No results to calculate inference statistics")

    report = f"""# Analysis Report

## Session Summary

- **Session ID:** {summary.session_id}
- **Status:** {summary.status}
- **Duration:** {summary.duration_sec:.1f} seconds
- **Video File:** {summary.video_file or 'N/A'}
- **Total Inferences:** {summary.total_results}

## Inference Statistics

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

    logger.info(f"Fallback report generated, length: {len(report)} characters")
    logger.info("=" * 80)

    return report

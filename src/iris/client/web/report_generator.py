"""LLM-powered report generation for analysis sessions."""

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
    sample_results = results[:10] if len(results) > 10 else results

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
    """Build the prompt for LLM report generation.

    Args:
        summary: Session summary data.

    Returns:
        Formatted prompt string.
    """
    context = {
        "session_id": summary.session_id,
        "status": summary.status,
        "duration_sec": round(summary.duration_sec, 2),
        "video_file": summary.video_file,
        "annotation_file": summary.annotation_file,
        "config": summary.config,
        "total_results": summary.total_results,
        "sample_results": summary.sample_results,
        "ground_truth_count": summary.ground_truth_count,
        "annotations_sample": summary.annotations_sample,
    }

    return f"""Analyze this video inference session and generate a comprehensive Markdown report.

## Session Data

```json
{json.dumps(context, indent=2, default=str)}
```

## Instructions

Generate a professional analysis report in Markdown format with the following sections:

1. **Executive Summary** (2-3 sentences)
   - Overall session outcome
   - Key findings at a glance

2. **Session Configuration**
   - Video file analyzed
   - FPS and frame settings
   - Any notable configuration choices

3. **Inference Performance**
   - Total inferences completed
   - Average inference time
   - Throughput analysis

4. **Results Analysis**
   - Common actions/tools detected
   - Patterns in the predictions
   - Quality observations

5. **Ground Truth Comparison** (if annotations available)
   - Alignment with ground truth
   - Notable discrepancies
   - Accuracy assessment

6. **Recommendations**
   - Suggested improvements
   - Configuration optimizations
   - Next steps

Use clear, concise language. Include specific numbers where available.
Format the output as proper Markdown with headers, bullet points, and emphasis where appropriate.
"""


async def generate_report_stream_anthropic(
    summary: SessionSummary,
) -> AsyncGenerator[str, None]:
    """Generate report using Anthropic Claude API with streaming.

    Args:
        summary: Session summary data.

    Yields:
        Markdown text chunks as they're generated.
    """
    try:
        import anthropic
    except ImportError:
        yield "# Report Generation Error\n\n"
        yield "The `anthropic` package is not installed. "
        yield "Install it with: `pip install anthropic`\n"
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        yield "# Report Generation Error\n\n"
        yield "ANTHROPIC_API_KEY environment variable is not set.\n"
        return

    prompt = build_report_prompt(summary)

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)

        async with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    except anthropic.APIError as e:
        yield f"\n\n---\n\n**API Error:** {e}\n"
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        yield f"\n\n---\n\n**Error generating report:** {e}\n"


async def generate_report_stream_openai(
    summary: SessionSummary,
) -> AsyncGenerator[str, None]:
    """Generate report using OpenAI API with streaming.

    Args:
        summary: Session summary data.

    Yields:
        Markdown text chunks as they're generated.
    """
    try:
        import openai
    except ImportError:
        yield "# Report Generation Error\n\n"
        yield "The `openai` package is not installed. "
        yield "Install it with: `pip install openai`\n"
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        yield "# Report Generation Error\n\n"
        yield "OPENAI_API_KEY environment variable is not set.\n"
        return

    prompt = build_report_prompt(summary)

    try:
        client = openai.AsyncOpenAI(api_key=api_key)

        stream = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except openai.APIError as e:
        yield f"\n\n---\n\n**API Error:** {e}\n"
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        yield f"\n\n---\n\n**Error generating report:** {e}\n"


async def generate_report_stream_gemini(
    summary: SessionSummary,
) -> AsyncGenerator[str, None]:
    """Generate report using Google Gemini API with streaming.

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

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        yield "# Report Generation Error\n\n"
        yield "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set.\n"
        return

    prompt = build_report_prompt(summary)

    try:
        client = genai.Client(api_key=api_key)

        async for chunk in await client.aio.models.generate_content_stream(
            model="gemini-2.0-flash",
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
    provider: str = "anthropic",
) -> AsyncGenerator[str, None]:
    """Generate a streaming report for an analysis session.

    Args:
        session: Session data from database.
        results: List of inference results.
        annotations: Optional ground truth annotations.
        provider: LLM provider ("anthropic", "openai", or "gemini").

    Yields:
        Markdown text chunks as they're generated.
    """
    summary = build_report_context(session, results, annotations)

    if provider == "openai":
        async for chunk in generate_report_stream_openai(summary):
            yield chunk
    elif provider == "gemini":
        async for chunk in generate_report_stream_gemini(summary):
            yield chunk
    else:
        # Default to Anthropic
        async for chunk in generate_report_stream_anthropic(summary):
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

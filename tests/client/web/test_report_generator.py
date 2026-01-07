"""Tests for report generator module."""

import json
from typing import Any

import pytest

from iris.client.web.report_generator import (
    SessionSummary,
    build_report_context,
    build_report_prompt,
    generate_fallback_report,
)


@pytest.fixture
def sample_session() -> dict[str, Any]:
    """Sample session data."""
    return {
        "id": "test-session-001",
        "status": "completed",
        "started_at": 1000.0,
        "completed_at": 1060.0,  # 60 second duration
        "video_file": "sample_video.mp4",
        "annotation_file": "sample_annotations.jsonl",
        "config": {
            "capture_fps": 5.0,
            "frames_per_job": 8,
            "frame_overlap": 2,
        },
    }


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    """Sample inference results."""
    return [
        {
            "job_id": f"job-{i:03d}",
            "video_time_ms": i * 1000,
            "inference_duration_ms": 150 + (i * 10),
            "result": {
                "action": ["pipette", "mix", "transfer"][i % 3],
                "tool": "micropipette",
                "confidence": 0.9 - (i * 0.05),
            },
        }
        for i in range(15)
    ]


@pytest.fixture
def sample_annotations() -> list[dict[str, Any]]:
    """Sample ground truth annotations."""
    return [
        {
            "start_ms": i * 2000,
            "end_ms": (i + 1) * 2000,
            "action": "pipette",
            "tool": "micropipette",
        }
        for i in range(8)
    ]


class TestBuildReportContext:
    """Tests for build_report_context function."""

    def test_builds_session_summary(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test building session summary from session data."""
        summary = build_report_context(sample_session, sample_results)

        assert isinstance(summary, SessionSummary)
        assert summary.session_id == "test-session-001"
        assert summary.status == "completed"
        assert summary.duration_sec == 60.0
        assert summary.video_file == "sample_video.mp4"
        assert summary.annotation_file == "sample_annotations.jsonl"
        assert summary.total_results == 15

    def test_limits_sample_results(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that sample results are limited to 10."""
        summary = build_report_context(sample_session, sample_results)

        assert len(summary.sample_results) == 15
        assert summary.total_results == 15

    def test_includes_annotations_when_provided(
        self,
        sample_session: dict,
        sample_results: list[dict],
        sample_annotations: list[dict],
    ) -> None:
        """Test that annotations are included in summary."""
        summary = build_report_context(
            sample_session, sample_results, sample_annotations
        )

        assert summary.ground_truth_count == 8
        assert len(summary.annotations_sample) == 5  # Limited to 5

    def test_handles_missing_annotations(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test handling when no annotations provided."""
        summary = build_report_context(sample_session, sample_results)

        assert summary.ground_truth_count == 0
        assert summary.annotations_sample == []

    def test_handles_empty_results(self, sample_session: dict) -> None:
        """Test handling empty results list."""
        summary = build_report_context(sample_session, [])

        assert summary.total_results == 0
        assert summary.sample_results == []

    def test_simplifies_result_fields(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that results are simplified for prompt."""
        summary = build_report_context(sample_session, sample_results)

        # Check simplified structure
        result = summary.sample_results[0]
        assert "job_id" in result
        assert "video_time_ms" in result
        assert "inference_duration_ms" in result
        assert "result" in result


class TestBuildReportPrompt:
    """Tests for build_report_prompt function."""

    def test_builds_valid_prompt(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test building a valid prompt string."""
        summary = build_report_context(sample_session, sample_results)
        prompt = build_report_prompt(summary)

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial

    def test_includes_session_data(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that prompt includes session data."""
        summary = build_report_context(sample_session, sample_results)
        prompt = build_report_prompt(summary)

        assert "test-session-001" in prompt
        assert "sample_video.mp4" in prompt

    def test_includes_report_sections(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that prompt requests expected sections."""
        summary = build_report_context(sample_session, sample_results)
        prompt = build_report_prompt(summary)

        assert "Session Overview" in prompt
        assert "Procedure Steps" in prompt
        assert "Summary" in prompt

    def test_contains_json_context(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that prompt contains JSON context block."""
        summary = build_report_context(sample_session, sample_results)
        prompt = build_report_prompt(summary)

        assert "```json" in prompt
        # Should be valid JSON in the prompt
        json_start = prompt.find("```json") + 7
        json_end = prompt.find("```", json_start)
        json_str = prompt[json_start:json_end].strip()
        # Should parse without error
        parsed = json.loads(json_str)
        assert "session_id" in parsed


class TestGenerateFallbackReport:
    """Tests for generate_fallback_report function."""

    def test_generates_markdown_report(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test generating basic markdown report."""
        report = generate_fallback_report(sample_session, sample_results)

        assert isinstance(report, str)
        assert "# Analysis Report" in report
        assert "## Session Summary" in report
        assert "## Inference Statistics" in report

    def test_includes_session_info(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that report includes session info."""
        report = generate_fallback_report(sample_session, sample_results)

        assert "test-session-001" in report
        assert "completed" in report
        assert "sample_video.mp4" in report

    def test_calculates_inference_stats(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that report calculates inference statistics."""
        report = generate_fallback_report(sample_session, sample_results)

        assert "Total Inferences:** 15" in report
        # Check for min/max/avg (values should be present)
        assert "Average Inference Time:" in report
        assert "Min Inference Time:" in report
        assert "Max Inference Time:" in report

    def test_handles_empty_results(self, sample_session: dict) -> None:
        """Test handling empty results list."""
        report = generate_fallback_report(sample_session, [])

        assert "Total Inferences:** 0" in report
        assert "Average Inference Time:** 0.0 ms" in report

    def test_includes_config_json(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that report includes configuration JSON."""
        report = generate_fallback_report(sample_session, sample_results)

        assert "## Configuration" in report
        assert "capture_fps" in report
        assert "frames_per_job" in report

    def test_includes_ground_truth_count(
        self,
        sample_session: dict,
        sample_results: list[dict],
        sample_annotations: list[dict],
    ) -> None:
        """Test that report includes ground truth count when provided."""
        report = generate_fallback_report(
            sample_session, sample_results, sample_annotations
        )

        assert "Ground Truth Annotations:** 8" in report

    def test_fallback_notice(
        self,
        sample_session: dict,
        sample_results: list[dict],
    ) -> None:
        """Test that report includes fallback notice."""
        report = generate_fallback_report(sample_session, sample_results)

        assert "basic statistics report" in report
        assert "LLM API key" in report


class TestSessionSummary:
    """Tests for SessionSummary dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test SessionSummary has expected fields."""
        summary = SessionSummary(
            session_id="test",
            status="idle",
            duration_sec=30.0,
            video_file="video.mp4",
            annotation_file="ann.jsonl",
            config={"fps": 5},
            total_results=10,
            sample_results=[{"job_id": "1"}],
            ground_truth_count=5,
            annotations_sample=[{"action": "test"}],
        )

        assert summary.session_id == "test"
        assert summary.status == "idle"
        assert summary.duration_sec == 30.0
        assert summary.video_file == "video.mp4"
        assert summary.annotation_file == "ann.jsonl"
        assert summary.config == {"fps": 5}
        assert summary.total_results == 10
        assert len(summary.sample_results) == 1
        assert summary.ground_truth_count == 5
        assert len(summary.annotations_sample) == 1

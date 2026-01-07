"""Sample data fixtures for testing."""

import pytest


@pytest.fixture
def sample_session_data() -> dict:
    """Sample session data for testing."""
    return {
        "id": "test-session-123",
        "config": {
            "capture_fps": 5.0,
            "frames_per_job": 8,
            "frame_overlap": 2,
        },
        "video_file": "test_video.mp4",
        "annotation_file": "test_annotations.jsonl",
    }


@pytest.fixture
def sample_result_data() -> dict:
    """Sample inference result data for testing."""
    return {
        "session_id": "test-session-123",
        "job_id": "job-001",
        "video_time_ms": 1000,
        "inference_start_ms": 1000.0,
        "inference_end_ms": 1150.0,
        "frame_start": 5,
        "frame_end": 12,
        "result": {"action": "pipette", "tool": "micropipette", "confidence": 0.95},
    }

"""Pytest configuration and fixtures."""

import os
import sqlite3
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_db_path(temp_db_path: Path) -> Generator[Path, None, None]:
    """Patch DB_PATH to use temporary database."""
    with patch("iris.client.web.database.DB_PATH", temp_db_path):
        yield temp_db_path


@pytest.fixture
def initialized_db(mock_db_path: Path) -> Generator[Path, None, None]:
    """Initialize database with schema."""
    from iris.client.web.database import init_db

    init_db()
    yield mock_db_path


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

"""Tests for BatchVideoJob."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import MagicMock, Mock

import pytest
from PIL import Image

from iris.server.inference.jobs.batch_video import BatchVideoJob
from iris.server.inference.jobs.base import JobStatus


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = Mock()
    model.config.model_type = "qwen"
    model.device = "cpu"
    return model


@pytest.fixture
def mock_processor():
    """Create a mock processor."""
    processor = Mock()
    processor.apply_chat_template.return_value = "test prompt"
    processor.batch_decode.return_value = ["result1", "result2"]
    return processor


@pytest.fixture
def executor():
    """Create a thread pool executor."""
    return ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def sample_frames():
    """Create sample frame data (JPEG bytes)."""
    frames = []
    for i in range(4):
        img = Image.new("RGB", (100, 100), color=(i * 60, i * 60, i * 60))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        frames.append(buffer.getvalue())
    return frames


def test_batch_video_job_init(executor):
    """Test BatchVideoJob initialization."""
    segments = [
        {"frames": [b"frame1"], "segment_id": "seg_0", "prompt": "test", "client_fps": 5.0},
        {"frames": [b"frame2"], "segment_id": "seg_1", "prompt": "test", "client_fps": 5.0},
    ]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,
        processor=None,
        executor=executor,
        segments=segments,
        max_new_tokens=128,
    )

    assert job.job_id == "test_job"
    assert len(job.segments) == 2
    assert job.max_new_tokens == 128
    assert job.status == JobStatus.PENDING


def test_batch_video_job_segments_data(executor, sample_frames):
    """Test BatchVideoJob correctly stores segment data."""
    segments = [
        {"frames": sample_frames[:2], "segment_id": "seg_0", "prompt": "prompt1", "client_fps": 5.0},
        {"frames": sample_frames[2:], "segment_id": "seg_1", "prompt": "prompt1", "client_fps": 5.0},
    ]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,
        processor=None,
        executor=executor,
        segments=segments,
        max_new_tokens=64,
    )

    assert job.segments[0]["segment_id"] == "seg_0"
    assert job.segments[1]["segment_id"] == "seg_1"
    assert len(job.segments[0]["frames"]) == 2
    assert len(job.segments[1]["frames"]) == 2


@pytest.mark.asyncio
async def test_batch_video_job_requires_model(executor):
    """Test BatchVideoJob fails without model injection."""
    segments = [
        {"frames": [b"frame1"], "segment_id": "seg_0", "prompt": "test", "client_fps": 5.0},
    ]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,  # Not injected
        processor=None,
        executor=executor,
        segments=segments,
    )

    with pytest.raises(RuntimeError, match="Model/processor not injected"):
        await job.execute()

    assert job.status == JobStatus.FAILED


def test_batch_video_job_is_qwen_model(executor, mock_model):
    """Test _is_qwen_model detection."""
    segments = [{"frames": [b"frame1"], "segment_id": "seg_0", "prompt": "test", "client_fps": 5.0}]

    job = BatchVideoJob(
        job_id="test_job",
        model=mock_model,
        processor=None,
        executor=executor,
        segments=segments,
    )

    assert job._is_qwen_model() is True

    # Test with non-Qwen model
    mock_model.config.model_type = "smolvlm"
    assert job._is_qwen_model() is False


def test_batch_size_property(executor):
    """Test batch size is correctly reported."""
    segments = [
        {"frames": [b"frame1"], "segment_id": f"seg_{i}", "prompt": "test", "client_fps": 5.0}
        for i in range(16)
    ]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,
        processor=None,
        executor=executor,
        segments=segments,
    )

    assert len(job.segments) == 16


def test_batch_video_job_callback_assignment(executor):
    """Test callback assignment."""
    segments = [{"frames": [b"frame1"], "segment_id": "seg_0", "prompt": "test", "client_fps": 5.0}]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,
        processor=None,
        executor=executor,
        segments=segments,
    )

    result_callback = Mock()
    log_callback = Mock()

    job.result_callback = result_callback
    job.log_callback = log_callback

    assert job.result_callback == result_callback
    assert job.log_callback == log_callback


def test_segment_id_tracking(executor):
    """Test that segment IDs are tracked correctly."""
    segments = [
        {"frames": [b"f1"], "segment_id": "custom_id_1", "prompt": "p1", "client_fps": 5.0},
        {"frames": [b"f2"], "segment_id": "custom_id_2", "prompt": "p2", "client_fps": 5.0},
        {"frames": [b"f3"], "segment_id": "custom_id_3", "prompt": "p3", "client_fps": 5.0},
    ]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,
        processor=None,
        executor=executor,
        segments=segments,
    )

    assert job.segments[0]["segment_id"] == "custom_id_1"
    assert job.segments[1]["segment_id"] == "custom_id_2"
    assert job.segments[2]["segment_id"] == "custom_id_3"


def test_prompt_handling(executor):
    """Test that prompts are correctly stored per segment."""
    segments = [
        {"frames": [b"f1"], "segment_id": "seg_0", "prompt": "Prompt for segment 0", "client_fps": 5.0},
        {"frames": [b"f2"], "segment_id": "seg_1", "prompt": "Prompt for segment 1", "client_fps": 5.0},
    ]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,
        processor=None,
        executor=executor,
        segments=segments,
    )

    assert job.segments[0]["prompt"] == "Prompt for segment 0"
    assert job.segments[1]["prompt"] == "Prompt for segment 1"


def test_client_fps_handling(executor):
    """Test that client_fps is correctly stored per segment."""
    segments = [
        {"frames": [b"f1"], "segment_id": "seg_0", "prompt": "test", "client_fps": 2.5},
        {"frames": [b"f2"], "segment_id": "seg_1", "prompt": "test", "client_fps": 5.0},
    ]

    job = BatchVideoJob(
        job_id="test_job",
        model=None,
        processor=None,
        executor=executor,
        segments=segments,
    )

    assert job.segments[0]["client_fps"] == 2.5
    assert job.segments[1]["client_fps"] == 5.0

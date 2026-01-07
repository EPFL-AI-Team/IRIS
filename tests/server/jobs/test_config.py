"""Tests for job configuration and types."""

import pytest
from pydantic import ValidationError

from iris.server.jobs.types import JobType, TriggerMode
from iris.server.jobs.config import JobConfig, VideoJobConfig, SingleFrameJobConfig


class TestJobType:
    """Tests for JobType enum."""

    def test_job_type_values(self) -> None:
        """Test JobType enum has expected values."""
        assert JobType.SINGLE_FRAME.value == "single_frame"
        assert JobType.VIDEO.value == "video"
        assert JobType.BATCH_VIDEO.value == "batch_video"

    def test_job_type_string_conversion(self) -> None:
        """Test JobType can be used as string."""
        assert str(JobType.SINGLE_FRAME) == "JobType.SINGLE_FRAME"
        assert JobType.VIDEO == "video"


class TestTriggerMode:
    """Tests for TriggerMode enum."""

    def test_trigger_mode_values(self) -> None:
        """Test TriggerMode enum has expected values."""
        assert TriggerMode.PERIODIC.value == "periodic"
        assert TriggerMode.MANUAL.value == "manual"
        assert TriggerMode.DISABLED.value == "disabled"

    def test_trigger_mode_comparison(self) -> None:
        """Test TriggerMode can be compared to strings."""
        assert TriggerMode.PERIODIC == "periodic"
        assert TriggerMode.MANUAL == "manual"
        assert TriggerMode.DISABLED == "disabled"


class TestJobConfig:
    """Tests for base JobConfig."""

    def test_job_config_requires_job_type(self) -> None:
        """Test JobConfig requires job_type field."""
        config = JobConfig(job_type=JobType.VIDEO)
        assert config.job_type == JobType.VIDEO

    def test_job_config_default_prompt(self) -> None:
        """Test JobConfig has default prompt."""
        config = JobConfig(job_type=JobType.VIDEO)
        assert config.prompt == "Describe what you see in one sentence."

    def test_job_config_custom_prompt(self) -> None:
        """Test JobConfig accepts custom prompt."""
        config = JobConfig(
            job_type=JobType.VIDEO,
            prompt="What tools are visible?",
        )
        assert config.prompt == "What tools are visible?"

    def test_job_config_auto_generates_job_id(self) -> None:
        """Test job_id is None by default (auto-generated later)."""
        config = JobConfig(job_type=JobType.VIDEO)
        assert config.job_id is None

    def test_job_config_custom_job_id(self) -> None:
        """Test JobConfig accepts custom job_id."""
        config = JobConfig(job_type=JobType.VIDEO, job_id="my-custom-job")
        assert config.job_id == "my-custom-job"


class TestSingleFrameJobConfig:
    """Tests for SingleFrameJobConfig."""

    def test_single_frame_config_type(self) -> None:
        """Test SingleFrameJobConfig has correct job_type."""
        config = SingleFrameJobConfig()
        assert config.job_type == JobType.SINGLE_FRAME

    def test_single_frame_config_inherits_prompt(self) -> None:
        """Test SingleFrameJobConfig inherits prompt default."""
        config = SingleFrameJobConfig()
        assert config.prompt == "Describe what you see in one sentence."


class TestVideoJobConfig:
    """Tests for VideoJobConfig."""

    def test_video_config_type(self) -> None:
        """Test VideoJobConfig has correct job_type."""
        config = VideoJobConfig()
        assert config.job_type == JobType.VIDEO

    def test_video_config_has_defaults(self) -> None:
        """Test VideoJobConfig has reasonable defaults."""
        config = VideoJobConfig()

        # Check defaults exist (values may vary based on config.yaml)
        assert isinstance(config.buffer_size, int)
        assert isinstance(config.overlap_frames, int)
        assert isinstance(config.default_fps, (int, float))
        assert isinstance(config.max_new_tokens, int)
        assert isinstance(config.trigger_mode, TriggerMode)

    def test_video_config_buffer_size_validation(self) -> None:
        """Test buffer_size must be >= 1."""
        with pytest.raises(ValidationError):
            VideoJobConfig(buffer_size=0)

        with pytest.raises(ValidationError):
            VideoJobConfig(buffer_size=-1)

        # Valid
        config = VideoJobConfig(buffer_size=1)
        assert config.buffer_size == 1

    def test_video_config_overlap_frames_validation(self) -> None:
        """Test overlap_frames must be >= 0."""
        with pytest.raises(ValidationError):
            VideoJobConfig(overlap_frames=-1)

        # Valid
        config = VideoJobConfig(overlap_frames=0)
        assert config.overlap_frames == 0

    def test_video_config_max_new_tokens_validation(self) -> None:
        """Test max_new_tokens must be >= 1."""
        with pytest.raises(ValidationError):
            VideoJobConfig(max_new_tokens=0)

        # Valid
        config = VideoJobConfig(max_new_tokens=1)
        assert config.max_new_tokens == 1

    def test_video_config_custom_values(self) -> None:
        """Test VideoJobConfig accepts custom values."""
        config = VideoJobConfig(
            buffer_size=16,
            overlap_frames=8,
            default_fps=10.0,
            max_new_tokens=256,
            trigger_mode=TriggerMode.MANUAL,
        )

        assert config.buffer_size == 16
        assert config.overlap_frames == 8
        assert config.default_fps == 10.0
        assert config.max_new_tokens == 256
        assert config.trigger_mode == TriggerMode.MANUAL

    def test_video_config_model_dump(self) -> None:
        """Test VideoJobConfig can be serialized."""
        config = VideoJobConfig(buffer_size=4, overlap_frames=2)
        data = config.model_dump()

        assert data["job_type"] == "video"
        assert data["buffer_size"] == 4
        assert data["overlap_frames"] == 2
        assert "prompt" in data
        assert "trigger_mode" in data

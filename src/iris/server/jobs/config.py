"""Job configuration system with flexible trigger logic."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from iris.config import _yaml_config


class TriggerMode(str, Enum):
    """Trigger modes for video jobs."""

    PERIODIC = "periodic"  # Automatic (frame_count OR time_seconds)
    MANUAL = "manual"      # API call to /jobs/{id}/trigger
    DISABLED = "disabled"  # No automatic triggering (job-to-job only)


class TriggerConfig(BaseModel):
    """Trigger configuration for video jobs.

    Triggers inference based on mode:
    - PERIODIC: Triggers when frame_count OR time_seconds threshold met
    - MANUAL: Only triggers via API call
    - DISABLED: No automatic triggering (for job-to-job orchestration)
    """

    mode: TriggerMode = Field(default=TriggerMode.PERIODIC)
    frame_count: int | None = Field(default=5)       # For periodic mode
    time_seconds: float | None = Field(default=5.0)  # For periodic mode

    def should_trigger(self, frames: int, elapsed: float) -> bool:
        """Check if trigger conditions are met.

        Args:
            frames: Number of frames collected
            elapsed: Elapsed time since collection started

        Returns:
            True if trigger conditions met, False otherwise
        """
        if self.mode != TriggerMode.PERIODIC:
            return False

        count_met = self.frame_count and frames >= self.frame_count
        time_met = self.time_seconds and elapsed >= self.time_seconds
        return count_met or time_met


class JobType(str, Enum):
    """Available job types in the system."""

    SINGLE_FRAME = "single_frame"
    VIDEO = "video"


class JobConfig(BaseModel):
    """Base configuration for all job types."""

    job_type: JobType
    job_id: str | None = None  # Auto-generated if not provided
    prompt: str = Field(default="Describe what you see in one sentence.")


class SingleFrameJobConfig(JobConfig):
    """Configuration for single-frame inference jobs (testing)."""

    job_type: Literal[JobType.SINGLE_FRAME] = JobType.SINGLE_FRAME


class VideoJobConfig(JobConfig):
    """Configuration for video job (buffer + inference).

    Accepts incoming frames, buffers them, and triggers inference based on config.
    """

    job_type: Literal[JobType.VIDEO] = JobType.VIDEO

    # Trigger configuration
    trigger: TriggerConfig = Field(default_factory=TriggerConfig)

    # Frame handling
    frame_skip: int = Field(default=1, ge=1, description="Keep every Nth frame")
    max_buffer_size: int = Field(default=100, description="Prevent infinite growth")

    # Behavior
    continuous: bool = Field(default=True, description="Restart after trigger?")
    log_progress: bool = Field(default=True, description="Send WebSocket logs?")
    log_every_n_frames: int = Field(default=1, description="Log frequency")

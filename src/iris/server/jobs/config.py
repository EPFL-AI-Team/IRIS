"""Job configuration system with flexible trigger logic."""

from typing import Literal

from pydantic import BaseModel, Field

from iris.config import _yaml_config
from iris.server.jobs.types import JobType, TriggerMode


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

    Simplified to 3 core parameters:
    - trigger_mode: When to run inference (PERIODIC/MANUAL/DISABLED)
    - buffer_size: Number of frames before triggering
    - overlap_frames: Frames to keep after inference for continuity
    """

    job_type: Literal[JobType.VIDEO] = JobType.VIDEO

    # Core parameters (only 3!)
    trigger_mode: TriggerMode = Field(default=TriggerMode.PERIODIC)
    buffer_size: int = Field(default=8, ge=1, description="Frames before inference")
    overlap_frames: int = Field(default=4, ge=0, description="Frames to keep after inference")

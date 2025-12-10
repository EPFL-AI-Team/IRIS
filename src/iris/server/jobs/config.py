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

    # Load defaults from project root config.yaml (YAML-only, no env overrides)
    _jobs_video_cfg = _yaml_config.get("jobs", {}).get("video", {})

    trigger_mode: TriggerMode = Field(
        default=_jobs_video_cfg.get("trigger_mode", TriggerMode.PERIODIC),
        description="When to run inference",
    )
    buffer_size: int = Field(
        default=_jobs_video_cfg.get("buffer_size", 8),
        ge=1,
        description="Frames before inference",
    )
    overlap_frames: int = Field(
        default=_jobs_video_cfg.get("overlap_frames", 4),
        ge=0,
        description="Frames to keep after inference",
    )
    sample_fps: int = Field(
        default=_jobs_video_cfg.get("sample_fps", 5),
        ge=0,
        description="Video sampling rate (frames per second) fed to the VLM",
    )
    max_new_tokens: int = Field(
        default=_jobs_video_cfg.get("max_new_tokens", 128),
        ge=1,
        description="Generation length for VLM outputs",
    )

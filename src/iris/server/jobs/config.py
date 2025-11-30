"""Job configuration system with flexible trigger logic."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TriggerConfig(BaseModel):
    """Trigger configuration for frame collection jobs.

    Triggers when EITHER condition is met (whichever comes first):
    - frame_count: Number of frames collected
    - time_seconds: Elapsed time since collection started

    Set disabled=True to prevent auto-triggering.
    """

    frame_count: int | None = None
    time_seconds: float | None = None
    disabled: bool = False

    def should_trigger(self, frames: int, elapsed: float) -> bool:
        """Check if trigger conditions are met.

        Args:
            frames: Number of frames collected
            elapsed: Elapsed time since collection started

        Returns:
            True if trigger conditions met, False otherwise
        """
        if self.disabled:
            return False

        if self.frame_count and frames >= self.frame_count:
            return True
        if self.time_seconds and elapsed >= self.time_seconds:
            return True
        return False


class JobType(str, Enum):
    """Available job types in the system."""

    SINGLE_FRAME = "single_frame"
    FRAME_COLLECTION = "frame_collection"
    VIDEO_INFERENCE = "video_inference"


class JobConfig(BaseModel):
    """Base configuration for all job types."""

    job_type: JobType
    job_id: str | None = None  # Auto-generated if not provided
    prompt: str = Field(default="Describe what you see in one sentence.")


class SingleFrameJobConfig(JobConfig):
    """Configuration for single-frame inference jobs (testing)."""

    job_type: Literal[JobType.SINGLE_FRAME] = JobType.SINGLE_FRAME


class FrameCollectionJobConfig(JobConfig):
    """Configuration for frame collection jobs.

    Continuously collects frames and triggers VideoInferenceJob based on thresholds.
    """

    job_type: Literal[JobType.FRAME_COLLECTION] = JobType.FRAME_COLLECTION
    trigger: TriggerConfig = Field(
        default_factory=lambda: TriggerConfig(
            frame_count=16,
            time_seconds=5.0
        ),
        description="Trigger conditions for batch inference"
    )
    frame_skip: int = Field(
        default=60,
        description="Keep every Nth frame (1 = keep all frames)"
    )
    debug_logging: bool = Field(
        default=False,
        description="Log collection progress"
    )
    continuous: bool = Field(
        default=True,
        description="Restart collection after triggering (True) or stop after one batch (False)"
    )


class VideoInferenceJobConfig(JobConfig):
    """Configuration for video inference jobs.

    Processes batch of frames with VLM (placeholder for future memory buffer features).
    """

    job_type: Literal[JobType.VIDEO_INFERENCE] = JobType.VIDEO_INFERENCE
    max_frames: int = Field(
        default=16,
        description="Maximum frames to process (placeholder for future features)"
    )

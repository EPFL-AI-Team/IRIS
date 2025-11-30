"""Job system for flexible inference task management."""

from iris.server.jobs.config import (
    JobConfig,
    JobType,
    TriggerConfig,
    SingleFrameJobConfig,
    FrameCollectionJobConfig,
    VideoInferenceJobConfig,
)
from iris.server.jobs.factory import JobFactory
from iris.server.jobs.manager import JobManager

__all__ = [
    "JobConfig",
    "JobType",
    "TriggerConfig",
    "SingleFrameJobConfig",
    "FrameCollectionJobConfig",
    "VideoInferenceJobConfig",
    "JobFactory",
    "JobManager",
]

"""Job system for flexible inference task management."""

from iris.server.jobs.config import (
    JobConfig,
    JobType,
    TriggerMode,
    SingleFrameJobConfig,
    VideoJobConfig,
)
from iris.server.jobs.factory import JobFactory
from iris.server.jobs.manager import JobManager

__all__ = [
    "JobConfig",
    "JobType",
    "TriggerMode",
    "SingleFrameJobConfig",
    "VideoJobConfig",
    "JobFactory",
    "JobManager",
]

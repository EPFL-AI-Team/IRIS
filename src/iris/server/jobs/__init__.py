"""Job system for flexible inference task management."""

from iris.server.jobs.config import (
    JobConfig,
    JobType,
    TriggerConfig,
    TriggerMode,
    SingleFrameJobConfig,
    VideoJobConfig,
)
from iris.server.jobs.factory import JobFactory
from iris.server.jobs.manager import JobManager

__all__ = [
    "JobConfig",
    "JobType",
    "TriggerConfig",
    "TriggerMode",
    "SingleFrameJobConfig",
    "VideoJobConfig",
    "JobFactory",
    "JobManager",
]

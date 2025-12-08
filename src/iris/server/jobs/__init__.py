"""Job system for flexible inference task management.

Import from submodules directly:
- from iris.server.jobs.config import JobConfig, VideoJobConfig, etc.
- from iris.server.jobs.types import TriggerMode, JobType
- from iris.server.jobs.factory import JobFactory
- from iris.server.jobs.manager import JobManager
"""

# Re-export only the types to avoid circular imports
from iris.server.jobs.types import JobType, TriggerMode

__all__ = [
    "JobType",
    "TriggerMode",
]

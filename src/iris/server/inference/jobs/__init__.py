"""Job classes for inference execution.

This module contains the job abstraction and concrete implementations
for different types of ML inference tasks.
"""

from iris.server.inference.jobs.base import DummyJob, Job, JobStatus
from iris.server.inference.jobs.single_frame import SingleFrameJob
from iris.server.inference.jobs.video import VideoJob

__all__ = [
    "DummyJob",
    "Job",
    "JobStatus",
    "SingleFrameJob",
    "VideoJob",
]

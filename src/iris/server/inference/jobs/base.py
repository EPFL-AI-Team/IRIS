"""Base job class and status enum for inference jobs.

This module defines the abstract base class that all inference jobs must implement,
along with the job lifecycle states.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class JobStatus(Enum):
    """Job lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(ABC):
    """
    Base class for all inference jobs.

    This allows us to run any ML pipeline on a separate thread while other things are running.
    Jobs are server-side constructs that include callback mechanisms for communicating
    results back to WebSocket clients.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.submitted_at = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.result: Any = None
        self.error: str | None = None
        self.processing_time: float = 0.0

    def format_result(self) -> str:
        """
        Provides a standard, one-line summary for a completed job.
        Subclasses should override this to provide more detailed output.
        """
        header = f"Job Completed: {self.job_id} ({self.job_type})"
        separator = "-" * (len(header) + 4)
        return f"\n\n{header}\n{separator}\n  - Result: {self.result}\n{separator}\n"

    @abstractmethod
    async def execute(self) -> Any:
        """
        DO THE WORK.

        This method should run the job and store its output
        in 'self.result' or 'self.error'.
        """
        pass

    @abstractmethod
    def to_response_dict(self) -> dict:
        """Serialize job-specific data for WebSocket response.

        This method allows each job type to define its own response format
        while maintaining a consistent base structure in the WebSocket handler.

        Returns:
            Dictionary with job-specific data (result, metrics, etc.)
        """
        pass

    @property
    def job_type(self) -> str:
        """Return class name for logging."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.job_type}(id={self.job_id}, status={self.status.value})"


class DummyJob(Job):
    """Simple test job for benchmarking and testing."""

    def __init__(self, job_id: str, sleep_time: float = 1.0):
        super().__init__(job_id)
        self.sleep_time = sleep_time

    async def execute(self) -> str:
        """Simulate work."""
        self.status = JobStatus.RUNNING
        self.started_at = time.time()

        # Simulate blocking work
        await asyncio.sleep(self.sleep_time)

        self.status = JobStatus.COMPLETED
        self.completed_at = time.time()
        return f"Job {self.job_id} completed"

    def to_response_dict(self) -> dict:
        """Serialize DummyJob data for response."""
        return {
            "result": self.result,
            "sleep_time": self.sleep_time,
        }

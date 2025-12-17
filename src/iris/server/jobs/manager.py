"""Job lifecycle management and frame routing."""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PIL import Image

from iris.server.jobs.config import JobConfig
from iris.server.jobs.factory import JobFactory
from iris.vlm.inference.queue.jobs import Job

if TYPE_CHECKING:
    from iris.server.dependencies import ServerState

logger = logging.getLogger(__name__)


class JobManager:
    """Centralized job lifecycle and frame routing management.

    Responsibilities:
    - Track active jobs
    - Route frames to jobs that accept them
    - Provide start/stop/status API
    - Thread-safe operations with asyncio.Lock
    """

    def __init__(self, state: "ServerState"):
        """Initialize JobManager with server state.

        Args:
            state: ServerState singleton with model, processor, queue
        """
        self.state = state
        self.active_jobs: dict[str, Job] = {}
        self.lock = asyncio.Lock()
        self.log_callbacks: list = []  # Callable[[dict], None]

    def register_log_callback(self, callback: Callable[[dict], None]) -> None:
        """Register callback for WebSocket logging.

        Args:
            callback: Function that accepts dict with log message
        """
        self.log_callbacks.append(callback)

    async def start_job(self, config: JobConfig) -> str:
        """Create and start a job, return job_id.

        Args:
            config: Job configuration (validated Pydantic model)

        Returns:
            job_id of the started job

        Raises:
            RuntimeError: If job submission fails (queue full)
        """
        if self.state.shutting_down:
            logger.warning(
                "Rejecting start_job during shutdown: job_type=%s", config.job_type
            )
            raise RuntimeError("Server shutting down")

        async with self.lock:
            # Create job using factory
            job = JobFactory.create_job(
                config=config,
                model=self.state.model,
                processor=self.state.processor,
                executor=self.state.queue.executor,
                queue=self.state.queue,
            )

            # Set log callback if job supports it
            if hasattr(job, "log_callback"):
                job.log_callback = lambda msg: self._broadcast_log(msg)

            # Add to active jobs
            self.active_jobs[job.job_id] = job

            # Submit to queue for execution
            submitted = await self.state.queue.submit(job)
            if not submitted:
                del self.active_jobs[job.job_id]
                raise RuntimeError("Failed to submit job: queue full")

            logger.info(f"Started job: {job.job_id} ({config.job_type})")

            # Broadcast job start to any WebSocket listeners
            self._broadcast_log({
                "type": "log",
                "job_id": job.job_id,
                "message": f"Job started: {job.job_id} ({config.job_type})",
                "timestamp": time.time(),
            })
            return job.job_id

    def get_job(self, job_id: str) -> Job | None:
        """Get job instance by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job instance or None if not found
        """
        return self.active_jobs.get(job_id)

    async def stop_job(self, job_id: str) -> bool:
        """Stop a running job.

        Args:
            job_id: Job identifier

        Returns:
            True if job stopped successfully, False if job not found

        Raises:
            ValueError: If job cannot be stopped (already queued)
        """
        async with self.lock:
            job = self.active_jobs.get(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found in active_jobs")
                return False

            # Stop the job
            if hasattr(job, "stop"):
                job.stop()
                logger.info(f"Called stop() on {job_id}")

            # CRITICAL: Remove from active_jobs immediately
            del self.active_jobs[job_id]
            logger.info(f"Stopped job: {job_id}")
            return True

    async def stop_all_jobs(self) -> dict[str, Any]:
        """Stop all active jobs.

        Returns:
            Dictionary with count of stopped jobs and any errors
        """
        async with self.lock:
            job_ids = list(self.active_jobs.keys())
            stopped_count = 0
            errors = []

            for job_id in job_ids:
                try:
                    job = self.active_jobs.get(job_id)
                    if job:
                        # Stop the job if it supports it
                        if hasattr(job, "stop"):
                            job.stop()

                        # Remove from active_jobs
                        del self.active_jobs[job_id]
                        stopped_count += 1
                        logger.info(f"Stopped job: {job_id}")
                except Exception as e:
                    error_msg = f"Failed to stop job {job_id}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            return {
                "stopped_count": stopped_count,
                "errors": errors
            }

    async def route_frame(
        self,
        frame: Image.Image,
        frame_id: int,
        timestamp: float,
        client_fps: float | None = None,
    ) -> None:
        """Route incoming frame to all active jobs that accept frames.

        Args:
            frame: PIL Image to route
            frame_id: Frame identifier for logging
            timestamp: Frame arrival timestamp
            client_fps: Capture FPS reported by client (optional)
        """
        async with self.lock:
            for job_id, job in list(self.active_jobs.items()):
                # Only route to jobs that accept frames
                if job.accepts_frames():
                    try:
                        await job.add_frame(
                            frame, frame_id, timestamp, client_fps=client_fps
                        )
                    except Exception as e:
                        logger.error(
                            f"Error routing frame {frame_id} to job {job_id}: {e}",
                            exc_info=True,
                        )

    async def get_job_status(self, job_id: str) -> dict | None:
        """Get current status of a specific job.

        Args:
            job_id: Job identifier

        Returns:
            Status dict or None if job not found
        """
        async with self.lock:
            job = self.active_jobs.get(job_id)
            if not job:
                return None

            status = {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "status": job.status.value,
                "started_at": job.started_at,
            }

            # Add job-specific status info if available
            if hasattr(job, "get_status_dict"):
                status.update(job.get_status_dict())

            return status

    async def list_active_jobs(self) -> list[dict]:
        """List all active jobs.

        Returns:
            List of job status dicts
        """
        async with self.lock:
            return [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "status": job.status.value,
                    "started_at": job.started_at,
                }
                for job in self.active_jobs.values()
            ]

    async def cleanup_completed_jobs(self) -> None:
        """Remove completed jobs from active tracking.

        This should be called periodically to prevent memory leaks.
        """
        async with self.lock:
            completed = [
                job_id
                for job_id, job in self.active_jobs.items()
                if job.status.value in ["completed", "failed"]
                and not job.accepts_frames()
            ]
            for job_id in completed:
                del self.active_jobs[job_id]
                logger.debug(f"Cleaned up completed job: {job_id}")

    def _broadcast_log(self, log_message: dict) -> None:
        """Send log message to all registered callbacks.

        Args:
            log_message: Dict with 'type', 'job_id', 'message', 'timestamp'
        """
        for callback in self.log_callbacks:
            callback(log_message)

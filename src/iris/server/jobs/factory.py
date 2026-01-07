"""Job factory for creating job instances from configuration."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from iris.server.jobs.config import (
    JobConfig,
    VideoJobConfig,
)
from iris.server.jobs.types import JobType

# Note: VideoJob is imported locally in _create_video_job to avoid circular imports


class JobFactory:
    """Factory for creating job instances based on configuration."""

    @staticmethod
    def create_job(
        config: JobConfig,
        model: Any | None,
        processor: Any | None,
        executor: ThreadPoolExecutor,
        frames: list | None = None,
    ) -> Any:
        """Create job instance based on config.job_type.

        Args:
            config: Job configuration (validated Pydantic model)
            model: VLM model instance (ignored for VideoJob - injected by worker)
            processor: Model processor (ignored for VideoJob - injected by worker)
            executor: ThreadPoolExecutor for blocking GPU work
            frames: List of PIL Image frames (required for VideoJob)

        Returns:
            Job instance ready to be submitted to queue

        Raises:
            ValueError: If job type is unknown
            NotImplementedError: If job type requires additional data at creation
        """
        # Generate job_id if not provided
        if not config.job_id:
            config.job_id = f"{config.job_type.value}-{uuid.uuid4().hex[:8]}"

        # Dispatch based on job type
        if config.job_type == JobType.SINGLE_FRAME:
            # SingleFrameJob requires a frame at creation - must be created manually
            raise NotImplementedError(
                "SingleFrameJob requires a frame at creation time. "
                "Create it directly instead of using the factory."
            )
        elif config.job_type == JobType.VIDEO:
            if not isinstance(config, VideoJobConfig):
                raise ValueError("Invalid config type for VIDEO job")
            return JobFactory._create_video_job(
                config, model, processor, executor, frames or []
            )
        else:
            raise ValueError(f"Unknown job type: {config.job_type}")

    @staticmethod
    def _create_video_job(
        config: VideoJobConfig,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        frames: list,
    ) -> Any:
        """Create VideoJob from simplified configuration."""
        # Import here to avoid circular dependencies
        from iris.server.inference.jobs import VideoJob

        # job_id is guaranteed to be set by create_job(), but add fallback for type safety
        job_id = config.job_id or f"video-{uuid.uuid4().hex[:8]}"

        return VideoJob(
            job_id=job_id,
            model=None,  # Will be injected by worker
            processor=None,  # Will be injected by worker
            executor=executor,
            frames=frames,
            prompt=config.prompt,
            buffer_size=config.buffer_size,
            overlap_frames=config.overlap_frames,
            default_fps=config.default_fps,
            client_fps=config.default_fps,
            max_new_tokens=config.max_new_tokens,
        )

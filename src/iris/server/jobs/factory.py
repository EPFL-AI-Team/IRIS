"""Job factory for creating job instances from configuration."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from iris.server.jobs.config import (
    JobConfig,
    VideoJobConfig,
)
from iris.server.jobs.types import JobType

# if TYPE_CHECKING:
#     from iris.vlm.inference.queue.jobs import VideoJob
#     from iris.vlm.inference.queue.queue import InferenceQueue


class JobFactory:
    """Factory for creating job instances based on configuration."""

    @staticmethod
    def create_job(
        config: JobConfig,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        queue: Any,  # InferenceQueue
    ) -> Any:
        """Create job instance based on config.job_type.

        Args:
            config: Job configuration (validated Pydantic model)
            model: VLM model instance
            processor: Model processor
            executor: ThreadPoolExecutor for blocking GPU work
            queue: InferenceQueue for submitting triggered jobs

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
                config, model, processor, executor, queue
            )
        else:
            raise ValueError(f"Unknown job type: {config.job_type}")

    @staticmethod
    def _create_video_job(
        config: VideoJobConfig,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        queue: Any,  # InferenceQueue
    ) -> Any:
        """Create VideoJob from simplified configuration."""
        # Import here to avoid circular dependencies
        from iris.vlm.inference.queue.jobs import VideoJob

        # job_id is guaranteed to be set by create_job(), but add fallback for type safety
        job_id = config.job_id or f"video-{uuid.uuid4().hex[:8]}"

        return VideoJob(
            job_id=job_id,
            model=model,
            processor=processor,
            executor=executor,
            queue=queue,
            prompt=config.prompt,
            trigger_mode=config.trigger_mode,
            buffer_size=config.buffer_size,
            overlap_frames=config.overlap_frames,
            sample_fps=config.sample_fps,
            max_new_tokens=config.max_new_tokens,
        )

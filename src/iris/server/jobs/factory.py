"""Job factory for creating job instances from configuration."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from iris.server.jobs.config import (
    JobConfig,
    JobType,
    FrameCollectionJobConfig,
)
from iris.vlm.inference.queue.queue import InferenceQueue


class JobFactory:
    """Factory for creating job instances based on configuration."""

    @staticmethod
    def create_job(
        config: JobConfig,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        queue: InferenceQueue,
    ):
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
        # Import here to avoid circular dependencies
        from iris.vlm.inference.queue.jobs import FrameCollectionJob

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
        elif config.job_type == JobType.FRAME_COLLECTION:
            return JobFactory._create_frame_collection_job(
                config, model, processor, executor, queue
            )
        elif config.job_type == JobType.VIDEO_INFERENCE:
            # VideoInferenceJob requires frames at creation - created by FrameCollectionJob
            raise NotImplementedError(
                "VideoInferenceJob requires frames at creation time. "
                "It should be created by FrameCollectionJob when triggered."
            )
        else:
            raise ValueError(f"Unknown job type: {config.job_type}")

    @staticmethod
    def _create_frame_collection_job(
        config: FrameCollectionJobConfig,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        queue: InferenceQueue,
    ):
        """Create FrameCollectionJob from configuration."""
        # Import here to avoid circular dependencies
        from iris.vlm.inference.queue.jobs import FrameCollectionJob

        return FrameCollectionJob(
            job_id=config.job_id,
            trigger=config.trigger,
            frame_skip=config.frame_skip,
            prompt=config.prompt,
            model=model,
            processor=processor,
            executor=executor,
            queue=queue,
            debug_logging=config.debug_logging,
            continuous=config.continuous,
        )

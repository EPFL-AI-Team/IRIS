import asyncio
import base64
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

if TYPE_CHECKING:
    from iris.server.jobs.config import TriggerConfig
    from iris.vlm.inference.queue.queue import InferenceQueue

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job lifecycle states"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(ABC):
    """
    Base class for all inference jobs.

    This allows us to run any ML pipeline on a separate thread while other things are running.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.submitted_at = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None
        self.result: Any = None
        self.error: str | None = None
        self.processing_time: float = 0.0

    def format_result(self) -> str:
        """
        Provides a standard, one-line summary for a completed job.
        Subclasses should override this to provide more detailed output.
        """
        # This acts as a default for any job that doesn't have a custom formatter.
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

    def accepts_frames(self) -> bool:
        """Override this to indicate job accepts frame routing from JobManager.

        Returns:
            True if job accepts frames via add_frame(), False otherwise
        """
        return False

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
        """Return class name for logging"""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.job_type}(id={self.job_id}, status={self.status.value})"


class DummyJob(Job):
    """Simple test job"""

    def __init__(self, job_id: str, sleep_time: float = 1.0):
        super().__init__(job_id)
        self.sleep_time = sleep_time

    async def execute(self) -> str:
        """Simulate work"""
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


class SingleFrameJob(Job):
    """
    Process one frame with a VLM
    """

    def __init__(
        self,
        job_id: str,
        frame: Image.Image,
        model: Any,
        processor: Any,
        prompt: str,  # Specific to SingleFrameJob
        executor: ThreadPoolExecutor,
        received_at: float,
    ):
        super().__init__(job_id)

        # Store everything we need for inference
        self.frame = frame
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.executor = executor  # For running blocking code
        self.received_at = received_at  # Store arrival time
        self.total_latency: float = 0.0  # Will store (completed - received)
        self.frame_b64: str | None = None  # Store frame as base64 for response

    def format_result(self) -> str:
        """
        Overrides the base method to provide a detailed, structured
        output specific to a SingleFrameJob.
        """
        header = f"Job Completed: {self.job_id} ({self.job_type})"
        separator = "-" * (len(header) + 4)

        # Clean up any weird whitespace from the model's output
        clean_result = str(self.result).strip().split("Assistant:")[-1]

        # Build the final, clean output block as a single string
        return (
            f"\n\n{header}\n{separator}\n"
            f"  - Processing Time: {self.processing_time:.2f} seconds\n"
            # f'  - Prompt: "{self.prompt}"\n'
            f"  - Result: {clean_result}\n"
            f"{separator}\n"
        )

    async def execute(self) -> None:
        """
        Coordinate the inference and store the result in self.result.
        """

        self.status = JobStatus.RUNNING
        self.started_at = time.time()
        loop = asyncio.get_event_loop()

        # Run blocking work in its own thread
        inference_output = await loop.run_in_executor(
            self.executor,  # Which thread to pool
            self._sync_inference,  # Non-async function to run
        )

        self.result = inference_output
        self.completed_at = time.time()
        self.status = JobStatus.COMPLETED
        self.processing_time = self.completed_at - self.started_at
        self.total_latency = self.completed_at - self.received_at

        # Clear data to save memory (not necessary for single frame)
        # self.frame = None

    def _sync_inference(self) -> str:
        """
        Does the GPU work (blocking)
        Runs in a separate thread so that its blocking nature won't interfere with other computations
        """
        logger.info(f"WORKER: Starting inference for {self.job_id}")

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.frame},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text], images=[self.frame], return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=128)

            # Slice the output to remove input tokens (cleaner than text parsing)
            generated_ids = outputs[0][len(inputs.input_ids[0]) :]

            # Decode only the new tokens
            result = self.processor.decode(generated_ids, skip_special_tokens=True)

        # Store frame as base64 for response
        buffer = BytesIO()
        self.frame.save(buffer, format="JPEG")
        self.frame_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.info(f"WORKER: Finished inference for {self.job_id}")
        return result

    def to_response_dict(self) -> dict:
        """Serialize SingleFrameJob data for WebSocket response."""
        return {
            "result": self.result,
            "frame": self.frame_b64,
            "metrics": {
                "inference_time": self.processing_time,
                "total_latency": self.total_latency,
                "received_at": self.received_at,
            },
        }


class FrameCollectionJob(Job):
    """Collects frames and triggers VideoInferenceJob based on thresholds.

    This job continuously buffers incoming frames and triggers batch inference
    when specified conditions are met (frame count OR elapsed time).
    """

    def __init__(
        self,
        job_id: str,
        trigger: "TriggerConfig",
        frame_skip: int,
        prompt: str,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        queue: "InferenceQueue",
        debug_logging: bool = False,
        continuous: bool = True,
    ):
        super().__init__(job_id)
        self.trigger = trigger
        self.frame_skip = frame_skip
        self.prompt = prompt
        self.model = model
        self.processor = processor
        self.executor = executor
        self.queue = queue
        self.debug_logging = debug_logging
        self.continuous = continuous

        # Collection state
        self.frame_buffer: list[Image.Image] = []
        self.buffer_start_time: float | None = None
        self.frame_counter = 0
        self.running = True
        self.triggered_count = 0

    async def execute(self) -> None:
        """Main loop - waits for frames to be added via add_frame()."""
        self.status = JobStatus.RUNNING
        self.started_at = time.time()

        # This job runs indefinitely until stopped
        # Frame addition happens via add_frame() called from JobManager
        while self.running:
            await asyncio.sleep(0.1)

        self.status = JobStatus.COMPLETED
        self.completed_at = time.time()
        self.processing_time = self.completed_at - self.started_at
        self.result = f"Collection stopped. Triggered {self.triggered_count} times."

    async def add_frame(
        self, frame: Image.Image, frame_id: int, timestamp: float
    ) -> None:
        """Called by JobManager to add frame to buffer.

        Args:
            frame: PIL Image to add to buffer
            frame_id: Frame identifier for logging
            timestamp: Frame arrival timestamp
        """
        self.frame_counter += 1

        # Apply frame skip
        if self.frame_counter % self.frame_skip != 0:
            return

        # Start buffer timer on first frame
        if not self.frame_buffer:
            self.buffer_start_time = timestamp

        self.frame_buffer.append(frame)

        if self.debug_logging:
            elapsed = timestamp - self.buffer_start_time
            logger.info(
                "[%s] Collected %d frames (%.1fs elapsed)",
                self.job_id,
                len(self.frame_buffer),
                elapsed,
            )

        # Check trigger conditions
        elapsed = timestamp - self.buffer_start_time
        if self.trigger.should_trigger(len(self.frame_buffer), elapsed):
            await self._trigger_inference()

    async def _trigger_inference(self) -> None:
        """Create and submit VideoInferenceJob when trigger conditions met."""
        self.triggered_count += 1

        logger.info(
            "[%s] Trigger met. Submitting VideoInferenceJob with %d frames",
            self.job_id,
            len(self.frame_buffer),
        )

        # Create VideoInferenceJob with buffered frames
        video_job = VideoInferenceJob(
            job_id=f"{self.job_id}-video-{self.triggered_count}",
            frames=self.frame_buffer.copy(),
            model=self.model,
            processor=self.processor,
            prompt=self.prompt,
            executor=self.executor,
        )

        # Submit to queue
        submitted = await self.queue.submit(video_job)
        if not submitted:
            logger.warning("[%s] Queue full, dropped video job", self.job_id)

        # Reset or stop
        if self.continuous:
            self.frame_buffer.clear()
            self.buffer_start_time = None
        else:
            self.running = False

    def stop(self) -> None:
        """Stop frame collection."""
        logger.info("[%s] Stopping frame collection", self.job_id)
        self.running = False

    def accepts_frames(self) -> bool:
        """Indicates this job accepts frame routing."""
        return True

    def get_status_dict(self) -> dict:
        """Get current status details for API responses."""
        elapsed = time.time() - self.buffer_start_time if self.buffer_start_time else 0
        return {
            "frames_collected": len(self.frame_buffer),
            "triggered_count": self.triggered_count,
            "elapsed_seconds": elapsed,
        }

    def to_response_dict(self) -> dict:
        """Serialize FrameCollectionJob data for WebSocket response."""
        return {
            "result": self.result,
            "frames_collected": len(self.frame_buffer),
            "triggered_count": self.triggered_count,
        }


class VideoInferenceJob(Job):
    """Process batch of frames with VLM.

    Placeholder implementation for future memory buffer features:
    - Track visual memory across frames
    - Use temporal context for understanding
    - Implement attention-based frame selection

    Currently processes first frame only.
    """

    def __init__(
        self,
        job_id: str,
        frames: list[Image.Image],
        model: Any,
        processor: Any,
        prompt: str,
        executor: ThreadPoolExecutor,
    ):
        super().__init__(job_id)
        self.frames = frames
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.executor = executor
        self.total_latency: float = 0.0

    async def execute(self) -> None:
        """Run batch inference in ThreadPoolExecutor."""
        self.status = JobStatus.RUNNING
        self.started_at = time.time()
        loop = asyncio.get_event_loop()

        # Run blocking inference in thread
        inference_output = await loop.run_in_executor(
            self.executor,
            self._sync_batch_inference,
        )

        self.result = inference_output
        self.completed_at = time.time()
        self.status = JobStatus.COMPLETED
        self.processing_time = self.completed_at - self.started_at
        self.total_latency = self.processing_time

        # Clear frames to save memory
        num_frames = len(self.frames)
        self.frames = []
        logger.info(f"[{self.job_id}] Cleared {num_frames} frames from memory")

    def _sync_batch_inference(self) -> str:
        """Batch inference on video frames (blocking GPU work).

        TODO: Implement memory buffer feature
        - Track visual memory across frames
        - Use temporal context for understanding
        - Implement attention-based frame selection
        - Round decayed compression for memory efficiency

        Current: Process first frame only (placeholder)
        """
        logger.info(
            f"WORKER: Batch inference for {self.job_id} ({len(self.frames)} frames)"
        )

        # PLACEHOLDER: Process first frame only
        frame = self.frames[0]

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text], images=[frame], return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=128)

            # Slice the output to remove input tokens
            generated_ids = outputs[0][len(inputs.input_ids[0]) :]

            # Decode only the new tokens
            result = self.processor.decode(generated_ids, skip_special_tokens=True)

        logger.info(f"WORKER: Finished batch inference for {self.job_id}")
        return result

    def to_response_dict(self) -> dict:
        """Serialize VideoInferenceJob data for WebSocket response."""
        return {
            "result": self.result,
            "frames_processed": len(self.frames) if self.frames else 0,
            "metrics": {
                "inference_time": self.processing_time,
                "total_latency": self.total_latency,
            },
        }

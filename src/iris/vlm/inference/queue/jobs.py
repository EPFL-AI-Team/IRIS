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

        # Logging callback (set by JobManager)
        self.log_callback: Any = None  # Callable[[dict], None]

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

    def _send_log(self, message: str) -> None:
        """Send WebSocket log message."""
        if self.log_callback:
            self.log_callback({
                "type": "log",
                "job_id": self.job_id,
                "message": message,
                "timestamp": time.time(),
            })


class VideoJob(Job):
    """Unified video job: buffer frames + batch inference.

    Accepts incoming frames, buffers them, and triggers inference based on:
    - Periodic: frame_count OR time_seconds threshold
    - Manual: API call to /jobs/{job_id}/trigger
    - Disabled: Only triggered by other jobs

    After inference, can optionally continue buffering (continuous mode).
    """

    def __init__(
        self,
        job_id: str,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        queue: "InferenceQueue",
        prompt: str = "Describe what you see in the video.",
        trigger: "TriggerConfig" = None,
        frame_skip: int = 1,
        max_buffer_size: int = 100,
        continuous: bool = True,
        log_progress: bool = True,
        log_every_n_frames: int = 1,
    ):
        super().__init__(job_id)
        self.model = model
        self.processor = processor
        self.executor = executor
        self.queue = queue
        self.prompt = prompt

        # Import TriggerConfig here to avoid circular import
        if trigger is None:
            from iris.server.jobs.config import TriggerConfig
            trigger = TriggerConfig()
        self.trigger = trigger

        self.frame_skip = frame_skip
        self.max_buffer_size = max_buffer_size
        self.continuous = continuous
        self.log_progress = log_progress
        self.log_every_n_frames = log_every_n_frames

        # Frame buffer
        self.frame_buffer: list[Image.Image] = []
        self.frame_counter = 0
        self.buffer_start_time: float | None = None

        # Control
        self.frame_event = asyncio.Event()
        self.stop_event = asyncio.Event()
        self.manual_trigger_event = asyncio.Event()

        # Logging callback (set by JobManager)
        self.log_callback: Any = None  # Callable[[dict], None]

    def accepts_frames(self) -> bool:
        """This job accepts incoming frames."""
        return True

    async def add_frame(self, frame: Image.Image, frame_id: int, timestamp: float) -> None:
        """Buffer incoming frame and check trigger conditions."""
        # Apply frame skip
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return

        # Start timing on first frame
        if self.buffer_start_time is None:
            self.buffer_start_time = timestamp

        # Add to buffer (enforce max size with FIFO)
        if len(self.frame_buffer) >= self.max_buffer_size:
            self.frame_buffer.pop(0)
        self.frame_buffer.append(frame.copy())

        # Send WebSocket log
        if self.log_progress and self.frame_counter % self.log_every_n_frames == 0:
            self._send_log(
                f"Buffered frame {len(self.frame_buffer)}/{self.trigger.frame_count or 'N/A'}"
            )

        # Signal frame received
        self.frame_event.set()

    async def trigger_inference(self) -> None:
        """Manually trigger inference (API call)."""
        from iris.server.jobs.config import TriggerMode

        if self.trigger.mode == TriggerMode.MANUAL or self.trigger.mode == TriggerMode.DISABLED:
            self.manual_trigger_event.set()

    async def execute(self) -> None:
        """Main loop: wait for frames, check triggers, run inference."""
        self.status = JobStatus.RUNNING
        logger.info(f"VideoJob {self.job_id} started (trigger={self.trigger.mode})")

        while not self.stop_event.is_set():
            try:
                # Wait for frame or stop signal (with timeout to check stop_event)
                await asyncio.wait_for(self.frame_event.wait(), timeout=1.0)
                self.frame_event.clear()
            except asyncio.TimeoutError:
                continue

            # Check if we should trigger
            elapsed = time.time() - self.buffer_start_time if self.buffer_start_time else 0
            should_auto_trigger = self.trigger.should_trigger(len(self.frame_buffer), elapsed)
            should_manual_trigger = self.manual_trigger_event.is_set()

            if should_auto_trigger or should_manual_trigger:
                self.manual_trigger_event.clear()

                # Log trigger
                self._send_log(f"Collected {len(self.frame_buffer)} frames, launching inference")

                # Run inference on buffered frames
                await self._run_inference()

                # Reset buffer if continuous
                if self.continuous:
                    self.frame_buffer.clear()
                    self.buffer_start_time = None
                else:
                    break

        self.status = JobStatus.COMPLETED
        logger.info(f"VideoJob {self.job_id} completed")

    async def _run_inference(self) -> None:
        """Run inference on buffered frames in ThreadPoolExecutor."""
        if not self.frame_buffer:
            return

        # Copy frames for inference
        frames = self.frame_buffer.copy()

        # Run blocking GPU work in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._sync_inference,
            frames,
            self.prompt,
        )

        # Store result
        self.result = result
        self._send_log(f"Inference completed: {len(frames)} frames processed")

    def _sync_inference(self, frames: list[Image.Image], prompt: str) -> str:
        """Blocking GPU inference (runs in ThreadPoolExecutor).

        TODO: User needs to explore Qwen video prompt template.
        Current implementation is placeholder - may need different format for video.
        Qwen2.5-VL might have native video support with special tokens.

        For now, processes only the first frame as a simple baseline.
        """
        if not frames:
            return "No frames to process"

        # Simple single-frame inference for now
        # TODO: Replace with proper video inference once Qwen template is figured out
        logger.info(f"WORKER: VideoJob inference for {self.job_id} ({len(frames)} frames)")

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frames[0]},
                        {"type": "text", "text": f"{prompt} (Processing {len(frames)} frames)"},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text], images=[frames[0]], return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(**inputs, max_new_tokens=128)

            # Slice the output to remove input tokens
            generated_ids = outputs[0][len(inputs.input_ids[0]) :]

            # Decode only the new tokens
            result = self.processor.decode(generated_ids, skip_special_tokens=True)

        logger.info(f"WORKER: Finished VideoJob inference for {self.job_id}")
        return result

    def _send_log(self, message: str) -> None:
        """Send WebSocket log message."""
        if self.log_callback:
            self.log_callback({
                "type": "log",
                "job_id": self.job_id,
                "message": message,
                "timestamp": time.time(),
            })

    def stop(self) -> None:
        """Stop buffering and exit."""
        self.stop_event.set()

    def get_status_dict(self) -> dict:
        """Return custom status information."""
        return {
            "frames_buffered": len(self.frame_buffer),
            "trigger_mode": str(self.trigger.mode),
            "continuous": self.continuous,
        }

    def to_response_dict(self) -> dict:
        """Serialize VideoJob data for WebSocket response."""
        return {
            "result": self.result,
            "frames_processed": len(self.frame_buffer),
        }

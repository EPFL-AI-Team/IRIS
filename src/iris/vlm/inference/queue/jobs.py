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
from qwen_vl_utils import process_vision_info

if TYPE_CHECKING:
    from iris.vlm.inference.queue.queue import InferenceQueue

from iris.server.jobs.types import TriggerMode

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
        self.completed_at: float | None = None
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
    """Simplified video job: buffer frames + batch inference.

    Accepts incoming frames, buffers them, and triggers inference based on mode:
    - PERIODIC: Auto-trigger when buffer reaches buffer_size
    - MANUAL: Trigger only via API call
    - DISABLED: Buffer but never process

    Uses overlap_frames for temporal continuity between inferences.
    """

    def __init__(
        self,
        job_id: str,
        model: Any,
        processor: Any,
        executor: ThreadPoolExecutor,
        queue: "InferenceQueue",
        prompt: str = """<video>\nAnalyze this video segment of a biological experiment. Output a valid JSON object with the following keys:\n- action: The specific movement (e.g., streaking, inspecting).\n- tool: The active instrument.\n- target: The object being acted upon.\n- context: The protocol step.\n- hand: Active hand (left/right).\n- sample_id: Any handwritten text or labels visible on the object (or "none").\n- notes: Visual details like agar color or colony presence.""",
        trigger_mode: TriggerMode | None = None,
        buffer_size: int = 8,
        overlap_frames: int = 4,
        default_fps: float = 5.0,
        max_new_tokens: int = 128,
        client_fps: float | None = None,
    ):
        super().__init__(job_id)
        self.model = model
        self.processor = processor
        self.executor = executor
        self.queue = queue
        self.prompt = prompt

        # Set default trigger mode if not provided
        if trigger_mode is None:
            trigger_mode = TriggerMode.PERIODIC
        self.trigger_mode = trigger_mode

        self.buffer_size = buffer_size
        self.overlap_frames = overlap_frames
        self.default_fps = float(default_fps)
        self.client_fps = (
            float(client_fps) if client_fps is not None else self.default_fps
        )
        self.max_new_tokens = max_new_tokens

        # Minimal state
        self.frame_buffer: list[Image.Image] = []
        self.stop_event = asyncio.Event()
        self.log_callback: Any = None  # Callable[[dict], None]
        self.result_callback: Any = (
            None  # Callable[[dict], None] - for real-time result broadcasting
        )

    def accepts_frames(self) -> bool:
        """This job accepts incoming frames."""
        return True

    async def add_frame(
        self,
        frame: Image.Image,
        frame_id: int,
        timestamp: float,
        client_fps: float | None = None,
    ) -> None:
        """Buffer frame and auto-trigger based on mode.

        PERIODIC: Auto-trigger when buffer reaches buffer_size
        MANUAL: Buffer and wait for API trigger
        DISABLED: Buffer but never process
        """
        # Update client FPS if provided; fallback to previous or default
        if client_fps is not None:
            self.client_fps = float(client_fps)
        elif self.client_fps is None:
            self.client_fps = self.default_fps
        self.frame_buffer.append(frame.copy())

        if self.trigger_mode == TriggerMode.PERIODIC:
            if len(self.frame_buffer) >= self.buffer_size:
                self._send_log(
                    f"Buffer full ({self.buffer_size} frames), running inference"
                )
                await self._run_inference()
                # Keep last N frames for overlap
                self.frame_buffer = self.frame_buffer[-self.overlap_frames :]
                self._send_log(f"Kept {self.overlap_frames} frames for overlap")

        elif self.trigger_mode == TriggerMode.MANUAL:
            self._send_log(
                f"Buffered frame {len(self.frame_buffer)} (waiting for manual trigger)"
            )

        # DISABLED mode: just buffer, never process

    async def trigger_inference(self) -> None:
        """Manually trigger inference (called by API endpoint).

        Only works in MANUAL mode.
        """
        if self.trigger_mode != TriggerMode.MANUAL:
            self._send_log(f"Cannot trigger: mode is {self.trigger_mode.value}")
            return

        if not self.frame_buffer:
            self._send_log("Cannot trigger: buffer empty")
            return

        self._send_log(f"Manual trigger: processing {len(self.frame_buffer)} frames")
        await self._run_inference()
        self.frame_buffer.clear()  # No overlap for manual mode

    async def execute(self) -> None:
        """Minimal execute - VideoJob processes in add_frame() instead of loop.

        This method is called when job is submitted to queue, but VideoJob
        doesn't use a traditional execute loop. Inference happens via:
        - PERIODIC: add_frame() auto-triggers
        - MANUAL: trigger_inference() API call
        - DISABLED: never triggers
        """
        self.status = JobStatus.RUNNING
        self.started_at = time.time()
        logger.info(f"[{self.job_id}] VideoJob ready (mode={self.trigger_mode.value})")

        # Stay alive until stopped
        while not self.stop_event.is_set():
            await asyncio.sleep(1.0)

        self.status = JobStatus.COMPLETED
        logger.info(f"[{self.job_id}] VideoJob completed")

    async def _run_inference(self) -> None:
        """Run inference on buffered frames in ThreadPoolExecutor."""
        if not self.frame_buffer:
            return

        frames_to_process = self.frame_buffer.copy()
        frame_count = len(frames_to_process)

        # Log the batch details
        self._send_log(
            f"Starting inference: {frame_count} frames | buffer={self.buffer_size} overlap={self.overlap_frames}"
        )

        # Track timing
        start_time = time.time()

        logger.info(f"[{self.job_id}] Running inference on {frame_count} frames")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._sync_inference, frames_to_process, self.prompt
        )

        # Calculate timing
        inference_time = time.time() - start_time

        # Update state
        self.result = result
        self.processing_time = inference_time

        frames_per_second = frame_count / inference_time if inference_time > 0 else 0.0
        summary = (
            f"Inference complete | frames={frame_count} "
            f"time={inference_time:.3f}s fps={frames_per_second:.2f}"
        )

        logger.info(f"[{self.job_id}] {summary}")
        self._send_log(summary)

        # Broadcast result immediately via callback
        if self.result_callback:
            result_data = {
                "type": "result",
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": "completed",
                "result": result,
                "frames_processed": frame_count,
                "inference_time": inference_time,
                "buffer_size": self.buffer_size,
                "overlap_frames": self.overlap_frames,
                "client_fps": self.client_fps,
                "sample_fps": self.client_fps,
                "timestamp": time.time(),
            }
            self.result_callback(result_data)

    def _sync_inference(self, frames: list[Image.Image], prompt: str) -> str:
        """Blocking GPU inference (runs in ThreadPoolExecutor).

        Processes frames as a video sequence with temporal understanding.
        Qwen2.5-VL uses sample_fps to understand frame timing.
        """
        if not frames:
            return "No frames to process"

        frame_count = len(frames)
        logger.info(
            f"WORKER: VideoJob inference for {self.job_id} ({frame_count} frames)"
        )

        with torch.no_grad():
            # Format as video with temporal information
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,  # List[PIL.Image]
                            "sample_fps": self.client_fps,  # Frame rate
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Extract vision inputs - handles video properly (ignore audio output)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                **(video_kwargs or {}),
            ).to(self.model.device)

            # Generate
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            # Decode only new tokens
            generated_ids = outputs[0][len(inputs.input_ids[0]) :]
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
        """Stop job (called when WebSocket disconnects)."""
        logger.info(f"[{self.job_id}] Stopping VideoJob")
        self.stop_event.set()

    def get_status_dict(self) -> dict:
        """Return custom status information."""
        return {
            "frames_buffered": len(self.frame_buffer),
            "trigger_mode": str(self.trigger_mode.value),
            "buffer_size": self.buffer_size,
            "overlap_frames": self.overlap_frames,
            "client_fps": self.client_fps,
            "default_fps": self.default_fps,
        }

    def to_response_dict(self) -> dict:
        """Serialize VideoJob data for WebSocket response."""
        response = {
            "result": self.result,
            "frames_processed": len(self.frame_buffer),
        }

        # Add metrics if available (after inference has run)
        if self.processing_time is not None:
            response["metrics"] = {
                "inference_time": self.processing_time,
                "total_latency": self.processing_time,
            }
            if self.started_at:
                response["metrics"]["received_at"] = self.started_at

        return response

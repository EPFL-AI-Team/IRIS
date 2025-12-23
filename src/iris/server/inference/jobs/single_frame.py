"""Single frame inference job.

This module contains the SingleFrameJob class for processing individual
frames through VLM inference. SingleFrameJob is a server-side construct that:

- Takes a single image frame at construction time
- Runs inference in a ThreadPoolExecutor thread to avoid blocking
- Uses callbacks (log_callback) to communicate with the WebSocket handler
- Returns the inference result along with base64-encoded frame for response
"""

import asyncio
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

import torch
from PIL import Image

from iris.server.inference.jobs.base import Job, JobStatus

logger = logging.getLogger(__name__)


class SingleFrameJob(Job):
    """
    Process one frame with a VLM.

    This job takes a single image frame, runs it through the VLM model,
    and returns the inference result along with metrics.
    """

    def __init__(
        self,
        job_id: str,
        frame: Image.Image,
        model: Any,
        processor: Any,
        prompt: str,
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

        # Logging callback (set by JobManager or WebSocket handler)
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

    def _sync_inference(self) -> str:
        """
        Does the GPU work (blocking).
        Runs in a separate thread so that its blocking nature won't interfere with other computations.
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

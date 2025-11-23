import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

import torch
from PIL import Image

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
        self.total_latency: float = 0.0 # Will store (completed - received)

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
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            
            # Decode only the new tokens
            result = self.processor.decode(generated_ids, skip_special_tokens=True)

        logger.info(f"WORKER: Finished inference for {self.job_id}")
        return result

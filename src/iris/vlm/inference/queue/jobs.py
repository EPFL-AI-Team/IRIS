import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

import torch
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job lifecycle states"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResult(BaseModel):
    """Result from job execution"""

    job_id: str
    status: JobStatus
    result: Any = None
    error: str | None = None


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

    @abstractmethod
    async def execute(self) -> Any:
        """
        DO THE WORK.

        Each job type implements this differently:
        - SingleFrameJob: runs model on one frame
        - BatchFrameJob: runs model on multiple frames
        - ActivationJob: quick check to decide next action

        MUST be async because it might await executor
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
        prompt: str,
        executor: ThreadPoolExecutor,
    ):
        super().__init__(job_id)

        # Store everything we need for inference
        self.frame = frame
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self.executor = executor  # For running blocking code

    async def execute(self) -> str:
        """
        Coordinate the inference
        """

        self.status = JobStatus.RUNNING
        self.started_at = time.time()

        loop = asyncio.get_event_loop()

        # Run blocking work in its own thread
        result = await loop.run_in_executor(
            self.executor,  # Which thread to pool
            self._sync_inference,  # Non-async function to run
        )

        self.status = JobStatus.COMPLETED
        self.completed_at = time.time()

        return result

    def _sync_inference(self) -> str:
        """
        Does the GPU work (blocking)
        Runs in a separate thread so that its blocking nature won't interfere with other computations
        """
        logger.info(f"WORKER: Starting inference for {self.job_id}")

        with torch.no_grad():
            # Prepare input
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

            # Decode embeddings back to text
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        logger.info(f"WORKER: Finished inference for {self.job_id}")
        return result

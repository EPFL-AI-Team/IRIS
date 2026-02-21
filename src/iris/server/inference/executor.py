"""Inference executor for processing ML jobs asynchronously.

The InferenceExecutor manages a queue of inference jobs and processes them
in separate threads to avoid blocking the main application loop.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from iris.server.inference.jobs.base import Job, JobStatus

logger = logging.getLogger(__name__)


class InferenceExecutor:
    """
    Manages a queue of inference jobs, processing them asynchronously
    in a separate thread to avoid blocking the main application loop.

    This executor:
    - Accepts job submissions via submit()
    - Processes jobs in background worker tasks
    - Routes completed jobs to a results queue
    - Supports multi-GPU setups via worker assignment
    """

    def __init__(
        self,
        max_queue_size: int = 30,
        num_workers: int = 1,
        model_id: str | None = None,
        hardware: str | None = None,
        model_dtype: str | None = None,
    ):
        """
        Initializes the core components of the executor.

        Args:
            max_queue_size: Max jobs to hold before dropping new ones.
            num_workers: Number of parallel workers (one model replica per worker).
            model_id: HuggingFace model ID for per-worker loading
            hardware: Hardware profile (e.g., "v100", "mac") for per-worker loading
            model_dtype: Model dtype override (float16, float32, bfloat16, auto)
        """
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue(maxsize=max_queue_size)
        self.results: asyncio.Queue[Job] = asyncio.Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers
        )  # Thread pool for running blocking GPU code
        self.workers: list[asyncio.Task] = []  # list to hold our running worker tasks
        self._running = False  # Used for graceful shutdown
        self.num_workers = num_workers

        # Model configuration for per-worker loading
        self.model_id = model_id
        self.hardware = hardware
        self.model_dtype = model_dtype

    async def start(self) -> None:
        """Starts the worker tasks that will consume jobs from the queue."""
        if self._running:
            logger.warning("Executor is already running.")
            return

        self._running = True
        for i in range(self.num_workers):
            # Create a background task for each worker (pass worker_id for GPU assignment)
            worker_task = asyncio.create_task(self._worker(worker_id=i))
            self.workers.append(worker_task)
        logger.info(f"Started {self.num_workers} inference workers.")

    async def stop(self) -> None:
        """Gracefully shuts down the executor and all worker tasks."""
        if not self._running:
            return

        logger.info("Stopping inference executor...")
        self._running = False

        # Send a "None" signal for each worker to unblock and exit
        for _ in self.workers:
            try:
                self.queue.put_nowait(None)  # Non-blocking
            except asyncio.QueueFull:
                pass  # Queue full, worker will exit when it drains

        # Wait for all worker tasks to complete
        await asyncio.gather(*self.workers, return_exceptions=True)

        # Shut down the thread pool
        self.executor.shutdown(wait=False)  # Don't wait for GPU threads
        logger.info("Inference executor stopped.")

    async def clear_queue(self) -> int:
        """Clear all pending jobs from the queue without stopping workers.

        Returns:
            Number of jobs removed from queue
        """
        cleared_count = 0

        # Drain the queue by consuming items without processing
        while not self.queue.empty():
            try:
                job = self.queue.get_nowait()
                # Skip None (shutdown signals)
                if job is not None:
                    cleared_count += 1
                    # Mark job as cancelled/failed
                    job.status = JobStatus.FAILED
                    job.error = "Cleared by system reset"
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Also clear results queue to prevent memory leaks
        while not self.results.empty():
            try:
                self.results.get_nowait()
                self.results.task_done()
            except asyncio.QueueEmpty:
                break

        logger.info(f"Cleared {cleared_count} pending jobs from queue")
        return cleared_count

    async def submit(self, job: Job) -> bool:
        """
        Submits a job to the executor for processing.

        Returns:
            True if the job was submitted, False if the queue was full and it was dropped.
        """
        try:
            # put_nowait is non-blocking and raises an error if the queue is full
            self.queue.put_nowait(job)
            logger.debug(f"Submitted {job}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue is full. Dropped {job}")
            return False

    async def get_result(self, timeout: float | None = None) -> Job | None:
        """Waits for and returns the next completed job object."""
        try:
            return await asyncio.wait_for(self.results.get(), timeout)
        except TimeoutError:
            return None

    async def _worker(self, worker_id: int = 0) -> None:
        """The core consumer loop that processes jobs by executing them, and queuing results.

        Args:
            worker_id: Worker index (0-indexed). Used for GPU assignment in multi-GPU setups.
        """
        worker_name = f"worker-{worker_id}"

        # Assign GPU based on worker_id
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                assigned_gpu = worker_id % device_count
                device = f"cuda:{assigned_gpu}"
                logger.info(
                    f"{worker_name} assigned to {device} "
                    f"(GPU {assigned_gpu + 1}/{device_count})"
                )
            else:
                device = "cpu"
                logger.info(f"{worker_name} using CPU (no GPUs available)")
        except ImportError:
            device = "cpu"
            logger.info(f"{worker_name} using CPU (torch not available)")

        # Load worker-specific model replica
        worker_model = None
        worker_processor = None
        if self.model_id:
            try:
                from iris.server.model_loader import load_model_and_processor

                logger.info(f"{worker_name} loading model replica: {self.model_id}")
                worker_model, worker_processor = load_model_and_processor(
                    model_id=self.model_id,
                    hardware=self.hardware,
                    model_dtype=self.model_dtype,
                )

                # Move model to assigned GPU if needed
                if device != "cpu" and hasattr(worker_model, "to"):
                    logger.info(f"{worker_name} moving model to {device}")
                    worker_model = worker_model.to(device) # type: ignore

                logger.info(f"{worker_name} model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"{worker_name} failed to load model: {e}", exc_info=True)
                # Continue without model - jobs will fail with validation error

        logger.info(f"{worker_name} has started.")
        try:
            while self._running:
                # Wait for a job to appear in the queue
                job: Job | None = await self.queue.get()
                # Check for the shutdown signal
                if job is None:
                    self.queue.task_done()
                    break

                try:
                    # Inject worker's model into job before execution
                    if hasattr(job, "model") and hasattr(job, "processor"):
                        job.model = worker_model
                        job.processor = worker_processor

                    # Log queue depth before processing
                    queue_depth = self.queue.qsize()
                    logger.info(f"{worker_name} processing {job}, queue_depth={queue_depth}")

                    # Run the job. The job updates its own internal state.
                    await job.execute()

                    # Log queue depth after completion to show remaining backlog
                    queue_depth = self.queue.qsize()
                    remaining_results = self.results.qsize()
                    remaining_jobs = queue_depth
                    logger.info(
                        f"{worker_name} completed {job}, queue_depth={queue_depth}, remaining_jobs={remaining_jobs}, pending_results={remaining_results}"
                    )

                except Exception as e:
                    logger.error(f"{worker_name} failed on {job}: {e}", exc_info=True)
                    # If it fails, update the job's state
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                finally:
                    # Pass the entire job object to the results queue
                    await self.results.put(job)
                    self.queue.task_done()
        except asyncio.CancelledError:
            logger.debug(f"{worker_name} cancelled during shutdown")

        logger.info(f"{worker_name} has stopped.")

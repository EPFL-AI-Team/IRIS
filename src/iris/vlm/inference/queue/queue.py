import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from iris.vlm.inference.queue.jobs import Job, JobStatus

logger = logging.getLogger(__name__)


class InferenceQueue:
    """
    Manages a queue of inference jobs, processing them asynchronously
    in a separate thread to avoid blocking the main application loop.
    """

    def __init__(self, max_queue_size: int = 30, num_workers: int = 1):
        """
        Initializes the core components of the queue.

        Args:
            max_queue_size: Max jobs to hold before dropping new ones.
            num_workers: Number of parallel threads for inference. For one GPU, use 1.
        """
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue(maxsize=max_queue_size)
        self.results: asyncio.Queue[Job] = asyncio.Queue()
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers
        )  # Thread pool for running blocking GPU code
        self.workers: list[asyncio.Task] = []  # list to hold our running worker tasks
        self._running = False  # Used for graceful shutdown
        self.num_workers = num_workers

    async def start(self) -> None:
        """Starts the worker tasks that will consume jobs from the queue."""
        if self._running:
            logger.warning("Queue is already running.")
            return

        self._running = True
        for i in range(self.num_workers):
            # Create a background task for each worker (pass worker_id for GPU assignment)
            worker_task = asyncio.create_task(self._worker(worker_id=i))
            self.workers.append(worker_task)
        logger.info(f"Started {self.num_workers} inference workers.")

    async def stop(self) -> None:
        """Gracefully shuts down the queue and all worker tasks."""
        if not self._running:
            return

        logger.info("Stopping inference queue...")
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
        logger.info("Inference queue stopped.")

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
        Submits a job to the queue for processing.

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

        logger.info(f"{worker_name} has started.")
        while self._running:
            # Wait for a job to appear in the queue
            job: Job | None = await self.queue.get()
            # Check for the shutdown signal
            if job is None:
                self.queue.task_done()
                break

            try:
                # Move model to this worker's GPU if needed
                if hasattr(job, "model") and hasattr(job.model, "to") and device != "cpu":
                    current_device = str(job.model.device) if hasattr(job.model, "device") else "unknown"
                    if device not in current_device:
                        logger.debug(
                            f"{worker_name}: Moving model from {current_device} to {device}"
                        )
                        job.model.to(device)

                # Log queue depth before processing
                queue_depth = self.queue.qsize()
                logger.info(f"{worker_name} processing {job}, queue_depth={queue_depth}")

                # Run the job. The job updates its own internal state.
                await job.execute()

                # Log queue depth after completion
                queue_depth = self.queue.qsize()
                logger.info(f"{worker_name} completed {job}, queue_depth={queue_depth}")

            except Exception as e:
                logger.error(f"{worker_name} failed on {job}: {e}", exc_info=True)
                # If it fails, update the job's state
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                # Pass the entire job object to the results queue
                await self.results.put(job)
                self.queue.task_done()

        logger.info(f"{worker_name} has stopped.")

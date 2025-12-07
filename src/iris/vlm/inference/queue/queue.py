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
            # Create a background task for each worker
            worker_task = asyncio.create_task(self._worker(f"worker-{i}"))
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

    async def _worker(self, name: str) -> None:
        """The core consumer loop that processes jobs by executing them, and queuing results."""
        logger.info(f"{name} has started.")
        while self._running:
            # Wait for a job to appear in the queue
            job: Job | None = await self.queue.get()
            # Check for the shutdown signal
            if job is None:
                self.queue.task_done()
                break

            try:
                logger.info(f"{name} is processing {job}")
                # Run the job. The job updates its own internal state.
                await job.execute()
            except Exception as e:
                logger.error(f"{name} failed on {job}: {e}", exc_info=True)
                # If it fails, update the job's state
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                # Pass the entire job object to the results queue
                await self.results.put(job)
                self.queue.task_done()

        logger.info(f"{name} has stopped.")

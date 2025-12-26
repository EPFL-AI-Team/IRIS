"""Server lifecycle management including shutdown and signal handling."""

import asyncio
import logging
import os
import signal
import threading
import time
from typing import Any

from iris.server.dependencies import get_server_state

logger = logging.getLogger(__name__)


class LifecycleHandler:
    """Manages server lifecycle including graceful shutdown.

    Responsibilities:
    - Handle shutdown signals (SIGINT/SIGTERM)
    - Track shutdown state
    - Monitor queue drainage during shutdown
    - Start background result drainer task
    """

    def __init__(self):
        """Initialize lifecycle handler."""
        self.shutdown_event = asyncio.Event()
        self.force_shutdown_event = asyncio.Event()
        self.shutdown_count = 0
        self.drainer_task: asyncio.Task | None = None

    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self.shutdown_event.is_set()

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown.

        On first signal: Mark shutdown, wait for queued jobs to complete.
        On second signal: Force immediate shutdown.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.shutdown_count += 1

        state = get_server_state()
        state.shutting_down = True

        # Capture quick stats to help the operator decide whether to wait or force-exit
        status = self._queue_status_snapshot()

        if self.shutdown_count == 1:
            logger.info(
                "Received %s. Graceful shutdown initiated; active_jobs=%d, queue_depth=%d, pending_results=%d.",
                signal.Signals(signum).name,
                status["active_jobs"],
                status["queue_depth"],
                status["pending_results"],
            )

            self.shutdown_event.set()
            total_pending = status["active_jobs"] + status["queue_depth"]

            if total_pending > 0:
                logger.info(
                    "Waiting for %d queued jobs to complete. Press Ctrl+C again to force shutdown.",
                    total_pending,
                )
                # Restore default handlers so second Ctrl+C triggers immediate uvicorn shutdown
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                # Start background thread to monitor queue and auto-shutdown when drained
                threading.Thread(
                    target=self._wait_for_drain_and_shutdown, daemon=True
                ).start()
            else:
                # Queue already empty, trigger shutdown immediately
                logger.info("Queue already empty - shutting down")
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            return

        # shutdown_count >= 2 shouldn't reach here since we restore default handlers
        # but handle it just in case
        logger.warning(
            "Received %s again. Force shutdown.",
            signal.Signals(signum).name,
        )
        self.force_shutdown_event.set()

    def _wait_for_drain_and_shutdown(self) -> None:
        """Background thread that waits for queue drain then triggers shutdown."""
        while True:
            time.sleep(0.5)
            if self.force_shutdown_event.is_set():
                return  # Already forcing shutdown

            status = self._queue_status_snapshot()
            if status["active_jobs"] == 0 and status["queue_depth"] == 0:
                logger.info("All queued jobs completed - initiating server shutdown")
                # Send SIGINT to self - default handler will trigger uvicorn shutdown
                os.kill(os.getpid(), signal.SIGINT)
                return

    def _queue_status_snapshot(self) -> dict[str, int]:
        """Return lightweight snapshot of queue/active job counts.

        Note: This is a utility function used by both HTTP endpoints and the
        signal handler. Shutdown checks should be done at the endpoint level.
        """
        state = get_server_state()
        queue_depth = state.queue.queue.qsize() if state.queue else 0
        active_jobs = (
            len(state.job_manager.active_jobs)
            if state.job_manager and state.job_manager.active_jobs is not None
            else 0
        )
        pending_results = state.queue.results.qsize() if state.queue else 0
        return {
            "queue_depth": queue_depth,
            "active_jobs": active_jobs,
            "pending_results": pending_results,
        }

    async def start_result_drainer(self) -> None:
        """Start background task to drain global results queue.

        This prevents memory leaks by consuming results that have already
        been handled by per-job callbacks.
        """
        state = get_server_state()

        async def _drain_loop() -> None:
            try:
                while True:
                    _job = await state.queue.results.get()
                    state.queue.results.task_done()
                    # Results are handled by per-job callbacks
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Global result drainer error: {e}")

        self.drainer_task = asyncio.create_task(_drain_loop())
        logger.info("Result drainer started")

    async def stop_result_drainer(self) -> None:
        """Stop the background result drainer task."""
        if self.drainer_task:
            self.drainer_task.cancel()
            try:
                await self.drainer_task
            except asyncio.CancelledError:
                pass
            logger.info("Result drainer stopped")

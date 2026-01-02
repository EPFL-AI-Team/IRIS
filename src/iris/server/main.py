"""IRIS Inference Server - Main Entry Point.

This module provides a clean entry point for the IRIS inference server,
orchestrating all components including routes, lifecycle management, and state.
"""

import asyncio
import logging
import os
import signal
import sys
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.inference.executor import InferenceExecutor
from iris.server.jobs.manager import JobManager
from iris.server.lifecycle import LifecycleHandler
from iris.server.logging_handler import WebSocketLogHandler
from iris.server.routes import jobs, system

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*torchao.*incompatible torch version.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Configure logging - reduce noise from transformers/HF
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# Reduce verbosity from external libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

config = ServerConfig()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage server startup and shutdown lifecycle."""
    # === STARTUP ===
    state = get_server_state()

    # 1. Initialize WebSocket log handler if enabled
    if config.enable_log_streaming:
        state.log_handler = WebSocketLogHandler(
            min_level=config.log_streaming_min_level
        )
        # Attach to root logger to capture all logs
        logging.getLogger().addHandler(state.log_handler)
        logger.info(
            "Log streaming enabled (min level: %s)", config.log_streaming_min_level
        )

    # 2. Initialize lifecycle handler
    state.lifecycle = LifecycleHandler()

    # 3. Register signal handlers
    signal.signal(signal.SIGINT, state.lifecycle.handle_signal)
    signal.signal(signal.SIGTERM, state.lifecycle.handle_signal)
    logger.info("Signal handlers registered for graceful shutdown")

    # 4. Start inference executor
    logger.info("Starting inference executor with per-worker model loading...")
    state.queue = InferenceExecutor(
        max_queue_size=config.max_queue_size,
        num_workers=config.num_workers,
        model_id=config.model_id,
        hardware=config.vlm_hardware,
        model_dtype=config.model_dtype,
    )
    await state.queue.start()

    # 5. Start global result drainer
    await state.lifecycle.start_result_drainer()

    # 6. Initialize metrics collector if enabled
    if config.enable_metrics:
        from iris.server.metrics import MetricsCollector

        state.metrics = MetricsCollector(
            persist=True,
            log_dir="logs/metrics",
            collect_gpu_metrics=True,
        )
        logger.info("Metrics collection enabled")

    # 7. Initialize job manager
    state.job_manager = JobManager(state)
    logger.info("Job manager initialized")

    state.model_loaded = True
    logger.info("Server ready!")

    yield

    # === SHUTDOWN ===
    logger.info("Shutting down...")

    # 1. Stop result drainer
    await state.lifecycle.stop_result_drainer()

    # 2. Close metrics collector
    if state.metrics:
        state.metrics.close()
        logger.info("Metrics collector closed")

    # 3. Stop inference executor
    if state.queue:
        status = state.lifecycle._queue_status_snapshot()
        logger.info(
            "Stopping executor (active_jobs=%d, queue_depth=%d)",
            status["active_jobs"],
            status["queue_depth"],
        )
        await state.queue.stop()

    logger.info("Server stopped.")


# Create FastAPI application
app = FastAPI(title="IRIS Inference Server", lifespan=lifespan)

# Include routers (WebSocket routes are in app.py, the main entry point)
app.include_router(jobs.router, tags=["jobs"])
app.include_router(system.router, tags=["system"])


def main() -> None:
    """Entry point for server."""
    import uvicorn

    try:
        uvicorn.run(app, host=config.host, port=config.port)
    except KeyboardInterrupt:
        logger.info("Server interrupted during startup. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()

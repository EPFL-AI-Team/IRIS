"""IRIS Inference Server - receives frames, runs VLM inference."""

import asyncio
import base64
import logging
import signal
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from PIL import Image

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.jobs.config import JobConfig
from iris.server.jobs.manager import JobManager
from iris.server.logging_handler import WebSocketLogHandler
from iris.vlm.inference.queue.queue import InferenceQueue
from iris.vlm.models import load_model_and_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ServerConfig()

# Initialize log streaming handler if enabled
log_streaming_handler: WebSocketLogHandler | None = None
if config.enable_log_streaming:
    log_streaming_handler = WebSocketLogHandler(
        min_level=config.log_streaming_min_level
    )
    # Add to root logger to capture all logs
    logging.getLogger().addHandler(log_streaming_handler)
    logger.info("Log streaming enabled (min level: %s)", config.log_streaming_min_level)

# Graceful shutdown management
shutdown_event = asyncio.Event()
force_shutdown_event = asyncio.Event()
shutdown_count = 0


def handle_shutdown_signal(signum: int, frame: any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_count
    shutdown_count += 1

    if shutdown_count == 1:
        logger.info(
            "Received signal %s. Initiating graceful shutdown (waiting for in-flight jobs, timeout: %.1fs). "
            "Send signal again to force shutdown.",
            signal.Signals(signum).name,
            config.graceful_shutdown_timeout,
        )
        shutdown_event.set()
    elif shutdown_count == 2:
        logger.warning(
            "Received signal %s again. Force shutdown initiated. Exiting immediately.",
            signal.Signals(signum).name,
        )
        force_shutdown_event.set()

        # Unregister handlers to allow default behavior (immediate exit on next signal)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown."""
    # Startup
    state = get_server_state()

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    logger.info("Signal handlers registered for graceful shutdown")

    logger.info("Loading model...")

    state.model, state.processor = load_model_and_processor(
        vlm_config_name=config.vlm_config,
        hardware=config.vlm_hardware,
    )

    logger.info("Starting inference queue...")
    state.queue = InferenceQueue(
        max_queue_size=config.max_queue_size, num_workers=config.num_workers
    )
    await state.queue.start()

    # Initialize metrics collector if enabled
    if config.enable_metrics:
        from iris.server.metrics import MetricsCollector

        state.metrics = MetricsCollector(
            persist=True,
            log_dir="logs/metrics",
            collect_gpu_metrics=True,
        )
        logger.info("Metrics collection enabled")

    # Initialize job manager
    state.job_manager = JobManager(state)
    logger.info("Job manager initialized")

    state.model_loaded = True
    logger.info("Server ready!")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Close metrics collector
    if state.metrics:
        state.metrics.close()
        logger.info("Metrics collector closed")

    if state.queue:
        if force_shutdown_event.is_set():
            logger.warning("Force shutdown - terminating all jobs immediately")
            await state.queue.stop()
            logger.warning("Exiting now")
            import sys

            sys.exit(1)  # Force exit if still running
        else:
            logger.info(
                "Graceful shutdown - waiting for in-flight jobs (timeout: %.1fs)",
                config.graceful_shutdown_timeout,
            )
            try:
                await asyncio.wait_for(
                    state.queue.stop(), timeout=config.graceful_shutdown_timeout
                )
                logger.info("All in-flight jobs completed successfully")
            except TimeoutError:
                logger.warning(
                    "Graceful shutdown timeout (%.1fs) exceeded. Some jobs may be incomplete.",
                    config.graceful_shutdown_timeout,
                )
    logger.info("Server stopped.")


app = FastAPI(title="IRIS Inference Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    """Health check endpoint."""
    state = get_server_state()
    return {
        "status": "healthy" if state.model_loaded else "loading",
        "model_loaded": state.model_loaded,
    }


@app.post("/jobs/start")
async def start_job(config: JobConfig) -> dict[str, any]:
    """Start a new job with specified configuration.

    Args:
        config: Job configuration (validated Pydantic model)

    Returns:
        Dictionary with job_id, status, job_type, and config
    """
    state = get_server_state()

    try:
        job_id = await state.job_manager.start_job(config)
        return {
            "job_id": job_id,
            "status": "started",
            "job_type": config.job_type.value,
            "config": config.model_dump(),
        }
    except Exception as e:
        logger.error(f"Failed to start job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> dict[str, any]:
    """Stop a running job.

    Args:
        job_id: Job identifier

    Returns:
        Dictionary with job_id, status, and message
    """
    state = get_server_state()

    try:
        success = await state.job_manager.stop_job(job_id)
        if success:
            return {
                "job_id": job_id,
                "status": "stopped",
                "message": "Job stopped successfully",
            }
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str) -> dict[str, any]:
    """Get status of a specific job.

    Args:
        job_id: Job identifier

    Returns:
        Dictionary with job status details
    """
    state = get_server_state()

    status = await state.job_manager.get_job_status(job_id)
    if status:
        return status
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/jobs/active")
async def list_active_jobs() -> dict[str, any]:
    """List all active jobs.

    Returns:
        Dictionary with list of active jobs
    """
    state = get_server_state()

    jobs = await state.job_manager.list_active_jobs()
    return {"active_jobs": jobs}


@app.websocket("/ws/stream")
async def inference_endpoint(websocket: WebSocket) -> None:
    """Receive frames and return inference results."""
    await websocket.accept()
    state = get_server_state()
    logger.info("Client connected")

    # Producer loop which receives frames and routes them to active jobs
    async def receive_loop() -> None:
        try:
            while True:
                data = await websocket.receive_json()
                arrival_time = time.time()

                frame_b64 = data["frame"]
                frame_id = data["frame_id"]

                image_data = base64.b64decode(frame_b64)
                image = Image.open(BytesIO(image_data))

                # Route frame to active jobs (decoupled from job creation)
                await state.job_manager.route_frame(
                    frame=image, frame_id=frame_id, timestamp=arrival_time
                )

        except WebSocketDisconnect:
            logger.info("Client disconnected (Receive Loop)")
        except Exception as e:
            logger.error("Receive loop error: %s", e, exc_info=True)

    # Consumer loop which watches queue result and sends them to the client
    async def send_loop() -> None:
        try:
            while True:
                # Wait specifically for the NEXT available result
                # accessing the internal results queue directly
                result_job = await state.queue.results.get()

                # Base response structure
                response = {
                    "job_id": result_job.job_id,
                    "job_type": result_job.job_type,
                    "status": result_job.status.value,
                }

                # Add job-specific data via polymorphism
                response.update(result_job.to_response_dict())

                await websocket.send_json(response)
                state.queue.results.task_done()

                # Record job metrics (if applicable)
                if state.metrics and hasattr(result_job, "processing_time"):
                    total_latency = getattr(
                        result_job, "total_latency", result_job.processing_time
                    )
                    state.metrics.record_job(
                        job_id=result_job.job_id,
                        inference_time=result_job.processing_time,
                        total_latency=total_latency,
                        status=result_job.status.value,
                        queue_depth=state.queue.queue.qsize(),
                    )

        except WebSocketDisconnect:
            logger.info("Client disconnected (Send Loop)")
        except Exception as e:
            logger.error("Send loop error: %s", e, exc_info=True)

    # Run both functions concurrently
    # This runs until one of them finishes (usually the receive loop on disconnect)
    await asyncio.gather(receive_loop(), send_loop())


@app.websocket("/ws/logs")
async def log_streaming_endpoint(websocket: WebSocket) -> None:
    """Stream server logs to connected clients."""
    if not config.enable_log_streaming or log_streaming_handler is None:
        await websocket.close(code=1008, reason="Log streaming is disabled")
        return

    await websocket.accept()
    log_streaming_handler.add_connection(websocket)
    logger.info(
        "Log streaming client connected (total: %d)",
        log_streaming_handler.get_connection_count(),
    )

    try:
        # Keep connection alive and wait for client disconnect
        while True:
            # Receive ping/pong to detect disconnection
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text('{"type": "ping"}')
    except WebSocketDisconnect:
        logger.info("Log streaming client disconnected")
    except Exception as e:
        logger.error("Log streaming error: %s", e, exc_info=True)
    finally:
        log_streaming_handler.remove_connection(websocket)
        logger.info(
            "Log streaming client removed (remaining: %d)",
            log_streaming_handler.get_connection_count(),
        )


@app.get("/metrics")
async def metrics_endpoint() -> dict[str, any]:
    """Get current metrics and statistics."""
    state = get_server_state()

    if not config.enable_metrics or state.metrics is None:
        return {
            "error": "Metrics collection is disabled",
            "enable_metrics": False,
        }

    return {
        "enable_metrics": True,
        "stats": state.metrics.get_stats(),
        "recent_jobs": state.metrics.get_recent_jobs(limit=20),
    }


def main() -> None:
    """Entry point for server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

"""IRIS Inference Server - receives frames, runs VLM inference."""

import asyncio
import base64
import logging
import os
import signal
import time
import uuid
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*torchao.*incompatible torch version.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Suppress tokenizers warning
from io import BytesIO

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from PIL import Image
from typing import Any

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.jobs.config import JobConfig, VideoJobConfig
from iris.server.jobs.types import TriggerMode
from iris.server.jobs.manager import JobManager
from iris.server.logging_handler import WebSocketLogHandler
from iris.vlm.inference.queue.queue import InferenceQueue
from iris.vlm.models import load_model_and_processor

# Configure logging - reduce noise from transformers/HF
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger(__name__)

# Reduce verbosity from external libraries
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

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


def handle_shutdown_signal(signum: int, frame: Any) -> None:
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
        model_id=config.model_id,
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
async def start_job(config: JobConfig) -> dict[str, Any]:
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
async def stop_job(job_id: str) -> dict[str, Any]:
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
async def get_job_status(job_id: str) -> dict[str, Any]:
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


@app.post("/jobs/{job_id}/trigger")
async def trigger_job(job_id: str) -> dict[str, str]:
    """Manually trigger inference for a VideoJob.

    Args:
        job_id: Job identifier

    Returns:
        Dictionary with status message

    Raises:
        HTTPException: 404 if job not found, 400 if job doesn't support manual triggering
    """
    state = get_server_state()

    # Get job from active jobs
    async with state.job_manager.lock:
        job = state.job_manager.active_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not hasattr(job, "trigger_inference"):
        raise HTTPException(
            status_code=400, detail="Job does not support manual triggering"
        )

    await job.trigger_inference()
    return {"status": "ok", "message": "Inference triggered"}


@app.get("/jobs/active")
async def list_active_jobs() -> dict[str, Any]:
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

    # Log queue for this connection
    log_queue = asyncio.Queue()

    def log_callback(msg: dict) -> None:
        """Callback to receive log messages from jobs."""
        asyncio.create_task(log_queue.put(msg))

    # Register log callback with JobManager
    state.job_manager.register_log_callback(log_callback)

    # AUTO-CREATE VIDEO JOB WITH UNIQUE ID PER CONNECTION
    connection_job_id = f"video_job_{uuid.uuid4().hex[:8]}"
    job_config = VideoJobConfig(
        job_id=connection_job_id,
        prompt="Describe what you see in the video.",
        trigger_mode=TriggerMode.PERIODIC,
        buffer_size=config.jobs.get("video", {}).get("buffer_size", 8),
        overlap_frames=config.jobs.get("video", {}).get("overlap_frames", 4),
    )

    try:
        job_id = await state.job_manager.start_job(job_config)
        logger.info(f"Auto-created VideoJob for connection: {job_id}")

        # Track WebSocket connection state
        connection_active = True
        pending_tasks = []

        # Register result callback for immediate broadcasting
        video_job = state.job_manager.get_job(job_id)
        if video_job and hasattr(video_job, 'result_callback'):
            async def result_handler(result_data: dict):
                """Handle result from VideoJob and broadcast via WebSocket."""
                if not connection_active:
                    return  # Skip if connection is closed
                
                try:
                    # Send to WebSocket client
                    await websocket.send_json(result_data)

                    # Record metrics
                    if state.metrics:
                        state.metrics.record_job(
                            job_id=result_data["job_id"],
                            inference_time=result_data.get("inference_time", 0.0),
                            total_latency=result_data.get("inference_time", 0.0),
                            status="completed",
                            queue_depth=state.queue.queue.qsize(),
                        )

                        # Save detailed result to session results file
                        state.metrics.record_inference_result(result_data)
                except (WebSocketDisconnect, RuntimeError) as e:
                    # WebSocket already closed, ignore silently
                    logger.debug(f"Skipping result send - WebSocket closed: {e}")
                except Exception as e:
                    logger.error(f"Error in result handler: {e}", exc_info=True)

            # Wrap async callback for sync context
            def sync_result_callback(result_data: dict):
                task = asyncio.create_task(result_handler(result_data))
                pending_tasks.append(task)

            video_job.result_callback = sync_result_callback

    except Exception as e:
        logger.error(f"Failed to auto-create VideoJob: {e}")
        await websocket.close(code=1011, reason="Failed to create video job")
        return

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
                # Check for log messages (non-blocking)
                try:
                    log_msg = log_queue.get_nowait()
                    await websocket.send_json(log_msg)
                except asyncio.QueueEmpty:
                    pass

                # Check for job results (with short timeout to allow log checking)
                try:
                    result_job = await asyncio.wait_for(
                        state.queue.results.get(), timeout=0.1
                    )

                    # Base response structure
                    response = {
                        "type": "result",
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
                except asyncio.TimeoutError:
                    # No results ready, continue loop to check logs
                    pass

        except WebSocketDisconnect:
            logger.info("Client disconnected (Send Loop)")
        except Exception as e:
            logger.error("Send loop error: %s", e, exc_info=True)

    # Run both functions concurrently
    # This runs until one of them finishes (usually the receive loop on disconnect)
    try:
        await asyncio.gather(receive_loop(), send_loop())
    finally:
        # Mark connection as inactive to stop result handler tasks
        connection_active = False
        
        # Cancel any pending result handler tasks
        for task in pending_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to finish cancellation
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        
        # CLEANUP: Stop and remove job when WebSocket disconnects
        try:
            await state.job_manager.stop_job(connection_job_id)
            logger.info(f"Cleaned up VideoJob on disconnect: {connection_job_id}")
        except Exception as e:
            logger.error(f"Failed to clean up job {connection_job_id}: {e}")


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
async def metrics_endpoint() -> dict[str, Any]:
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

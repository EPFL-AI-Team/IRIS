"""IRIS Inference Server - receives frames, runs VLM inference."""

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import time
import uuid
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from PIL import Image

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.jobs.config import JobConfig
from iris.server.jobs.manager import JobManager
from iris.server.logging_handler import WebSocketLogHandler
from iris.vlm.inference.queue.queue import InferenceQueue
from iris.vlm.models import load_model_and_processor

# Suppress known warnings
warnings.filterwarnings("ignore", message=".*torchao.*incompatible torch version.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Suppress tokenizers warning

# Configure logging - reduce noise from transformers/HF
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
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


def _queue_status_snapshot() -> dict[str, int]:
    """Return a lightweight snapshot of queue/active job counts."""
    state = get_server_state()

    if state.shutting_down:
        logger.warning("Rejecting process-video request during shutdown")
        raise HTTPException(status_code=503, detail="Server shutting down")
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


def handle_shutdown_signal(signum: int, frame: Any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_count
    shutdown_count += 1

    state = get_server_state()
    state.shutting_down = True

    # Capture quick stats to help the operator decide whether to wait or force-exit
    status = _queue_status_snapshot()

    if shutdown_count == 1:
        logger.info(
            "Received %s. Graceful shutdown requested; active_jobs=%d, queue_depth=%d, pending_results=%d. "
            "Waiting for jobs to finish. Send signal again to force shutdown.",
            signal.Signals(signum).name,
            status["active_jobs"],
            status["queue_depth"],
            status["pending_results"],
        )
        shutdown_event.set()
    elif shutdown_count == 2:
        logger.warning(
            "Received signal %s again. Force shutdown initiated. Exiting immediately.",
            signal.Signals(signum).name,
        )
        force_shutdown_event.set()

        # Cancel all running tasks to trigger shutdown
        try:
            loop = asyncio.get_running_loop()
            for task in asyncio.all_tasks(loop):
                task.cancel()
        except RuntimeError:
            pass  # No loop running

        # Unregister handlers to allow default behavior (immediate exit on next signal)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


async def global_result_drainer(state: Any) -> None:
    """Drain global results queue to prevent memory leaks."""
    try:
        while True:
            job = await state.queue.results.get()
            state.queue.results.task_done()
            # Results are handled by per-job callbacks
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Global result drainer error: {e}")


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

    # Start global result drainer
    result_drainer_task = asyncio.create_task(global_result_drainer(state))

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

    # Stop global result drainer
    result_drainer_task.cancel()
    try:
        await result_drainer_task
    except asyncio.CancelledError:
        pass

    # Close metrics collector
    if state.metrics:
        state.metrics.close()
        logger.info("Metrics collector closed")

    if state.queue:
        if force_shutdown_event.is_set():
            logger.warning("Force shutdown - terminating all jobs immediately")
            await state.queue.stop()
            import sys

            sys.exit(1)  # Force exit if still running
        else:
            status_before = _queue_status_snapshot()
            logger.info(
                "Graceful shutdown - waiting for in-flight jobs (no timeout). "
                "active_jobs=%d, queue_depth=%d, pending_results=%d",
                status_before["active_jobs"],
                status_before["queue_depth"],
                status_before["pending_results"],
            )

            await state.queue.stop()

            status_after = _queue_status_snapshot()
            logger.info(
                "Queue stopped. Remaining: active_jobs=%d, queue_depth=%d, pending_results=%d",
                status_after["active_jobs"],
                status_after["queue_depth"],
                status_after["pending_results"],
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

    if state.shutting_down:
        logger.warning(
            "Rejecting start_job request during shutdown: job_type=%s", config.job_type
        )
        raise HTTPException(status_code=503, detail="Server shutting down")

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


@app.get("/list-videos")
async def list_videos() -> dict:
    """List available pre-loaded videos for processing.

    Returns:
        Dictionary with list of video filenames and metadata
    """
    from pathlib import Path

    from iris.server.video_processor import VideoFrameExtractor

    # Video directory in static files
    video_dir = Path(__file__).parent.parent / "client" / "web" / "static" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]:
        for video_path in video_dir.glob(ext):
            try:
                extractor = VideoFrameExtractor(video_path)
                info = extractor.get_video_info()
                videos.append(info)
            except Exception as e:
                logger.warning(f"Failed to get info for {video_path.name}: {e}")
                videos.append({"filename": video_path.name, "error": str(e)})

    return {"videos": sorted(videos, key=lambda x: x.get("filename", ""))}


@app.post("/process-video")
async def process_video_endpoint(request: dict) -> dict:
    """Process a pre-loaded video file frame by frame.

    Args:
        request: Dictionary with video_name

    Returns:
        Dictionary with processing status and job info
    """
    from pathlib import Path

    from iris.server.video_processor import VideoFrameExtractor
    from iris.server.inference.jobs import VideoJob

    video_name = request.get("video_name")
    if not video_name:
        raise HTTPException(status_code=400, detail="video_name is required")

    # Find video file
    video_dir = Path(__file__).parent.parent / "client" / "web" / "static" / "videos"
    video_path = video_dir / video_name

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_name}")

    state = get_server_state()
    video_cfg = config.jobs.get("video", {})

    # Extract frames at target FPS
    target_fps = video_cfg.get("default_fps", 5.0)
    extractor = VideoFrameExtractor(video_path, target_fps=target_fps)

    try:
        frames = extractor.extract_frames()
    except Exception as e:
        logger.error(f"Failed to extract frames from {video_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to extract frames: {e!s}"
        ) from e

    if not frames:
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    # Create batch jobs (same as live stream)
    job_id_base = f"video_file_{uuid.uuid4().hex[:8]}"
    buffer_size = video_cfg.get("buffer_size", 8)
    overlap_frames = video_cfg.get("overlap_frames", 4)
    prompt = video_cfg.get("prompt", "Describe what you see in the video.")
    max_new_tokens = video_cfg.get("max_new_tokens", 128)

    batches_created = 0
    stride = buffer_size - overlap_frames

    for i in range(0, len(frames), stride):
        batch_frames = frames[i : i + buffer_size]

        # Skip if batch is too small (less than half buffer size)
        if len(batch_frames) < buffer_size // 2:
            logger.info(f"Skipping small batch at end: {len(batch_frames)} frames")
            break

        batch_job_id = f"{job_id_base}_batch_{batches_created}"

        if state.shutting_down:
            logger.info(
                "Shutdown in progress; skipping remaining video batches after %d created",
                batches_created,
            )
            break

        # Create VideoJob
        batch_job = VideoJob(
            job_id=batch_job_id,
            model=state.model,
            processor=state.processor,
            executor=state.queue.executor,
            frames=batch_frames,
            prompt=prompt,
            buffer_size=buffer_size,
            overlap_frames=overlap_frames,
            default_fps=target_fps,
            max_new_tokens=max_new_tokens,
            client_fps=target_fps,
        )

        # Submit to queue
        submitted = await state.queue.submit(batch_job)
        if submitted:
            batches_created += 1
            logger.info(f"Submitted video batch {batch_job_id}")
        else:
            logger.warning(f"Failed to submit batch {batch_job_id} - queue full")
            break

    return {
        "status": "processing",
        "job_id_base": job_id_base,
        "video_name": video_name,
        "total_frames": len(frames),
        "batches_created": batches_created,
        "frames_per_batch": buffer_size,
        "target_fps": target_fps,
    }


@app.websocket("/ws/stream")
async def inference_endpoint(websocket: WebSocket) -> None:
    """Receive frames and return inference results."""
    await websocket.accept()
    state = get_server_state()
    logger.info("Client connected")

    # Outgoing message queue for this connection (logs + results)
    outgoing_queue = asyncio.Queue()
    pending_tasks: list[asyncio.Task] = []

    def log_callback(msg: dict) -> None:
        """Callback to receive log messages from jobs."""
        task = asyncio.create_task(outgoing_queue.put(msg))
        pending_tasks.append(task)

    # Register log callback with JobManager
    state.job_manager.register_log_callback(log_callback)

    # Connection-level config and state
    connection_job_id = f"video_job_{uuid.uuid4().hex[:8]}"
    video_cfg = config.jobs.get("video", {})

    # Extract config values for batch job creation
    buffer_size = video_cfg.get("buffer_size", 8)
    overlap_frames = video_cfg.get("overlap_frames", 4)
    default_fps = video_cfg.get("default_fps", 5)
    prompt = video_cfg.get("prompt", "Describe what you see in the video.")
    max_new_tokens = video_cfg.get("max_new_tokens", 128)

    # Track WebSocket connection state
    connection_active = True

    # Result callback for batch jobs
    async def result_handler(result_data: dict) -> None:
        """Handle result from VideoJob and queue for sending."""
        if not connection_active:
            return  # Skip if connection is closed

        try:
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

            # Queue for sending
            await outgoing_queue.put(result_data)

        except Exception as e:
            logger.error(f"Error in result handler: {e}", exc_info=True)

    # Wrap async callback for sync context
    def sync_result_callback(result_data: dict) -> None:
        task = asyncio.create_task(result_handler(result_data))
        pending_tasks.append(task)

    # Producer loop which receives frames and buffers them locally
    async def receive_loop() -> None:
        # Local frame buffer for this connection
        frame_buffer = FrameBuffer(
            buffer_size=buffer_size,
            overlap_frames=overlap_frames,
        )
        batch_counter = 0
        client_fps = default_fps

        try:
            while True:
                # Support both text and binary JSON payloads from clients
                try:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect()

                    raw_payload = message.get("text")
                    if raw_payload is None:
                        raw_bytes = message.get("bytes")
                        if raw_bytes is None:
                            continue  # Ignore keepalives without payload
                        try:
                            raw_payload = raw_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            logger.error(
                                "Received non-UTF8 binary payload; dropping frame"
                            )
                            continue

                    data = json.loads(raw_payload)
                except RuntimeError as e:
                    if "WebSocket is not connected" in str(e):
                        raise WebSocketDisconnect() from e
                    raise e
                except json.JSONDecodeError:
                    logger.error(
                        "Invalid JSON payload received on /ws/stream; dropping frame"
                    )
                    continue

                frame_id = data["frame_id"]
                capture_time = data.get("timestamp")
                arrival_time = time.time()

                if capture_time is not None:
                    latency_ms = (arrival_time - capture_time) * 1000.0
                    msg = (
                        f"Frame {frame_id} timing: capture={capture_time:.3f}s, "
                        f"arrival={arrival_time:.3f}s, latency={latency_ms:.1f}ms"
                    )
                    logger.debug(msg)

                if "measured_fps" in data:
                    deviation = abs(data.get("fps", 0) - data.get("measured_fps", 0))
                    if deviation > 100.0:
                        logger.warning(
                            "Network/processing lag detected: capture=%s, actual=%.1f",
                            data.get("fps"),
                            data.get("measured_fps"),
                        )

                # Update client FPS if provided
                if "fps" in data:
                    client_fps = float(data["fps"])

                # Decode frame
                frame_b64 = data["frame"]
                image_data = base64.b64decode(frame_b64)
                image = Image.open(BytesIO(image_data))

                # Add frame to local buffer
                frame_buffer.add_frame(image)

                # When buffer reaches threshold, create and submit batch job
                if frame_buffer.is_ready():
                    batch_job_id = f"{connection_job_id}_batch_{batch_counter}"

                    # Create one-shot VideoJob with buffered frames
                    from iris.server.inference.jobs import VideoJob
                    batch_job = VideoJob(
                        job_id=batch_job_id,
                        model=state.model,
                        processor=state.processor,
                        executor=state.queue.executor,
                        frames=frame_buffer.get_batch(),
                        prompt=prompt,
                        buffer_size=buffer_size,
                        overlap_frames=overlap_frames,
                        default_fps=default_fps,
                        max_new_tokens=max_new_tokens,
                        client_fps=client_fps,
                    )

                    # Set result callback
                    batch_job.result_callback = sync_result_callback

                    if state.shutting_down:
                        logger.info(
                            "Shutdown in progress; skipping batch submission %s",
                            batch_job_id,
                        )
                        break

                    # Submit directly to queue
                    await state.queue.submit(batch_job)

                    # Log with queue depth
                    queue_depth = state.queue.queue.qsize()
                    logger.info(f"Submitted {batch_job_id}, queue_depth={queue_depth}")

                    # Keep last N frames for temporal overlap
                    frame_buffer.slide_window()
                    batch_counter += 1

        except WebSocketDisconnect as e:
            code = getattr(e, "code", None)
            reason = getattr(e, "reason", None)
            logger.info(
                "Client disconnected (Receive Loop) code=%s reason=%s",
                code,
                reason,
            )

            # Submit partial batch if there are remaining frames
            if len(frame_buffer) > 0:
                batch_job_id = f"{connection_job_id}_batch_{batch_counter}_partial"
                logger.info(f"Submitting partial batch with {len(frame_buffer)} frames")

                from iris.server.inference.jobs import VideoJob
                batch_job = VideoJob(
                    job_id=batch_job_id,
                    model=state.model,
                    processor=state.processor,
                    executor=state.queue.executor,
                    frames=frame_buffer.get_batch(),
                    prompt=prompt,
                    buffer_size=buffer_size,
                    overlap_frames=overlap_frames,
                    default_fps=default_fps,
                    max_new_tokens=max_new_tokens,
                    client_fps=client_fps,
                )
                batch_job.result_callback = sync_result_callback

                if state.shutting_down:
                    logger.info(
                        "Shutdown in progress; skipping partial batch submission %s",
                        batch_job_id,
                    )
                else:
                    await state.queue.submit(batch_job)

                queue_depth = state.queue.queue.qsize()
                logger.info(f"Submitted {batch_job_id}, queue_depth={queue_depth}")

        except Exception as e:
            logger.error("Receive loop error: %s", e, exc_info=True)

    # Consumer loop which watches queue result and sends them to the client
    async def send_loop() -> None:
        try:
            while True:
                # Check for log messages or results (non-blocking)
                try:
                    msg = await outgoing_queue.get()
                    await websocket.send_json(msg)
                except asyncio.CancelledError:
                    break
        except asyncio.CancelledError:
            logger.debug("Send loop cancelled during shutdown")
            # Don't raise - allow graceful cleanup
        except WebSocketDisconnect as e:
            code = getattr(e, "code", None)
            reason = getattr(e, "reason", None)
            logger.info(
                "Client disconnected (Send Loop) code=%s reason=%s",
                code,
                reason,
            )
        except Exception as e:
            logger.error("Send loop error: %s", e, exc_info=True)

    # Run both functions concurrently
    # This runs until one of them finishes (usually the receive loop on disconnect)
    try:
        await asyncio.gather(receive_loop(), send_loop())
    finally:
        logger.info(f"WebSocket disconnecting for {connection_job_id}")

        # Mark connection as inactive to stop result handler tasks
        connection_active = False

        # Cancel any pending result handler tasks
        for task in pending_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to finish cancellation
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # No job cleanup needed - batch jobs complete naturally
        logger.info(f"WebSocket cleanup complete for {connection_job_id}")


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


@app.delete("/system/clear")
async def clear_system(
    clear_logs: bool = True,
    stop_active_jobs: bool = True,
) -> dict[str, Any]:
    """Clear server state including queue, jobs, and metrics.

    This endpoint provides a way to reset the server state without restarting.
    Useful when switching between different experiments or data sources.

    Args:
        clear_logs: Whether to delete metrics files from disk (default: True)
        stop_active_jobs: Whether to stop running jobs (default: True)

    Returns:
        Dictionary with detailed status of clearing operations
    """
    state = get_server_state()

    result = {
        "status": "success",
        "cleared": {},
        "errors": [],
        "timestamp": time.time(),
    }

    # 1. Stop all active jobs (if requested)
    if stop_active_jobs and state.job_manager:
        try:
            logger.info("Stopping all active jobs...")
            job_result = await state.job_manager.stop_all_jobs()
            result["cleared"]["active_jobs"] = job_result["stopped_count"]
            if job_result["errors"]:
                result["errors"].extend(job_result["errors"])
        except Exception as e:
            error_msg = f"Failed to stop active jobs: {e}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            result["cleared"]["active_jobs"] = "error"
    else:
        result["cleared"]["active_jobs"] = "skipped"

    # 2. Clear pending jobs from queue
    if state.queue:
        try:
            logger.info("Clearing job queue...")
            cleared_count = await state.queue.clear_queue()
            result["cleared"]["pending_jobs"] = cleared_count
        except Exception as e:
            error_msg = f"Failed to clear queue: {e}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            result["cleared"]["pending_jobs"] = "error"
    else:
        result["cleared"]["pending_jobs"] = "queue_not_initialized"

    # 3. Reset metrics (and optionally delete files)
    if state.metrics:
        try:
            logger.info(f"Resetting metrics (clear_files={clear_logs})...")
            metrics_result = state.metrics.reset(clear_files=clear_logs)
            result["cleared"]["metrics"] = {
                "previous_totals": metrics_result["previous_totals"],
                "files_deleted": metrics_result["files_deleted"],
                "new_session_id": metrics_result["new_session_id"],
            }
            if metrics_result["errors"]:
                result["errors"].extend(metrics_result["errors"])
        except Exception as e:
            error_msg = f"Failed to reset metrics: {e}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            result["cleared"]["metrics"] = "error"
    else:
        result["cleared"]["metrics"] = "metrics_disabled"

    # 4. Determine overall status
    if result["errors"]:
        result["status"] = "partial_success"
        logger.warning(f"System clear completed with {len(result['errors'])} errors")
    else:
        logger.info("System clear completed successfully")

    return result


def main() -> None:
    """Entry point for server."""
    import uvicorn

    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except KeyboardInterrupt:
        # Handle Ctrl+C during uvicorn startup
        logger.info("Server interrupted during startup. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()

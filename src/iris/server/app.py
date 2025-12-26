"""IRIS Inference Server - receives frames, runs VLM inference."""

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.frame_buffer import FrameBuffer
from iris.server.inference.executor import InferenceExecutor
from iris.server.jobs.manager import JobManager
from iris.server.logging_handler import WebSocketLogHandler

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
    """Return a lightweight snapshot of queue/active job counts.

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


def _wait_for_drain_and_shutdown() -> None:
    """Background thread that waits for queue drain then triggers shutdown."""
    while True:
        time.sleep(0.5)
        if force_shutdown_event.is_set():
            return  # Already forcing shutdown
        status = _queue_status_snapshot()
        if status["active_jobs"] == 0 and status["queue_depth"] == 0:
            logger.info("All queued jobs completed - initiating server shutdown")
            # Send SIGINT to self - default handler will trigger uvicorn shutdown
            os.kill(os.getpid(), signal.SIGINT)
            return


def handle_shutdown_signal(signum: int, frame: Any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown.

    On first signal: Log status, mark shutdown, wait for ALL queued jobs to complete.
    On second signal: Force immediate shutdown.
    """
    global shutdown_count
    shutdown_count += 1

    state = get_server_state()
    state.shutting_down = True

    # Capture quick stats to help the operator decide whether to wait or force-exit
    status = _queue_status_snapshot()

    if shutdown_count == 1:
        logger.info(
            "Received %s. Graceful shutdown initiated; active_jobs=%d, queue_depth=%d, pending_results=%d.",
            signal.Signals(signum).name,
            status["active_jobs"],
            status["queue_depth"],
            status["pending_results"],
        )

        shutdown_event.set()
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
            threading.Thread(target=_wait_for_drain_and_shutdown, daemon=True).start()
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
    force_shutdown_event.set()


async def global_result_drainer(state: Any) -> None:
    """Drain global results queue to prevent memory leaks."""
    try:
        while True:
            _job = await state.queue.results.get()
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

    logger.info("Starting inference executor with per-worker model loading...")
    state.queue = InferenceExecutor(
        max_queue_size=config.max_queue_size,
        num_workers=config.num_workers,
        model_id=config.model_id,
        hardware=config.vlm_hardware,
        model_dtype=config.model_dtype,
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
        status = _queue_status_snapshot()
        logger.info(
            "Stopping executor (active_jobs=%d, queue_depth=%d)",
            status["active_jobs"],
            status["queue_depth"],
        )
        await state.queue.stop()
    logger.info("Server stopped.")


app = FastAPI(title="IRIS Inference Server", lifespan=lifespan)


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

    # Track WebSocket connection state (use dict for proper closure capture)
    connection_state = {"active": True}

    # Result callback for batch jobs
    async def result_handler(result_data: dict) -> None:
        """Handle result from VideoJob and queue for sending."""
        if not connection_state["active"]:
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
                # Check if server is shutting down - exit cleanly
                if state.shutting_down:
                    logger.info(
                        f"Shutdown initiated - closing connection {connection_job_id}"
                    )
                    break

                # Support both text and binary JSON payloads from clients
                try:
                    # Use timeout to periodically check shutdown flag
                    message = await asyncio.wait_for(websocket.receive(), timeout=0.5)
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
                except TimeoutError:
                    continue  # Re-check shutdown flag
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
                        model=None,  # Will be injected by worker
                        processor=None,  # Will be injected by worker
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
                    model=None,  # Will be injected by worker
                    processor=None,  # Will be injected by worker
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
            while connection_state["active"]:
                try:
                    # Use timeout to periodically check connection state
                    msg = await asyncio.wait_for(outgoing_queue.get(), timeout=0.5)
                    if not connection_state["active"]:
                        break
                    await websocket.send_json(msg)
                except TimeoutError:
                    continue  # Re-check connection_state
                except asyncio.CancelledError:
                    break
                except RuntimeError as e:
                    if "websocket.send" in str(e).lower():
                        logger.debug("WebSocket closed, exiting send loop")
                        break
                    raise
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
    except asyncio.CancelledError:
        logger.debug(f"WebSocket tasks cancelled for {connection_job_id}")
    finally:
        logger.info(f"WebSocket disconnecting for {connection_job_id}")

        # Mark connection as inactive to stop result handler tasks
        connection_state["active"] = False

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


def main() -> None:
    """Entry point for server."""
    import uvicorn

    try:
        uvicorn.run(app, host=config.host, port=config.port)
    except KeyboardInterrupt:
        # Handle Ctrl+C during uvicorn startup
        logger.info("Server interrupted during startup. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()

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
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from iris.config import _yaml_config
from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.frame_buffer import FrameBuffer
from iris.server.inference.executor import InferenceExecutor
from iris.server.jobs.manager import JobManager
from iris.server.logging_handler import WebSocketLogHandler


@dataclass
class SessionState:
    """Tracks session state for a WebSocket connection.

    Attributes:
        session_id: Unique identifier for this session.
        config: Session configuration (frames_per_segment, overlap_frames).
        mode: Session mode ("live" or "analysis").
        total_frames: Total frames expected (None for live mode).
        start_time: Unix timestamp when session started.
        frames_received: Counter of frames received.
        segments_processed: Counter of segments/batches processed.
    """

    session_id: str
    config: dict = field(default_factory=dict)
    mode: str = "live"
    total_frames: int | None = None
    start_time: float = field(default_factory=time.time)
    frames_received: int = 0
    segments_processed: int = 0

    def to_metrics(self, queue_depth: int) -> dict:
        """Generate metrics snapshot for broadcast.

        Args:
            queue_depth: Current number of jobs in queue.

        Returns:
            Dictionary with session metrics.
        """
        elapsed = time.time() - self.start_time
        rate = self.segments_processed / elapsed if elapsed > 0 else 0.0

        # Calculate segments_total for analysis mode
        segments_total = None
        if self.mode == "analysis" and self.total_frames:
            frames_per_segment = self.config.get("frames_per_segment", 8)
            overlap_frames = self.config.get("overlap_frames", 4)
            frames_per_chunk = frames_per_segment - overlap_frames
            if frames_per_chunk > 0:
                segments_total = max(
                    1, (self.total_frames + frames_per_chunk - 1) // frames_per_chunk
                )

        return {
            "type": "session_metrics",
            "session_id": self.session_id,
            "elapsed_seconds": round(elapsed, 1),
            "segments_processed": self.segments_processed,
            "segments_total": segments_total,
            "queue_depth": queue_depth,
            "processing_rate": round(rate, 2),
            "frames_received": self.frames_received,
            "batch_size": config.batch_inference.batch_size if config.batch_inference.enabled else 1,
        }


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


@app.get("/api/config/defaults")
async def get_config_defaults() -> dict:
    """Return configuration defaults for frontend initialization.

    Returns:
        Dictionary with server, video, and segment configuration defaults.
    """
    return {
        "server": {
            "host": config.host,
            "port": config.port,
            "model_id": config.model_id,
        },
        "video": {
            "width": _yaml_config.get("client", {}).get("video", {}).get("width", 640),
            "height": _yaml_config.get("client", {})
            .get("video", {})
            .get("height", 480),
            "capture_fps": _yaml_config.get("client", {})
            .get("video", {})
            .get("capture_fps", 10),
            "jpeg_quality": _yaml_config.get("client", {})
            .get("video", {})
            .get("jpeg_quality", 80),
            "camera_index": _yaml_config.get("client", {})
            .get("video", {})
            .get("camera_index", 0),
        },
        "segment": {
            "segment_time": _yaml_config.get("client", {})
            .get("segment", {})
            .get("segment_time", 1.0),
            "frames_per_segment": _yaml_config.get("client", {})
            .get("segment", {})
            .get("frames_per_segment", 8),
            "overlap_frames": _yaml_config.get("client", {})
            .get("segment", {})
            .get("overlap_frames", 4),
        },
    }


@app.post("/api/queue/clear")
async def clear_queue() -> dict:
    """Clear all pending inference jobs and free GPU memory.

    This endpoint clears the inference queue and triggers garbage collection
    to free any associated GPU memory. Use this when you want to stop
    processing and start fresh without waiting for queued jobs to complete.

    Returns:
        Dictionary with cleared count and status.
    """
    import gc

    state = get_server_state()

    # Clear the job queue
    cleared_count = 0
    if state.queue and state.queue.queue:
        while not state.queue.queue.empty():
            try:
                state.queue.queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break

    # Force garbage collection to free memory
    gc.collect()

    # Clear GPU cache if torch/CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except ImportError:
        pass  # torch not available

    logger.info(f"Cleared {cleared_count} jobs from queue")
    return {"cleared": cleared_count, "status": "ok"}


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for monitoring.

    Returns:
        Dictionary with status and queue depth.
    """
    state = get_server_state()
    queue_depth = state.queue.queue.qsize() if state.queue and state.queue.queue else 0
    return {
        "status": "ok",
        "queue_depth": queue_depth,
        "model_loaded": state.model_loaded,
    }


@app.websocket("/ws/stream")
async def inference_endpoint(websocket: WebSocket) -> None:
    """Receive frames and return inference results.

    Protocol:
        1. Client sends `session_config` message first with config params
        2. Server responds with `session_ack` containing session_id
        3. Client sends frame messages
        4. Server broadcasts `session_metrics` every 500ms
        5. Server sends `result` messages as inference completes
    """
    await websocket.accept()
    state = get_server_state()
    logger.info("Client connected")

    # Outgoing message queue for this connection (logs + results)
    outgoing_queue: asyncio.Queue[dict] = asyncio.Queue()
    pending_tasks: list[asyncio.Task] = []

    def log_callback(msg: dict) -> None:
        """Callback to receive log messages from jobs."""
        task = asyncio.create_task(outgoing_queue.put(msg))
        pending_tasks.append(task)

    # Register log callback with JobManager
    state.job_manager.register_log_callback(log_callback)

    # Track WebSocket connection state (use dict for proper closure capture)
    connection_state = {"active": True}

    # Session state - populated when session_config is received
    session: SessionState | None = None

    # Default config from server config.yaml
    video_cfg = config.jobs.get("video", {})
    default_buffer_size = video_cfg.get("buffer_size", 8)
    default_overlap = video_cfg.get("overlap_frames", 4)
    default_fps = video_cfg.get("default_fps", 5)
    prompt = video_cfg.get("prompt", "Describe what you see in the video.")
    max_new_tokens = video_cfg.get("max_new_tokens", 128)

    # These will be set from session config
    buffer_size = default_buffer_size
    overlap_frames = default_overlap

    # Result callback for batch jobs
    async def result_handler(result_data: dict) -> None:
        """Handle result from VideoJob and queue for sending."""
        nonlocal session
        if not connection_state["active"]:
            logger.debug(f"Skipping result {result_data.get('job_id')} - connection inactive")
            return  # Skip if connection is closed

        try:
            # Update session metrics
            if session:
                session.segments_processed += 1

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
            logger.info(f"Queueing result for job {result_data.get('job_id')} to send to client")
            await outgoing_queue.put(result_data)

        except Exception as e:
            logger.error(f"Error in result handler: {e}", exc_info=True)

    # Wrap async callback for sync context
    def sync_result_callback(result_data: dict) -> None:
        task = asyncio.create_task(result_handler(result_data))
        pending_tasks.append(task)

    # Producer loop which receives frames and buffers them locally
    async def receive_loop() -> None:
        nonlocal session, buffer_size, overlap_frames

        # Generate session ID for this connection
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        connection_job_id = f"video_job_{session_id}"

        # Local frame buffer - created after session config received
        frame_buffer: FrameBuffer | None = None
        batch_counter = 0
        client_fps = default_fps

        # Batch inference accumulator (analysis mode only)
        batch_accumulator: list[dict] = []  # [{"frames": [...], "segment_id": str, "prompt": str, "client_fps": float}, ...]
        batch_segment_counter = 0

        # Track the last job to wait for it on completion
        last_submitted_job: Any = None

        try:
            # Wait for session_config as first message
            logger.info(f"Waiting for session_config from client...")

            while True:
                # Check if server is shutting down - exit cleanly
                if state.shutting_down:
                    logger.info(
                        f"Shutdown initiated - closing connection {connection_job_id}"
                    )
                    break

                # ---------------------------------------------------------
                # TCP BACKPRESSURE: Pause reading if queue is too full
                # ---------------------------------------------------------
                # If we stop reading from the socket, the TCP window closes,
                # forcing the client (browser) to slow down sending.
                queue_depth = state.queue.queue.qsize() if state.queue else 0
                if (
                    session
                    and session.mode == "live"
                    and queue_depth > (config.live_queue_threshold + 1)
                ):
                    await asyncio.sleep(0.1)
                    continue
                # ---------------------------------------------------------

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

                if data.get("type") == "complete":
                    logger.info(f"Received completion signal for session {session_id}. Flushing buffers...")

                    # 1. Flush any pending batch segments
                    if batch_accumulator:
                        from iris.server.inference.jobs.batch_video import BatchVideoJob
                        final_batch_id = f"{connection_job_id}_batch_final"
                        
                        logger.info(f"Flushing {len(batch_accumulator)} segments in final batch")
                        final_batch = BatchVideoJob(
                            job_id=final_batch_id,
                            model=None, processor=None, executor=state.queue.executor,
                            segments=batch_accumulator.copy(),
                            max_new_tokens=max_new_tokens,
                        )
                        final_batch.result_callback = sync_result_callback
                        final_batch.log_callback = log_callback
                        
                        await state.queue.submit(final_batch)
                        last_submitted_job = final_batch
                        batch_accumulator.clear()

                    # 2. Flush any partial frames in buffer
                    elif frame_buffer and len(frame_buffer) > 0:
                        from iris.server.inference.jobs import VideoJob
                        partial_id = f"{connection_job_id}_batch_partial"
                        
                        logger.info(f"Flushing {len(frame_buffer)} partial frames")
                        # Calculate timestamp for partial batch
                        buffered_timestamps = frame_buffer.get_metadata()
                        video_time_ms = (buffered_timestamps[0] * 1000.0) if (buffered_timestamps and buffered_timestamps[0]) else 0.0

                        batch_job = VideoJob(
                            job_id=partial_id,
                            model=None, processor=None, executor=state.queue.executor,
                            frames=frame_buffer.get_batch(),
                            prompt=prompt,
                            buffer_size=buffer_size,
                            overlap_frames=overlap_frames,
                            default_fps=default_fps,
                            max_new_tokens=max_new_tokens,
                            client_fps=client_fps,
                            video_time_ms=video_time_ms,
                        )
                        batch_job.result_callback = sync_result_callback
                        
                        await state.queue.submit(batch_job)
                        last_submitted_job = batch_job

                    # 3. Wait for the last job to complete
                    if last_submitted_job:
                        logger.info(f"Waiting for job {last_submitted_job.job_id} to complete...")
                        while last_submitted_job.status.value not in ["completed", "failed", "cancelled"]:
                            await asyncio.sleep(0.1)
                        logger.info("All jobs completed.")

                    # 4. Send completion acknowledgement
                    await outgoing_queue.put({"type": "processing_complete"})
                    break

                # Handle session_config message
                if data.get("type") == "session_config":
                    cfg = data.get("config", {})
                    buffer_size = cfg.get("frames_per_segment", default_buffer_size)
                    overlap_frames = cfg.get("overlap_frames", default_overlap)

                    # Create session state
                    session = SessionState(
                        session_id=session_id,
                        config={
                            "frames_per_segment": buffer_size,
                            "overlap_frames": overlap_frames,
                        },
                        mode=data.get("mode", "live"),
                        total_frames=data.get("total_frames"),
                        start_time=time.time(),
                    )

                    # Create frame buffer with session config
                    frame_buffer = FrameBuffer(
                        buffer_size=buffer_size,
                        overlap_frames=overlap_frames,
                    )

                    # Send session_ack
                    ack_message = {
                        "type": "session_ack",
                        "session_id": session_id,
                        "config": session.config,
                    }
                    await outgoing_queue.put(ack_message)

                    logger.info(
                        f"Session {session_id} configured: "
                        f"frames_per_segment={buffer_size}, overlap={overlap_frames}, "
                        f"mode={session.mode}"
                    )
                    continue

                # Ensure session is configured before processing frames
                if session is None or frame_buffer is None:
                    logger.warning("Received frame before session_config, ignoring")
                    continue

                # Process frame message
                if "frame_id" not in data:
                    logger.warning(
                        f"Unknown message type: {data.get('type', 'unknown')}"
                    )
                    continue

                frame_id = data["frame_id"]
                session.frames_received += 1
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

                # Decode frame to raw bytes ONLY - do not decode to PIL yet
                frame_b64 = data.get("frame")
                if not frame_b64:
                    continue

                try:
                    frame_bytes = base64.b64decode(frame_b64)
                    frame_timestamp = data.get("timestamp")  # Capture timestamp

                    # ---------------------------------------------------------
                    # BACKPRESSURE: Drop frames if queue is too full
                    # ---------------------------------------------------------
                    current_queue_depth = state.queue.queue.qsize()
                    if (
                        session
                        and session.mode == "live"
                        and current_queue_depth >= config.live_queue_threshold
                    ):
                        logger.debug(
                            f"Dropping frame {frame_id}, queue depth {current_queue_depth}"
                        )
                        continue
                    # ---------------------------------------------------------

                    # Add raw bytes to buffer with timestamp metadata
                    frame_buffer.add_frame(frame_bytes, metadata=frame_timestamp)
                except Exception as e:
                    logger.error(f"Error decoding frame: {e}")
                    continue

                # When buffer reaches threshold, handle batching
                if frame_buffer.is_ready():
                    # Get buffered frames and metadata
                    buffered_frames = frame_buffer.get_batch()
                    buffered_timestamps = frame_buffer.get_metadata()

                    # Determine video time for this segment (start time)
                    video_time_ms = 0.0
                    if buffered_timestamps and buffered_timestamps[0] is not None:
                        # Use actual timestamp from first frame
                        video_time_ms = buffered_timestamps[0] * 1000.0
                    else:
                        # Fallback to calculated time
                        stride = buffer_size - overlap_frames
                        if stride > 0 and client_fps > 0:
                            video_time_ms = (batch_segment_counter * stride / client_fps) * 1000.0

                    # Check if batch inference enabled for analysis mode
                    batch_cfg = config.batch_inference
                    should_batch = (
                        batch_cfg.enabled
                        and session.mode == "analysis"
                        and batch_cfg.batch_size > 1
                    )

                    if should_batch:
                        # Accumulate segment for batch processing
                        segment_data = {
                            "frames": buffered_frames,
                            "segment_id": f"seg_{batch_segment_counter}",
                            "prompt": prompt,
                            "client_fps": client_fps,
                            "video_time_ms": video_time_ms,
                        }
                        batch_accumulator.append(segment_data)
                        batch_segment_counter += 1

                        logger.info(
                            f"Accumulated segment {batch_segment_counter}, "
                            f"batch progress: {len(batch_accumulator)}/{batch_cfg.batch_size}"
                        )

                        # Check if batch ready
                        if len(batch_accumulator) >= batch_cfg.batch_size:
                            # Create BatchVideoJob
                            from iris.server.inference.jobs.batch_video import BatchVideoJob

                            batch_job_id = f"{connection_job_id}_batch_{batch_counter}"
                            batch_job = BatchVideoJob(
                                job_id=batch_job_id,
                                model=None,  # Injected by worker
                                processor=None,
                                executor=state.queue.executor,
                                segments=batch_accumulator.copy(),
                                max_new_tokens=max_new_tokens,
                            )

                            batch_job.result_callback = sync_result_callback
                            batch_job.log_callback = log_callback

                            if not state.shutting_down:
                                await state.queue.submit(batch_job)
                                last_submitted_job = batch_job
                                queue_depth = state.queue.queue.qsize()
                                logger.info(
                                    f"Submitted BatchVideoJob {batch_job_id}: "
                                    f"{len(batch_accumulator)} segments, queue_depth={queue_depth}"
                                )

                            # Clear accumulator
                            batch_accumulator.clear()
                            batch_counter += 1

                        # Slide window for next segment
                        frame_buffer.slide_window()

                    else:
                        # Original single-segment processing
                        batch_job_id = f"{connection_job_id}_batch_{batch_counter}"

                        from iris.server.inference.jobs import VideoJob

                        batch_job = VideoJob(
                            job_id=batch_job_id,
                            model=None,
                            processor=None,
                            executor=state.queue.executor,
                            frames=buffered_frames,
                            prompt=prompt,
                            buffer_size=buffer_size,
                            overlap_frames=overlap_frames,
                            default_fps=default_fps,
                            max_new_tokens=max_new_tokens,
                            client_fps=client_fps,
                            video_time_ms=video_time_ms,
                        )

                        batch_job.result_callback = sync_result_callback

                        if not state.shutting_down:
                            await state.queue.submit(batch_job)
                            last_submitted_job = batch_job
                            queue_depth = state.queue.queue.qsize()
                            logger.info(
                                f"Submitted VideoJob {batch_job_id}: "
                                f"{len(buffered_frames)} frames, queue_depth={queue_depth}"
                            )

                        batch_counter += 1
                        frame_buffer.slide_window()

        except WebSocketDisconnect as e:
            code = getattr(e, "code", None)
            reason = getattr(e, "reason", None)
            logger.info(
                "Client disconnected (Receive Loop) code=%s reason=%s",
                code,
                reason,
            )

            # NOTE: We no longer flush here on disconnect. The client is responsible
            # for sending a "complete" message before disconnecting to ensure flushing.

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
                    msg_type = msg.get("type", "unknown")
                    logger.info(f"Sending message type={msg_type} to client")
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

    # Metrics broadcast loop - sends session metrics every 500ms
    async def metrics_broadcast_loop() -> None:
        """Broadcast session metrics to client every 500ms.

        Only sends metrics when there's actual activity to avoid spamming
        the connection with unchanged data.
        """
        last_segments_processed = 0
        last_queue_depth = 0
        last_frames_received = 0

        try:
            while connection_state["active"]:
                await asyncio.sleep(0.5)
                if session and connection_state["active"]:
                    queue_depth = state.queue.queue.qsize() if state.queue else 0

                    # Only send metrics if something has changed
                    if (
                        session.segments_processed != last_segments_processed
                        or queue_depth != last_queue_depth
                        or session.frames_received != last_frames_received
                    ):
                        metrics = session.to_metrics(queue_depth)
                        await outgoing_queue.put(metrics)

                        # Update last known values
                        last_segments_processed = session.segments_processed
                        last_queue_depth = queue_depth
                        last_frames_received = session.frames_received
        except asyncio.CancelledError:
            pass  # Normal shutdown
        except Exception as e:
            logger.error("Metrics broadcast error: %s", e, exc_info=True)

    # Run all loops concurrently
    # This runs until one of them finishes (usually the receive loop on disconnect)
    try:
        await asyncio.gather(
            receive_loop(),
            send_loop(),
            metrics_broadcast_loop(),
        )
    except asyncio.CancelledError:
        logger.debug("WebSocket tasks cancelled")
    finally:
        session_id = session.session_id if session else "unknown"
        logger.info(f"WebSocket disconnecting for session {session_id}")

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
        logger.info(f"WebSocket cleanup complete for session {session_id}")


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
        uvicorn.run(app, host=config.host, port=config.port, log_level="info")
    except KeyboardInterrupt:
        # Handle Ctrl+C during uvicorn startup
        logger.info("Server interrupted during startup. Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()

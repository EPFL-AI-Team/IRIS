"""WebSocket route handlers for IRIS server."""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.frame_buffer import FrameBuffer
from iris.server.inference.jobs import VideoJob

logger = logging.getLogger(__name__)
config = ServerConfig()
router = APIRouter()


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
        """Generate metrics snapshot for broadcast."""
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
        }


@router.websocket("/ws/stream")
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

        try:
            # Wait for session_config as first message
            logger.info("Waiting for session_config from client...")

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
                    logger.warning(f"Unknown message type: {data.get('type', 'unknown')}")
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

                # Decode frame
                frame_b64 = data["frame"]
                image_data = base64.b64decode(frame_b64)
                image = Image.open(BytesIO(image_data))

                # Add frame to local buffer
                frame_buffer.add_frame(image)

                # When buffer reaches threshold, create and submit batch job
                if frame_buffer.is_ready():
                    batch_job_id = f"{connection_job_id}_batch_{batch_counter}"

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
            if frame_buffer is not None and len(frame_buffer) > 0:
                partial_job_id = f"{connection_job_id}_batch_{batch_counter}_partial"
                logger.info(f"Submitting partial batch with {len(frame_buffer)} frames")

                batch_job = VideoJob(
                    job_id=partial_job_id,
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
                        partial_job_id,
                    )
                else:
                    await state.queue.submit(batch_job)

                queue_depth = state.queue.queue.qsize()
                logger.info(f"Submitted {partial_job_id}, queue_depth={queue_depth}")

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

    # Metrics broadcast loop - sends session metrics every 500ms
    async def metrics_broadcast_loop() -> None:
        """Broadcast session metrics to client every 500ms."""
        try:
            while connection_state["active"]:
                await asyncio.sleep(0.5)
                if session and connection_state["active"]:
                    queue_depth = state.queue.queue.qsize() if state.queue else 0
                    metrics = session.to_metrics(queue_depth)
                    await outgoing_queue.put(metrics)
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


@router.websocket("/ws/logs")
async def log_streaming_endpoint(websocket: WebSocket) -> None:
    """Stream server logs to connected clients."""
    state = get_server_state()

    if not config.enable_log_streaming or state.log_handler is None:
        await websocket.close(code=1008, reason="Log streaming is disabled")
        return

    await websocket.accept()
    state.log_handler.add_connection(websocket)
    logger.info(
        "Log streaming client connected (total: %d)",
        state.log_handler.get_connection_count(),
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
        state.log_handler.remove_connection(websocket)
        logger.info(
            "Log streaming client removed (remaining: %d)",
            state.log_handler.get_connection_count(),
        )

"""WebSocket routes for IRIS Inference Server."""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.server.frame_buffer import FrameBuffer
from iris.server.logging_handler import WebSocketLogHandler

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
    duration_sec: float | None = None  # Required for analysis mode
    start_time: float = field(default_factory=time.time)
    frames_received: int = 0
    segments_processed: int = 0

    def to_metrics(self, queue_depth: int) -> dict:
        """Generate metrics snapshot for broadcast.

        Args:
            queue_depth: Current number of jobs in queue (internal use, not sent to client).

        Returns:
            Dictionary with session metrics.
        """
        elapsed = time.time() - self.start_time

        # Calculate segments_total for analysis mode using logical FPS
        segments_total = None
        batch_size = None
        if self.mode == "analysis" and self.duration_sec:
            frames_per_segment = self.config.get("frames_per_segment", 8)
            overlap_frames = self.config.get("overlap_frames", 4)
            segment_time = self.config.get("segment_time", 1.0)

            # Validate inputs
            if segment_time <= 0:
                logger.warning(f"Invalid segment_time: {segment_time}, using default 1.0")
                segment_time = 1.0

            # Calculate logical FPS and effective frames
            logical_fps = frames_per_segment / segment_time
            effective_total_frames = int(self.duration_sec * logical_fps)

            frames_per_chunk = frames_per_segment - overlap_frames
            if frames_per_chunk > 0:
                segments_total = max(
                    1, (effective_total_frames + frames_per_chunk - 1) // frames_per_chunk
                )

            # Include batch size only for analysis mode
            batch_size = config.batch_inference.batch_size if config.batch_inference.enabled else 1

        return {
            "type": "session_metrics",
            "session_id": self.session_id,
            "elapsed_seconds": round(elapsed, 1),
            "segments_processed": self.segments_processed,
            "segments_total": segments_total,
            "batch_size": batch_size,
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
            logger.info("Waiting for session_config from client...")

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

                    # 1. Promote partial buffer to accumulator if we are already batching
                    # This ensures the last segment (even if partial) is included in the final batch
                    if batch_accumulator and frame_buffer and len(frame_buffer) > 0:
                        logger.info(f"Promoting {len(frame_buffer)} partial frames to batch accumulator")
                        buffered_frames = frame_buffer.get_batch()
                        buffered_timestamps = frame_buffer.get_metadata()

                        video_time_ms = 0.0
                        if buffered_timestamps and buffered_timestamps[0] is not None:
                            video_time_ms = buffered_timestamps[0] * 1000.0
                        else:
                            stride = buffer_size - overlap_frames
                            if stride > 0 and client_fps > 0:
                                video_time_ms = (batch_segment_counter * stride / client_fps) * 1000.0

                        segment_data = {
                            "frames": buffered_frames,
                            "segment_id": f"seg_{batch_segment_counter}",
                            "prompt": prompt,
                            "client_fps": client_fps,
                            "video_time_ms": video_time_ms,
                        }
                        batch_accumulator.append(segment_data)
                        batch_segment_counter += 1
                        frame_buffer.clear()

                    # 2. Flush any pending batch segments
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

                    # 3. Flush any partial frames in buffer (only if not promoted above)
                    if frame_buffer and len(frame_buffer) > 0:
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

                    # 4. Wait for the last job to complete
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
                    segment_time = cfg.get("segment_time", 1.0)
                    mode = data.get("mode", "live")
                    duration_sec = data.get("duration_sec")

                    # Validate required fields for analysis mode
                    if mode == "analysis" and duration_sec is None:
                        logger.error("Analysis mode requires duration_sec in session_config")
                        await outgoing_queue.put({
                            "type": "error",
                            "message": "Analysis mode requires duration_sec parameter"
                        })
                        continue

                    # Create session state
                    session = SessionState(
                        session_id=session_id,
                        config={
                            "frames_per_segment": buffer_size,
                            "overlap_frames": overlap_frames,
                            "segment_time": segment_time,
                        },
                        mode=mode,
                        duration_sec=duration_sec,
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
                    
                    logger.debug(f"Segment video_time_ms: {video_time_ms} (timestamp={buffered_timestamps[0] if buffered_timestamps else 'None'}, batch_seg={batch_segment_counter})")

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
                            from iris.server.inference.jobs.batch_video import (
                                BatchVideoJob,
                            )

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
                        batch_segment_counter += 1
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
        """Broadcast session metrics to client every 1s.

        Sends metrics periodically to ensure elapsed time and rate
        updates are visible on the client even during long inference steps.
        """
        try:
            while connection_state["active"]:
                await asyncio.sleep(1.0)
                if session and connection_state["active"]:
                    queue_depth = state.queue.queue.qsize() if state.queue else 0

                    # Always send metrics to keep timer alive
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

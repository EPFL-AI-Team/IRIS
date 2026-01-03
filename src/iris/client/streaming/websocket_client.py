"""Client to allow streaming of the CameraCapture device"""

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import Callable

import websockets
from websockets.client import WebSocketClientProtocol

from iris.client.capture.camera import CameraCapture

logger = logging.getLogger(__name__)


class StreamingClient:
    """WebSocket client for streaming video frames"""

    def __init__(
        self,
        ws_url: str,
        camera: CameraCapture,
        jpeg_quality: int = 80,
        result_callback: Callable[[dict], None] | None = None,
        buffer_size: int = 8,
        overlap_frames: int = 4,
        session_config: dict | None = None,
    ):
        self.ws_url = ws_url
        self.camera = camera
        self.jpeg_quality = jpeg_quality
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time: float | None = None
        self.result_callback = result_callback
        self.connection_state = "disconnected"  # Track connection state

        # Session configuration
        # Use provided config or fall back to legacy args
        if session_config:
            self.session_config = session_config
            # Sync legacy attributes for consistency
            self.buffer_size = session_config.get("frames_per_segment", buffer_size)
            self.overlap_frames = session_config.get("overlap_frames", overlap_frames)
        else:
            self.buffer_size = buffer_size
            self.overlap_frames = overlap_frames
            self.session_config = {
                "frames_per_segment": buffer_size,
                "overlap_frames": overlap_frames,
            }

        self.capture_fps: float = float(camera.fps)

    async def stream(self) -> None:
        """Connect and stream frames with auto-reconnect."""
        self.running = True
        self.connection_state = "connecting"
        retry_delay = 1.0
        max_delay = 30.0

        attempt_count = 0
        while self.running:
            try:
                self.connection_state = "connecting"
                # Disable client-side pings; let server handle keepalive. This avoids
                # spurious timeouts (1011) on slow or bursty networks/inference.
                async with (
                    websockets.connect(
                        self.ws_url,
                        ping_interval=20,  # Send ping every 20s
                        ping_timeout=60,  # Timeout after 60s (increased to prevent timeout during inference)
                        close_timeout=30.0,
                        max_queue=None,
                    ) as ws
                ):
                    logger.info("Connected to %s", self.ws_url)
                    self.connection_state = "connected"
                    retry_delay = 1.0  # Reset on successful connection
                    attempt_count = 0
                    await self._stream_loop(ws)

            except websockets.exceptions.InvalidStatusCode as e:
                self.connection_state = "error"
                logger.error("Server returned invalid status code: %s", e)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
                attempt_count += 1
            except websockets.exceptions.WebSocketException as e:
                self.connection_state = "error"
                attempt_count += 1
                logger.warning(
                    "Connection failed: %s. Retrying in %.1fs... (attempt %d)",
                    e,
                    retry_delay,
                    attempt_count,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
            except ConnectionRefusedError:
                self.connection_state = "error"
                attempt_count += 1
                logger.error(
                    "Connection refused. Is the server running at %s? (attempt %d)",
                    self.ws_url,
                    attempt_count,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
            except Exception as e:
                self.connection_state = "error"
                logger.error("Unexpected error: %s", e, exc_info=True)
                break

    async def _stream_loop(self, ws: WebSocketClientProtocol) -> None:
        """Main streaming loop once connected.

        Protocol:
        1. Send session_config as first message
        2. Wait for session_ack response
        3. Start send/receive loops for frame streaming
        """
        # Send session_config FIRST before starting frame loop
        config_message = {
            "type": "session_config",
            "config": self.session_config,
            "mode": "live",
            "total_frames": None,
        }
        await ws.send(json.dumps(config_message))
        logger.info("Sent session_config: %s", self.session_config)

        # Wait for session_ack
        try:
            ack_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack_data = json.loads(ack_response)
            if ack_data.get("type") == "session_ack":
                self.session_id = ack_data.get("session_id")
                logger.info("Session established: %s", self.session_id)
                if self.result_callback:
                    self.result_callback(ack_data)
            else:
                logger.warning("Expected session_ack, got: %s", ack_data.get("type"))
        except TimeoutError:
            logger.error("Timeout waiting for session_ack")
            return

        # Now run send and receive loops concurrently
        send_task = asyncio.create_task(self._send_loop(ws))
        recv_task = asyncio.create_task(self._recv_loop(ws))

        try:
            # Wait for either to finish (e.g. on error or stop)
            done, pending = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel the other task
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Re-raise exception if any
            for task in done:
                if task.exception():
                    raise task.exception()

        except Exception as e:
            # Ensure both are cancelled on external error
            send_task.cancel()
            recv_task.cancel()
            raise e

    async def _send_loop(self, ws: WebSocketClientProtocol) -> None:
        """Loop for sending frames."""
        while self.running:
            try:
                frame_jpeg = self.camera.get_frame_jpeg(quality=self.jpeg_quality)
                if frame_jpeg is None:
                    await asyncio.sleep(0.1)
                    continue

                # Calculate send rate FPS (for monitoring)
                now = time.time()
                measured_fps = 0.0
                if self.last_frame_time is not None:
                    elapsed = now - self.last_frame_time
                    measured_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                self.last_frame_time = now

                message = {
                    "frame": base64.b64encode(frame_jpeg).decode("utf-8"),
                    "frame_id": self.frame_count,
                    "timestamp": now,
                    "fps": self.capture_fps,  # For model inference
                    "measured_fps": measured_fps,  # For network monitoring
                }

                await ws.send(json.dumps(message))
                self.frame_count += 1

                # Throttle to target FPS (e.g., 5 FPS = 0.2s sleep)
                target_interval = (
                    1.0 / self.capture_fps if self.capture_fps > 0 else 0.01
                )
                await asyncio.sleep(target_interval)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(
                    "Send loop closed: code=%s reason=%s",
                    getattr(e, "code", None),
                    getattr(e, "reason", None),
                )
                raise

    async def _recv_loop(self, ws: WebSocketClientProtocol) -> None:
        """Loop for receiving results."""
        while self.running:
            try:
                response = await ws.recv()
                message = json.loads(response)

                # Forward relevant messages to the frontend
                msg_type = message.get("type")
                logger.debug(f"Received from inference server: type={msg_type}")

                if msg_type == "result":
                    frames = message.get("frames_processed", 0)
                    inference_time = message.get("inference_time", 0.0)
                    fps = frames / inference_time if inference_time > 0 else 0.0
                    logger.info(
                        "Result job=%s status=%s frames=%s infer=%.3fs fps=%.2f",
                        message.get("job_id"),
                        message.get("status"),
                        frames,
                        inference_time,
                        fps,
                    )
                    if self.result_callback:
                        logger.debug(f"Forwarding result to callback: job_id={message.get('job_id')}")
                        self.result_callback(message)

                elif msg_type == "session_metrics":
                    # Forward session metrics to frontend for real-time display
                    logger.debug(f"Session metrics: segments={message.get('segments_processed')}, queue={message.get('queue_depth')}")
                    if self.result_callback:
                        logger.debug("Forwarding metrics to callback")
                        self.result_callback(message)

                elif msg_type == "log":
                    log_text = message.get("message")
                    log_job = message.get("job_id")
                    log_level = message.get("level", "INFO")
                    level = logging.INFO
                    if log_text:
                        logger.log(level, "Job log: [%s] %s", log_job, log_text)

                        # Check for batch submission to notify UI with pending card
                        if self.result_callback and "Submitted video_job_" in log_text:
                            match = re.search(
                                r"Submitted (video_job_\w+_batch_\d+)", log_text
                            )
                            if match:
                                job_id = match.group(1)
                                logger.debug(f"Forwarding batch_submitted to callback: job_id={job_id}")
                                self.result_callback({
                                    "type": "batch_submitted",
                                    "job_id": job_id,
                                    "timestamp": time.time(),
                                    "status": "processing",
                                })

                        # Forward all logs to frontend (not just special ones)
                        if self.result_callback:
                            logger.debug(f"Forwarding log to callback: level={log_level}")
                            self.result_callback({
                                "type": "log",
                                "message": log_text,
                                "job_id": log_job,
                                "level": log_level,
                                "timestamp": time.time(),
                            })

                elif msg_type == "session_ack":
                    # Session ack already handled at startup, but forward if received again
                    logger.info(f"Session acknowledged: session_id={message.get('session_id')}")
                    if self.result_callback:
                        logger.debug("Forwarding session_ack to callback")
                        self.result_callback(message)

                else:
                    # Unknown message type - log it
                    logger.warning(f"Unknown message type from inference server: {msg_type}")

            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(
                    "Receive loop closed: code=%s reason=%s",
                    getattr(e, "code", None),
                    getattr(e, "reason", None),
                )
                raise
            except Exception as e:
                logger.error("Error receiving message: %s", e)
                # Don't break loop on parse error, but break on connection error
                if isinstance(e, websockets.exceptions.ConnectionClosed):
                    raise e

    def stop(self) -> None:
        """Stop streaming."""
        self.running = False
        self.connection_state = "disconnected"

    def get_fps(self) -> float:
        """Calculate current FPS."""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

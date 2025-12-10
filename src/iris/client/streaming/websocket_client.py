"""Client to allow streaming of the CameraCapture device"""

import asyncio
import base64
import json
import logging
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
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,  # Send ping every 20s (was: None)
                    ping_timeout=30,  # Timeout after 10s (was: None)
                    close_timeout=30.0,
                    max_queue=None,
                ) as ws:
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
        """Main streaming loop once connected."""
        # Run send and receive loops concurrently
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
                await asyncio.sleep(0.01)

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

                # Only store result-type messages (not log messages)
                if message.get("type") == "result":
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
                        self.result_callback(message)
                elif message.get("type") == "log":
                    log_text = message.get("message")
                    log_job = message.get("job_id")
                    level = logging.INFO
                    if log_text:
                        logger.log(level, "Job log: [%s] %s", log_job, log_text)
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

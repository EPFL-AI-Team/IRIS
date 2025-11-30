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
        self.result_callback = result_callback

    async def stream(self) -> None:
        """Connect and stream frames with auto-reconnect."""
        self.running = True
        retry_delay = 1.0
        max_delay = 30.0

        attempt_count = 0
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Connected to %s", self.ws_url)
                    retry_delay = 1.0  # Reset on successful connection
                    attempt_count = 0
                    await self._stream_loop(ws)

            except websockets.exceptions.InvalidStatusCode as e:
                logger.error("Server returned invalid status code: %s", e)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
                attempt_count += 1
            except websockets.exceptions.WebSocketException as e:
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
                attempt_count += 1
                logger.error(
                    "Connection refused. Is the server running at %s? (attempt %d)",
                    self.ws_url,
                    attempt_count,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
            except Exception as e:
                logger.error("Unexpected error: %s", e, exc_info=True)
                break

    async def _stream_loop(self, ws: WebSocketClientProtocol) -> None:
        """Main streaming loop once connected."""
        while self.running:
            try:
                frame_jpeg = self.camera.get_frame_jpeg(quality=self.jpeg_quality)
                if frame_jpeg is None:
                    await asyncio.sleep(0.1)
                    continue

                # Calculate FPS
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0.0

                message = {
                    "timestamp": time.time(),
                    "frame_id": self.frame_count,
                    "fps": fps,
                    "frame": base64.b64encode(frame_jpeg).decode("utf-8"),
                }

                await ws.send(json.dumps(message))
                self.frame_count += 1

                # Try to receive result (non-blocking)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    result = json.loads(response)
                    logger.debug("Received result: %s", result)
                    if self.result_callback:
                        self.result_callback(result)
                except TimeoutError:
                    pass  # No result yet

                await asyncio.sleep(0.01)

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server")
                break

    def stop(self) -> None:
        """Stop streaming."""
        self.running = False

    def get_fps(self) -> float:
        """Calculate current FPS."""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

"""Client to allow streaming of the CameraCapture device"""

import asyncio
import base64
import json
import time

import websockets

from iris.client.capture.camera import CameraCapture


class StreamingClient:
    """WebSocket client for streaming video frames"""

    def __init__(self, ws_url: str, camera: CameraCapture, jpeg_quality: int = 80):
        self.ws_url = ws_url
        self.camera = camera
        self.jpeg_quality = jpeg_quality
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()

    async def stream(self) -> None:
        """Connect and stream frames with auto-reconnect."""
        retry_delay = 1.0
        max_delay = 30.0
        
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    print(f"Connected to {self.ws_url}")
                    retry_delay = 1.0  # Reset on successful connection
                    await self._stream_loop(ws)
                    
            except websockets.exceptions.WebSocketException as e:
                print(f"Connection failed: {e}. Retrying in {retry_delay:.1f}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

    async def _stream_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Main streaming loop once connected."""
        while self.running:
            try:
                frame_jpeg = self.camera.get_frame_jpeg(quality=80)
                if frame_jpeg is None:
                    await asyncio.sleep(0.1)
                    continue

                message = {
                    "timestamp": time.time(),
                    "frame_id": self.frame_count,
                    "frame": base64.b64encode(frame_jpeg).decode("utf-8"),
                }

                await ws.send(json.dumps(message))
                self.frame_count += 1
                
                # Try to receive result (non-blocking)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    result = json.loads(response)
                    print(f"Result: {result.get('result', 'N/A')}")
                except TimeoutError:
                    pass  # No result yet
                
                await asyncio.sleep(0.01)
                
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server")
                break

    def stop(self) -> None:
        """Stop streaming."""
        self.running = False

    def get_fps(self) -> float:
        """Calculate current FPS."""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0

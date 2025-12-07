"""
Configuration of the state of the whole app, for dependency injection (avoids global variables)
"""

import asyncio
import subprocess

from iris.client.capture.camera import CameraCapture
from iris.client.config import ClientConfig
from iris.client.streaming.websocket_client import StreamingClient


class AppState:
    """Application state with dependency injection."""

    def __init__(self):
        self.config = ClientConfig()
        self.camera: CameraCapture | None = None
        self.streaming_client: StreamingClient | None = None
        self.streaming_task: asyncio.Task | None = None
        self.latest_result: dict | None = None
        self.tunnel_process: subprocess.Popen | None = None


# Singleton instance
client_state = AppState()


def get_app_state() -> AppState:
    """Dependency injection for app state."""
    return client_state

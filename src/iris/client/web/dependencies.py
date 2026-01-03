"""
Configuration of the state of the whole app, for dependency injection (avoids global variables)
"""

import asyncio
import uuid

from iris.client.capture.camera import CameraCapture
from iris.client.capture.video_file import VideoFileCapture
from iris.client.config import ClientConfig
from iris.client.streaming.websocket_client import StreamingClient


class AppState:
    """Application state with dependency injection."""

    def __init__(self):
        self.config = ClientConfig()
        self.camera: CameraCapture | None = None
        self.streaming_client: StreamingClient | None = None
        self.streaming_task: asyncio.Task | None = None
        self.results_history: list[dict] = []
        self.max_results_history: int = 100  # Keep last 100 results

        # Persistent session ID - created on startup, reset only by user action
        # This groups all logs/results for persistence across inference runs
        self.session_id: str = f"sess_{uuid.uuid4().hex[:8]}"

        # Session configuration (segment settings)
        self.session_config: dict = {}

        # Streaming state
        self.is_streaming: bool = False

        # Inference server status
        self.inference_server_alive: bool = False

        # Session management (ephemeral - cleared on stop)
        self.current_session: dict | None = None

        # Analysis-specific state
        self.analysis_video_capture: VideoFileCapture | None = None
        self.analysis_streaming_client: StreamingClient | None = None
        self.analysis_streaming_task: asyncio.Task | None = None
        self.active_analysis_job: dict | None = None
        self.analysis_annotations: list[dict] = []
        self.analysis_results: list[dict] = []

    def reset_session(self) -> str:
        """Reset session - generate new ID and clear history."""
        self.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self.results_history.clear()
        self.session_config = {}
        return self.session_id


# Singleton instance
client_state = AppState()


def get_app_state() -> AppState:
    """Dependency injection for app state."""
    return client_state

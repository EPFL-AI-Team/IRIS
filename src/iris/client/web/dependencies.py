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

        # Persistent session IDs - fixed for local user to ensure persistence across refreshes
        # We separate Live and Analysis contexts so one doesn't wipe the other
        self.live_session_id: str = "sess_live_default"
        self.analysis_session_id: str = "sess_analysis_default"

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

    def clear_live_session(self) -> None:
        """Clear live session history."""
        self.results_history.clear()
        self.session_config = {}

    def clear_analysis_session(self) -> None:
        """Clear analysis session state."""
        self.analysis_results.clear()
        self.active_analysis_job = None


# Singleton instance
client_state = AppState()


def get_app_state() -> AppState:
    """Dependency injection for app state."""
    return client_state

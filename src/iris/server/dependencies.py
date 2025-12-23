"""Server-side state management with dependency injection."""

from typing import TYPE_CHECKING

from iris.server.inference.executor import InferenceExecutor
from iris.server.metrics import MetricsCollector

if TYPE_CHECKING:
    from iris.server.jobs.manager import JobManager


class ServerState:
    """Server application state."""

    def __init__(self):
        self.model = None  # DEPRECATED: Per-worker model loading - kept for backward compat
        self.processor = None  # DEPRECATED: Per-worker model loading - kept for backward compat
        self.queue: InferenceExecutor | None = None
        self.model_loaded = False  # Indicates workers are ready (not that state.model exists)
        self.metrics: MetricsCollector | None = None
        self.job_manager: JobManager | None = None
        self.shutting_down: bool = False


# Singleton
_server_state = ServerState()


def get_server_state() -> ServerState:
    """Get server state for dependency injection."""
    return _server_state

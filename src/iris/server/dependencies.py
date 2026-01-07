"""Server-side state management with dependency injection."""

from typing import TYPE_CHECKING

from iris.server.inference.executor import InferenceExecutor
from iris.server.metrics import MetricsCollector

if TYPE_CHECKING:
    from iris.server.jobs.manager import JobManager
    from iris.server.lifecycle import LifecycleHandler
    from iris.server.logging_handler import WebSocketLogHandler


class ServerState:
    """Server application state."""

    def __init__(self):
        self.queue: InferenceExecutor | None = None
        self.model_loaded = (
            False  # Indicates workers are ready
        )
        self.metrics: MetricsCollector | None = None
        self.job_manager: JobManager | None = None
        self.shutting_down: bool = False
        self.log_handler: WebSocketLogHandler | None = None
        self.lifecycle: LifecycleHandler | None = None


# Singleton
_server_state = ServerState()


def get_server_state() -> ServerState:
    """Get server state for dependency injection."""
    return _server_state

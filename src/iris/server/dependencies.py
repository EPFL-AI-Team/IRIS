"""Server-side state management with dependency injection."""

from iris.server.metrics import MetricsCollector
from iris.vlm.inference.queue.queue import InferenceQueue


class ServerState:
    """Server application state."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.queue: InferenceQueue | None = None
        self.model_loaded = False
        self.metrics: MetricsCollector | None = None


# Singleton
_server_state = ServerState()


def get_server_state() -> ServerState:
    """Get server state for dependency injection."""
    return _server_state

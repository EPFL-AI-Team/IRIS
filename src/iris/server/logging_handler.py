"""WebSocket log streaming handler for IRIS server."""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import WebSocket


class WebSocketLogHandler(logging.Handler):
    """Custom logging handler that broadcasts logs to WebSocket clients."""

    def __init__(self, min_level: str = "INFO"):
        """Initialize the handler with a minimum log level."""
        super().__init__()
        self.connections: list[WebSocket] = []
        self.setLevel(getattr(logging, min_level.upper()))

        # Set formatter for structured log messages
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to all connected WebSocket clients."""
        try:
            log_entry = self.format(record)
            # Broadcast to all connected clients (non-blocking)
            for ws in self.connections[:]:  # Copy list to avoid modification during iteration
                try:
                    # Create task to send without awaiting
                    asyncio.create_task(ws.send_text(log_entry))
                except Exception as e:
                    # Remove disconnected clients
                    if ws in self.connections:
                        self.connections.remove(ws)
        except Exception:
            # Don't let logging errors crash the application
            self.handleError(record)

    def add_connection(self, websocket: WebSocket) -> None:
        """Add a new WebSocket connection to receive logs."""
        if websocket not in self.connections:
            self.connections.append(websocket)

    def remove_connection(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.connections:
            self.connections.remove(websocket)

    def get_connection_count(self) -> int:
        """Return the number of active log streaming connections."""
        return len(self.connections)

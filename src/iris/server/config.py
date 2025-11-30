from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Server inference configuration."""

    model_key: str = Field(default="smolvlm2")
    max_queue_size: int = Field(default=10)
    num_workers: int = Field(default=1)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)
    graceful_shutdown_timeout: float = Field(
        default=30.0, description="Seconds to wait for in-flight jobs before force shutdown"
    )
    enable_log_streaming: bool = Field(
        default=True, description="Stream server logs to clients via WebSocket"
    )
    log_streaming_min_level: str = Field(
        default="INFO", description="Minimum log level to stream (DEBUG, INFO, WARNING, ERROR)"
    )
    enable_metrics: bool = Field(
        default=True, description="Collect and persist metrics"
    )

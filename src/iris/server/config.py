from pydantic import BaseModel, Field

from iris.config import MemoryBufferConfig, _yaml_config


class ServerConfig(BaseModel):
    """Server inference configuration."""

    model_key: str = Field(default=_yaml_config.get("server", {}).get("model_key", "smolvlm2"))
    # VLM configuration (hybrid system)
    vlm_config: str = Field(
        default=_yaml_config.get("server", {}).get("vlm_config", "serve"),
        description="VLM config name (references configs/vlm/{name}.yaml)"
    )
    vlm_hardware: str | None = Field(
        default=_yaml_config.get("server", {}).get("vlm_hardware"),
        description="Hardware profile override (mac_m3, v100, a100, h100, cpu)"
    )
    max_queue_size: int = Field(default=_yaml_config.get("server", {}).get("max_queue_size", 10))
    num_workers: int = Field(default=_yaml_config.get("server", {}).get("num_workers", 1))
    host: str = Field(default=_yaml_config.get("server", {}).get("host", "0.0.0.0"))
    port: int = Field(default=_yaml_config.get("server", {}).get("port", 8001))
    graceful_shutdown_timeout: float = Field(
        default=_yaml_config.get("server", {}).get("graceful_shutdown_timeout", 30.0),
        description="Seconds to wait for in-flight jobs before force shutdown"
    )
    enable_log_streaming: bool = Field(
        default=_yaml_config.get("server", {}).get("enable_log_streaming", True),
        description="Stream server logs to clients via WebSocket"
    )
    log_streaming_min_level: str = Field(
        default=_yaml_config.get("server", {}).get("log_streaming_min_level", "INFO"),
        description="Minimum log level to stream (DEBUG, INFO, WARNING, ERROR)"
    )
    enable_metrics: bool = Field(
        default=_yaml_config.get("server", {}).get("enable_metrics", True),
        description="Collect and persist metrics"
    )

    # Memory buffer configuration
    memory_buffer: MemoryBufferConfig = Field(
        default_factory=lambda: MemoryBufferConfig(**_yaml_config.get("memory_buffer", {}))
    )

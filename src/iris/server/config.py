from pydantic import BaseModel, Field

from iris.config import _yaml_config


class ServerConfig(BaseModel):
    """Server inference configuration."""

    # Model selection (direct from config.yaml)
    model_id: str = Field(
        default=_yaml_config.get("server", {}).get("model_id", "qwen2.5-7b"),
        description="HuggingFace model ID or key from MODEL_CONFIGS"
    )
    # Optional hardware optimization
    vlm_hardware: str | None = Field(
        default=_yaml_config.get("server", {}).get("vlm_hardware"),
        description="Hardware profile: v100, mac, or null for auto-detect"
    )
    # Optional dtype override (takes precedence over hardware profile)
    model_dtype: str | None = Field(
        default=_yaml_config.get("server", {}).get("model_dtype"),
        description="Override model dtype (float16, float32, bfloat16, auto). Falls back to hardware profile if not set."
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
    jobs: dict = Field(
        default_factory=lambda: _yaml_config.get("jobs", {}),
        description="Job configurations (video, etc.)"
    )

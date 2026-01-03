from typing import Any

from pydantic import BaseModel, Field, model_validator

from iris.config import _yaml_config


class ServerConfig(BaseModel):
    """Server inference configuration."""

    # Model selection (direct from config.yaml)
    model_id: str = Field(
        default=_yaml_config.get("server", {}).get("model_id", "qwen2.5-7b"),
        description="HuggingFace model ID or path to local model checkpoint",
    )
    # Optional hardware optimization
    vlm_hardware: str | None = Field(
        default=_yaml_config.get("server", {}).get("vlm_hardware"),
        description="Hardware profile: v100, mac, or null for auto-detect",
    )
    # Optional dtype override (takes precedence over hardware profile)
    model_dtype: str | None = Field(
        default=_yaml_config.get("server", {}).get("model_dtype"),
        description="Override model dtype (float16, float32, bfloat16, auto). Falls back to hardware profile if not set.",
    )

    live_queue_threshold: int = Field(
        default=_yaml_config.get("server", {}).get("live_queue_threshold", 1),
        ge=0,
        description=(
            "When to start dropping frames in live mode. "
            "For single-user scenarios, set to same value as max_queue_size (e.g., 1-2)."
        ),
    )

    max_queue_size: int = Field(
        default=_yaml_config.get("server", {}).get("max_queue_size", 50),
        ge=1,
        description="Hard limit on queue capacity (live: 1-2, analysis: 50-100). Prevents memory exhaustion.",
    )
    num_workers: int = Field(
        default=_yaml_config.get("server", {}).get("num_workers", 1),
        ge=1,
        description="Number of parallel inference workers",
    )
    host: str = Field(
        default=_yaml_config.get("server", {}).get("host", "0.0.0.0"),
        description="Server bind address",
    )
    port: int = Field(
        default=_yaml_config.get("server", {}).get("port", 8001),
        ge=1,
        le=65535,
        description="Server port number",
    )
    graceful_shutdown_timeout: float = Field(
        default=_yaml_config.get("server", {}).get("graceful_shutdown_timeout", 30.0),
        ge=0,
        description="Seconds to wait for in-flight jobs before force shutdown",
    )
    enable_log_streaming: bool = Field(
        default=_yaml_config.get("server", {}).get("enable_log_streaming", True),
        description="Stream server logs to clients via WebSocket",
    )
    log_streaming_min_level: str = Field(
        default=_yaml_config.get("server", {}).get("log_streaming_min_level", "INFO"),
        description="Minimum log level to stream (DEBUG, INFO, WARNING, ERROR)",
    )
    enable_metrics: bool = Field(
        default=_yaml_config.get("server", {}).get("enable_metrics", True),
        description="Collect and persist metrics",
    )
    jobs: dict = Field(
        default_factory=lambda: _yaml_config.get("server", {}).get("jobs", {}),
        description="Job configurations (video, etc.)",
    )

    @model_validator(mode="after")
    def validate_queue_parameters(self) -> "ServerConfig":
        """Validate queue parameter relationships."""
        if self.live_queue_threshold > self.max_queue_size:
            raise ValueError(
                f"live_queue_threshold ({self.live_queue_threshold}) must be <= "
                f"max_queue_size ({self.max_queue_size}). "
                f"The threshold for dropping frames should not exceed the hard queue limit."
            )
        return self

    @classmethod
    def from_cli_args(cls, args: Any, yaml_config: dict[str, Any]) -> "ServerConfig":
        """Create ServerConfig from CLI arguments, using YAML config as base."""
        config_dict = yaml_config.get("server", {}).copy()

        # Override with CLI arguments (only if provided)
        if args.model_id is not None:
            config_dict["model_id"] = args.model_id
        if args.vlm_hardware is not None:
            config_dict["vlm_hardware"] = args.vlm_hardware
        if args.model_dtype is not None:
            config_dict["model_dtype"] = args.model_dtype
        if args.port is not None:
            config_dict["port"] = args.port
        if args.host is not None:
            config_dict["host"] = args.host
        if args.num_workers is not None:
            config_dict["num_workers"] = args.num_workers
        if args.max_queue_size is not None:
            config_dict["max_queue_size"] = args.max_queue_size
        if args.live_queue_threshold is not None:
            config_dict["live_queue_threshold"] = args.live_queue_threshold
        # If you want to allow jobs override via CLI, add here:
        # if args.jobs is not None:
        #     config_dict["jobs"] = args.jobs

        return cls(**config_dict)

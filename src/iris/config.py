"""Centralized configuration loader for IRIS."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def load_yaml_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config.yaml (relative to project root)

    Returns:
        Dictionary with configuration values
    """
    # Find project root (where config.yaml lives)
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent  # src/iris/config.py -> project root

    config_file = project_root / config_path

    if not config_file.exists():
        # Return empty dict if no config file (use defaults)
        return {}

    with open(config_file) as f:
        return yaml.safe_load(f) or {}


class MemoryBufferConfig(BaseModel):
    """Configuration for StreamBridge memory buffer.

    The memory buffer stores visual embeddings across frames to provide
    temporal context for video understanding. Uses round-decayed compression
    to prioritize recent frames while maintaining historical context.
    """

    enabled: bool = Field(
        default=False,
        description="Enable memory buffer (requires embedding storage implementation)",
    )
    max_buffer_tokens: int = Field(
        default=16384,
        description="Maximum tokens in buffer (Qwen2.5-VL context window)",
    )
    tokens_per_frame: int = Field(
        default=210, description="Tokens per frame after Vision-Language Merger pooling"
    )
    fps: float = Field(
        default=1.0, description="Frame sampling rate (frames per second)"
    )
    compression_strategy: str = Field(
        default="round_decayed", description="Compression strategy for memory buffer"
    )
    recency_decay_factor: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Decay factor for weighting older frames (0.0 to 1.0)",
    )

    @property
    def max_frames_in_buffer(self) -> int:
        """Derived: Maximum frames that fit in buffer."""
        return self.max_buffer_tokens // self.tokens_per_frame


# Load config on module import
_yaml_config = load_yaml_config()

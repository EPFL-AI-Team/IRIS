"""Centralized configuration loader for IRIS."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file. If None, reads from IRIS_CONFIG_FILE
                    environment variable or defaults to "config.yaml".
                    Can be absolute or relative to project root.

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If specified config file doesn't exist
    """
    # Priority: explicit path > env var > default
    if config_path is None:
        config_path = os.getenv("IRIS_CONFIG_FILE", "configs/config.yaml")

    # Find project root (where config.yaml lives)
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent  # src/iris/config.py -> project root

    # Handle absolute vs relative paths
    if Path(config_path).is_absolute():
        config_file = Path(config_path)
    else:
        config_file = project_root / config_path

    # Strict checking - fail fast if file doesn't exist
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Searched path: {config_path}\n"
            f"Set IRIS_CONFIG_FILE environment variable to specify alternate location."
        )

    with open(config_file) as f:
        return yaml.safe_load(f) or {}


# Load config on module import (fallback to empty dict if default doesn't exist)
try:
    _yaml_config = load_yaml_config()
except FileNotFoundError:
    # Fallback to empty config if default doesn't exist (use hardcoded defaults)
    _yaml_config = {}

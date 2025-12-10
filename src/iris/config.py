"""Centralized configuration loader for IRIS."""

from pathlib import Path
from typing import Any

import yaml


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


# Load config on module import
_yaml_config = load_yaml_config()

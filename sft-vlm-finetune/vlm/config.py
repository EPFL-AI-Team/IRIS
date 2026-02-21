"""
VLM configuration loading utilities.
"""

from pathlib import Path

import yaml

import logging

logger = logging.getLogger(__name__)


def load_hardware_profile(hardware: str) -> dict:
    """Load hardware optimization profile.

    Args:
        hardware: Profile name (e.g., "v100", "mac")

    Returns:
        Dict with hardware-specific settings (model, quantization, etc.)

    Example return:
        {
            "model": {"dtype": "bfloat16"},
            "quantization": {"load_in_8bit": False}
        }
    """
    # Navigate from this file to repo root (4 levels up: vlm -> iris -> src -> root)
    repo_root = Path(__file__).parent.parent.parent
    profile_path = repo_root / "configs" / "vlm" / "hardware" / f"{hardware}.yaml"

    if not profile_path.exists():
        logger.warning(f"Hardware profile not found: {hardware}, using defaults")
        return {}

    with open(profile_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded hardware profile: {hardware}")
    return config or {}


def load_config(config_name: str, hardware: str | None = None) -> dict:
    """Load config from path or name.

    Args:
        config_name: Absolute path (/scratch/iris/train.yaml) or name ("train")
        hardware: Optional hardware override ("v100", "mac", etc)

    Returns:
        Merged config dict
    """
    # Check if absolute path
    config_path = Path(config_name)
    if config_path.is_absolute() and config_path.exists():
        cfg = _load_yaml(config_path)
        logger.info(f"Loaded config from: {config_path}")
    else:
        # Resolve relative to project structure
        base_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "vlm"
            / f"{config_name}.yaml"
        )
        if not base_path.exists():
            raise FileNotFoundError(f"Config not found: {base_path}")
        cfg = _load_yaml(base_path)
        logger.info(f"Loaded config: {config_name}")

    # Apply hardware overrides
    if hardware:
        hw_path = (
            Path(__file__).parent.parent.parent
            / "configs"
            / "vlm"
            / "hardware"
            / f"{hardware}.yaml"
        )
        if hw_path.exists():
            hw_cfg = _load_yaml(hw_path)
            cfg = _merge_dicts(cfg, hw_cfg)
            logger.info(f"Applied hardware profile: {hardware}")

    return cfg


def _load_yaml(path: Path) -> dict:
    """Load YAML file"""
    with open(path) as f:
        return yaml.safe_load(f)


def _merge_dicts(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

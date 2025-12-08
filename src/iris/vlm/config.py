"""
VLM configuration loading utilities.
"""

from pathlib import Path

import yaml

from iris.utils.logging import setup_logger

logger = setup_logger(__name__)


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
    profile_path = Path(f"configs/vlm/hardware/{hardware}.yaml")

    if not profile_path.exists():
        logger.warning(f"Hardware profile not found: {hardware}, using defaults")
        return {}

    with open(profile_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded hardware profile: {hardware}")
    return config or {}


def load_config(config_name: str, hardware: str | None = None) -> dict:
    """Load config from configs/vlm/{config_name}.yaml + optional hardware override.

    Args:
        config_name: "train" or "serve" (loads configs/vlm/{config_name}.yaml)
        hardware: Optional hardware override ("mac_m3", "v100_qlora", etc)

    Returns:
        Merged config dict

    Example:
        cfg = load_config("train", hardware="v100_qlora")
        cfg = load_config("serve")  # No hardware override
    """
    base_path = Path(f"configs/vlm/{config_name}.yaml")
    cfg = _load_yaml(base_path)

    if hardware:
        hw_path = Path(f"configs/vlm/hardware/{hardware}.yaml")
        if hw_path.exists():
            hw_cfg = _load_yaml(hw_path)
            cfg = _merge_dicts(cfg, hw_cfg)

    return cfg


def _load_yaml(path: Path) -> dict:
    """Load single YAML file"""
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

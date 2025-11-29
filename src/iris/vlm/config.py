"""
CLI entry for fine-tuning VLM with config-based setup
"""

from pathlib import Path

import yaml


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

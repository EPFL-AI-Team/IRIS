from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    annotations_dir: Path
    videos_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class DatasetPaths:
    profile: DatasetProfile

    frames_dir: Path
    csv_per_video_dir: Path
    jsonl_per_video_dir: Path

    consolidated_csv: Path
    consolidated_jsonl: Path

    splits_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    default_profile: str
    default_split_name: str

    canonical_max_frames: int
    frames_per_segment: int

    # Quotas for downsampling when creating splits
    train_per_verb: int
    val_test_per_verb: int

    profiles: dict[str, DatasetProfile]


def _expand_tilde(path: Path) -> Path:
    if str(path).startswith("~"):
        return Path(str(path).replace("~", str(Path.home()), 1))
    return path


def _warn_if_not_absolute(label: str, path: Path) -> None:
    if not path.is_absolute():
        logger.warning(
            "%s is not absolute (%s). Prefer absolute paths in dataset_config.yaml.",
            label,
            path,
        )


def load_dataset_config(config_path: Path) -> DatasetConfig:
    config_path = _expand_tilde(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Invalid YAML: expected a mapping at top-level")

    default_profile = str(data.get("default_profile", "mac"))
    default_split_name = str(data.get("default_split_name", "default"))

    frames_cfg = data.get("frames", {})
    if frames_cfg is None:
        frames_cfg = {}
    if not isinstance(frames_cfg, dict):
        raise ValueError("Invalid YAML: expected 'frames' to be a mapping")

    canonical_max_frames = int(frames_cfg.get("canonical_max_frames", 16))
    frames_per_segment = int(frames_cfg.get("per_segment", 4))
    if canonical_max_frames <= 0:
        raise ValueError("Invalid frames.canonical_max_frames: must be > 0")
    if frames_per_segment <= 0:
        raise ValueError("Invalid frames.per_segment: must be > 0")
    if frames_per_segment > canonical_max_frames:
        raise ValueError(
            "Invalid frames.per_segment: cannot exceed frames.canonical_max_frames"
        )

    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise ValueError("Invalid YAML: expected 'profiles' mapping")

    profiles: dict[str, DatasetProfile] = {}
    for name, p in raw_profiles.items():
        if not isinstance(p, dict):
            raise ValueError(f"Invalid profile '{name}': expected mapping")

        annotations_dir = _expand_tilde(Path(str(p.get("annotations_dir", ""))))
        videos_dir = _expand_tilde(Path(str(p.get("videos_dir", ""))))
        output_dir = _expand_tilde(Path(str(p.get("output_dir", ""))))

        _warn_if_not_absolute(f"profiles.{name}.annotations_dir", annotations_dir)
        _warn_if_not_absolute(f"profiles.{name}.videos_dir", videos_dir)
        _warn_if_not_absolute(f"profiles.{name}.output_dir", output_dir)

        profiles[str(name)] = DatasetProfile(
            name=str(name),
            annotations_dir=annotations_dir,
            videos_dir=videos_dir,
            output_dir=output_dir,
        )

    return DatasetConfig(
        default_profile=default_profile,
        default_split_name=default_split_name,
        canonical_max_frames=canonical_max_frames,
        frames_per_segment=frames_per_segment,
        train_per_verb=int(data.get("quotas", {}).get("train_per_verb", 1000)),
        val_test_per_verb=int(data.get("quotas", {}).get("val_test_per_verb", 200)),
        profiles=profiles,
    )


def resolve_paths(
    config: DatasetConfig, *, profile_name: str | None, split_name: str | None
) -> tuple[DatasetPaths, str]:
    selected_profile = profile_name or config.default_profile
    if selected_profile not in config.profiles:
        raise KeyError(
            f"Unknown profile '{selected_profile}'. Available: {sorted(config.profiles)}"
        )

    profile = config.profiles[selected_profile]

    output_dir = profile.output_dir

    frames_dir = output_dir / "frames"
    csv_per_video_dir = output_dir / "csv_annotations"
    jsonl_per_video_dir = output_dir / "jsonl_annotations"

    consolidated_csv = output_dir / "all_annotations.csv"
    consolidated_jsonl = output_dir / "raw_data.jsonl"

    effective_split_name = split_name or config.default_split_name
    splits_dir = output_dir / "splits" / effective_split_name

    return (
        DatasetPaths(
            profile=profile,
            frames_dir=frames_dir,
            csv_per_video_dir=csv_per_video_dir,
            jsonl_per_video_dir=jsonl_per_video_dir,
            consolidated_csv=consolidated_csv,
            consolidated_jsonl=consolidated_jsonl,
            splits_dir=splits_dir,
        ),
        effective_split_name,
    )


def validate_inputs(paths: DatasetPaths) -> None:
    if not paths.profile.annotations_dir.exists():
        raise FileNotFoundError(
            f"annotations_dir does not exist: {paths.profile.annotations_dir}"
        )

    if not paths.profile.videos_dir.exists():
        raise FileNotFoundError(
            f"videos_dir does not exist: {paths.profile.videos_dir}"
        )


def ensure_output_dirs(paths: DatasetPaths) -> None:
    paths.profile.output_dir.mkdir(parents=True, exist_ok=True)
    paths.frames_dir.mkdir(parents=True, exist_ok=True)
    paths.csv_per_video_dir.mkdir(parents=True, exist_ok=True)
    paths.jsonl_per_video_dir.mkdir(parents=True, exist_ok=True)
    paths.splits_dir.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

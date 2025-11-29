"""PyTorch Dataset classes for loading video frames and text."""

from typing import Any


class EgoTimeQADataset:
    """Dataset for EgoTimeQA video frames and text."""

    def __init__(self, manifest_path: str, cfg: dict) -> None:
        """Initialize dataset with manifest path and config."""
        # Load manifest (JSON/JSONL with video paths, labels)
        pass

    def __getitem__(self, idx: int) -> Any:
        """Load and return preprocessed frame and text tensors."""
        # Load frame + text
        # Return preprocessed tensors
        pass

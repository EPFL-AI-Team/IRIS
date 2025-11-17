"""PyTorch Dataset for egocentric laboratory videos"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from .preprocessing import VideoPreprocessor
from .augmentation import VideoAugmentation


class VideoDataset(Dataset):
    """
    Dataset for egocentric laboratory videos with action annotations.

    Expected annotation format (JSON):
    {
        "video_id": "video_001",
        "video_path": "path/to/video.mp4",
        "annotations": [
            {"start_frame": 0, "end_frame": 120, "action_id": 11, "action_name": "pipetting"},
            {"start_frame": 121, "end_frame": 240, "action_id": 21, "action_name": "vortexing"},
            ...
        ]
    }
    """

    def __init__(
        self,
        annotation_file: str,
        video_dir: str,
        preprocessor: VideoPreprocessor,
        augmentation: Optional[VideoAugmentation] = None,
        split: str = "train",
    ):
        """
        Args:
            annotation_file: Path to JSON annotation file
            video_dir: Directory containing videos
            preprocessor: VideoPreprocessor instance
            augmentation: VideoAugmentation instance (optional)
            split: Dataset split ('train', 'val', 'test')
        """
        self.video_dir = Path(video_dir)
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.split = split

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Create sample list (each sample is a video segment)
        self.samples = self._create_sample_list()

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _create_sample_list(self) -> List[Dict]:
        """Create list of samples from annotations"""
        samples = []

        for video_data in self.annotations:
            video_path = self.video_dir / video_data["video_path"]

            if not video_path.exists():
                print(f"Warning: Video not found: {video_path}")
                continue

            # Create a sample for each action annotation
            for annotation in video_data["annotations"]:
                sample = {
                    "video_path": str(video_path),
                    "video_id": video_data["video_id"],
                    "start_frame": annotation["start_frame"],
                    "end_frame": annotation["end_frame"],
                    "action_id": annotation["action_id"],
                    "action_name": annotation.get("action_name", "unknown"),
                }
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample.

        Returns:
            frames: [num_frames, channels, height, width]
            label: Action class ID
            metadata: Dictionary with sample info
        """
        sample = self.samples[idx]

        # Load video segment
        frames = self.preprocessor.process_video_segment(
            sample["video_path"],
            sample["start_frame"],
            sample["end_frame"],
        )

        # Apply augmentation (only during training)
        if self.augmentation and self.split == "train":
            frames = self.augmentation(frames)

        label = sample["action_id"]

        metadata = {
            "video_id": sample["video_id"],
            "video_path": sample["video_path"],
            "action_name": sample["action_name"],
            "start_frame": sample["start_frame"],
            "end_frame": sample["end_frame"],
        }

        return frames, label, metadata


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of (frames, label, metadata) tuples

    Returns:
        frames: [batch_size, num_frames, channels, height, width]
        labels: [batch_size]
        metadata: List of metadata dictionaries
    """
    frames_list, labels_list, metadata_list = zip(*batch)

    frames = torch.stack(frames_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)

    return frames, labels, list(metadata_list)


def create_dataloaders(
    train_annotation_file: str,
    val_annotation_file: str,
    test_annotation_file: str,
    video_dir: str,
    preprocessor: VideoPreprocessor = None, # added None here
    augmentation: Optional[VideoAugmentation] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders.

    Args:
        train_annotation_file: Path to train annotations
        val_annotation_file: Path to val annotations
        test_annotation_file: Path to test annotations
        video_dir: Directory containing videos
        preprocessor: VideoPreprocessor instance
        augmentation: VideoAugmentation instance (only applied to train)
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = VideoDataset(
        train_annotation_file,
        video_dir,
        preprocessor,
        augmentation=augmentation,
        split="train",
    )

    val_dataset = VideoDataset(
        val_annotation_file,
        video_dir,
        preprocessor,
        augmentation=None,  # No augmentation for validation
        split="val",
    )

    test_dataset = VideoDataset(
        test_annotation_file,
        video_dir,
        preprocessor,
        augmentation=None,  # No augmentation for testing
        split="test",
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


class SimpleVideoDataset(Dataset):
    """
    Simple dataset for video files with whole-video labels.

    Expected format (text file):
    /path/to/video1.mp4 0
    /path/to/video2.mp4 3
    /path/to/video3.mp4 1

    Each line: video_path label (space-separated)
    """

    def __init__(
        self,
        list_file: str,
        preprocessor: VideoPreprocessor,
        augmentation: Optional[VideoAugmentation] = None,
        split: str = "train",
        sampling_strategy: str = "uniform",
    ):
        """
        Args:
            list_file: Path to text file with video paths and labels
            preprocessor: VideoPreprocessor instance
            augmentation: VideoAugmentation instance (optional)
            split: Dataset split ('train', 'val', 'test')
            sampling_strategy: Frame sampling strategy ('uniform', 'random', 'dense')
        """
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.split = split
        self.sampling_strategy = sampling_strategy if split == "train" else "uniform"

        # Load video paths and labels
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 2:
                    print(f"Warning: Skipping invalid line: {line}")
                    continue

                video_path, label = parts[0], int(parts[1])

                # Check if video exists
                if not Path(video_path).exists():
                    print(f"Warning: Video not found: {video_path}")
                    continue

                self.samples.append({
                    "video_path": video_path,
                    "label": label,
                })

        print(f"Loaded {len(self.samples)} samples for {split} split from {list_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get a sample.

        Returns:
            frames: [num_frames, channels, height, width]
            label: Action class ID
            metadata: Dictionary with sample info
        """
        sample = self.samples[idx]

        # Load entire video with sampling
        frames = self.preprocessor.process_video(
            sample["video_path"],
            sampling_strategy=self.sampling_strategy,
        )

        # Apply augmentation (only during training)
        if self.augmentation and self.split == "train":
            frames = self.augmentation(frames)

        label = sample["label"]

        metadata = {
            "video_path": sample["video_path"],
            "video_id": Path(sample["video_path"]).stem,
        }

        return frames, label, metadata


def create_simple_dataloaders(
    train_list_file: str,
    val_list_file: str,
    test_list_file: str,
    preprocessor: VideoPreprocessor,
    augmentation: Optional[VideoAugmentation] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders from simple text list files.

    Args:
        train_list_file: Path to train list file (video_path label per line)
        val_list_file: Path to val list file
        test_list_file: Path to test list file
        preprocessor: VideoPreprocessor instance
        augmentation: VideoAugmentation instance (only applied to train)
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = SimpleVideoDataset(
        train_list_file,
        preprocessor,
        augmentation=augmentation,
        split="train",
        sampling_strategy="random",  # Use random sampling for training
    )

    val_dataset = SimpleVideoDataset(
        val_list_file,
        preprocessor,
        augmentation=None,  # No augmentation for validation
        split="val",
        sampling_strategy="uniform",
    )

    test_dataset = SimpleVideoDataset(
        test_list_file,
        preprocessor,
        augmentation=None,  # No augmentation for testing
        split="test",
        sampling_strategy="uniform",
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def create_sample_annotations(output_file: str, num_samples: int = 10):
    """
    Create sample annotation file for testing.

    Args:
        output_file: Path to save annotations
        num_samples: Number of sample videos to create
    """
    annotations = []

    action_names = ["pipetting", "vortexing", "centrifuging", "pouring", "labeling"]
    action_ids = [11, 21, 31, 12, 2]

    for i in range(num_samples):
        video_data = {
            "video_id": f"video_{i:03d}",
            "video_path": f"video_{i:03d}.mp4",
            "annotations": []
        }

        # Create random action segments
        current_frame = 0
        while current_frame < 1000:
            action_idx = np.random.randint(0, len(action_names))
            segment_length = np.random.randint(60, 240)

            annotation = {
                "start_frame": current_frame,
                "end_frame": min(current_frame + segment_length, 1000),
                "action_id": action_ids[action_idx],
                "action_name": action_names[action_idx],
            }

            video_data["annotations"].append(annotation)
            current_frame += segment_length

        annotations.append(video_data)

    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Created sample annotations at {output_file}")


if __name__ == "__main__":
    # Test dataset
    print("Testing VideoDataset...")

    from .preprocessing import VideoPreprocessor

    # Create sample annotations
    import tempfile
    tmpdir = tempfile.mkdtemp()
    annotation_file = Path(tmpdir) / "annotations.json"
    create_sample_annotations(str(annotation_file), num_samples=5)

    # Create preprocessor
    preprocessor = VideoPreprocessor(num_frames=16, frame_size=224)

    # Note: This test would require actual video files
    print(f"\nSample annotations created at: {annotation_file}")
    print("To fully test the dataset, provide actual video files in the video_dir")

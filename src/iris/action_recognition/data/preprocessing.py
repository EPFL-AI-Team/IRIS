"""Video preprocessing utilities"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image
import decord
from decord import VideoReader, cpu, gpu


decord.bridge.set_bridge('torch')


class VideoPreprocessor:
    """
    Video preprocessing for VideoMAE.

    Handles:
    - Frame extraction at specified FPS
    - Resizing and normalization
    - Temporal sampling strategies
    """

    def __init__(
        self,
        num_frames: int = 16,
        frame_size: int = 224,
        frame_rate: Optional[int] = None,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            num_frames: Number of frames to sample from video
            frame_size: Target frame size (will be resized to frame_size x frame_size)
            frame_rate: Target frame rate (if None, use original FPS)
            mean: Mean for normalization (ImageNet default)
            std: Std for normalization (ImageNet default)
        """
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def load_video(self, video_path: Union[str, Path]) -> VideoReader:
        """Load video using decord"""
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            return vr
        except Exception as e:
            raise ValueError(f"Error loading video {video_path}: {e}")

    def sample_frame_indices(
        self,
        total_frames: int,
        sampling_strategy: str = "uniform",
    ) -> List[int]:
        """
        Sample frame indices from video.

        Args:
            total_frames: Total number of frames in video
            sampling_strategy: 'uniform', 'random', or 'dense'

        Returns:
            indices: List of frame indices to extract
        """
        if sampling_strategy == "uniform":
            # Uniformly sample frames across the video
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        elif sampling_strategy == "random":
            # Randomly sample frames (for training)
            indices = np.sort(np.random.choice(total_frames, self.num_frames, replace=False))

        elif sampling_strategy == "dense":
            # Sample consecutive frames from a random start point
            start_idx = np.random.randint(0, max(1, total_frames - self.num_frames))
            indices = np.arange(start_idx, start_idx + self.num_frames)

        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

        return indices.tolist()

    def preprocess_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Preprocess frames: resize, normalize.

        Args:
            frames: [num_frames, height, width, channels] in range [0, 255]

        Returns:
            frames: [num_frames, channels, height, width] normalized
        """
        # Convert to float and normalize to [0, 1]
        frames = frames.float() / 255.0

        # Permute to [num_frames, channels, height, width]
        frames = frames.permute(0, 3, 1, 2)

        # Resize
        if frames.shape[2] != self.frame_size or frames.shape[3] != self.frame_size:
            frames = torch.nn.functional.interpolate(
                frames,
                size=(self.frame_size, self.frame_size),
                mode='bilinear',
                align_corners=False,
            )

        # Normalize with ImageNet stats
        frames = (frames - self.mean) / self.std

        return frames

    def process_video(
        self,
        video_path: Union[str, Path],
        sampling_strategy: str = "uniform",
    ) -> torch.Tensor:
        """
        Load and preprocess video.

        Args:
            video_path: Path to video file
            sampling_strategy: Frame sampling strategy

        Returns:
            frames: [num_frames, channels, height, width]
        """
        # Load video
        vr = self.load_video(video_path)
        total_frames = len(vr)

        if total_frames < self.num_frames:
            raise ValueError(
                f"Video has {total_frames} frames but {self.num_frames} required"
            )

        # Sample frame indices
        indices = self.sample_frame_indices(total_frames, sampling_strategy)

        # Extract frames
        frames = vr.get_batch(indices)  # [num_frames, height, width, channels]

        # Preprocess
        frames = self.preprocess_frames(frames)

        return frames

    def process_video_segment(
        self,
        video_path: Union[str, Path],
        start_frame: int,
        end_frame: int,
    ) -> torch.Tensor:
        """
        Process a specific segment of video.

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index

        Returns:
            frames: [num_frames, channels, height, width]
        """
        vr = self.load_video(video_path)

        # Calculate frame indices within the segment
        segment_length = end_frame - start_frame
        if segment_length < self.num_frames:
            # Repeat frames if segment is too short
            indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
        else:
            # Sample uniformly within the segment
            indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)

        # Extract frames
        frames = vr.get_batch(indices.tolist())

        # Preprocess
        frames = self.preprocess_frames(frames)

        return frames


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    frame_rate: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> List[Path]:
    """
    Extract frames from video and save as images.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_rate: Target frame rate (if None, use original)
        max_frames: Maximum number of frames to extract

    Returns:
        frame_paths: List of paths to extracted frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_rate is None:
        frame_rate = original_fps

    frame_interval = int(original_fps / frame_rate)
    frame_count = 0
    saved_count = 0
    frame_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()

    return frame_paths


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        info: Dictionary with video metadata
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))

    info = {
        "num_frames": len(vr),
        "fps": vr.get_avg_fps(),
        "duration": len(vr) / vr.get_avg_fps(),
        "width": vr[0].shape[1],
        "height": vr[0].shape[0],
    }

    return info


if __name__ == "__main__":
    # Test preprocessing
    print("Testing VideoPreprocessor...")

    preprocessor = VideoPreprocessor(num_frames=16, frame_size=224)

    # Create dummy video for testing
    import tempfile
    import imageio

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"

        # Create a simple test video
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(30)]
        imageio.mimsave(video_path, frames, fps=10)

        # Test video info
        info = get_video_info(video_path)
        print(f"\nVideo info:")
        for k, v in info.items():
            print(f"  {k}: {v}")

        # Test preprocessing
        processed_frames = preprocessor.process_video(video_path)
        print(f"\nProcessed frames shape: {processed_frames.shape}")
        print(f"Frames min/max: {processed_frames.min():.3f} / {processed_frames.max():.3f}")

    print("\nPreprocessing test passed!")

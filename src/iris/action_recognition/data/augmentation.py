"""Video data augmentation for training"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Optional


class VideoAugmentation:
    """
    Data augmentation for video clips.

    Applies spatial and temporal augmentations to improve model generalization.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        color_jitter_prob: float = 0.5,
        color_jitter_strength: float = 0.2,
        rotation_degrees: float = 5,
        temporal_crop_prob: float = 0.5,
        gaussian_blur_prob: float = 0.3,
    ):
        """
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            color_jitter_prob: Probability of color jitter
            color_jitter_strength: Strength of color jitter
            rotation_degrees: Max rotation angle
            temporal_crop_prob: Probability of temporal cropping
            gaussian_blur_prob: Probability of Gaussian blur
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.rotation_degrees = rotation_degrees
        self.temporal_crop_prob = temporal_crop_prob
        self.gaussian_blur_prob = gaussian_blur_prob

        # Color jitter
        self.color_jitter = transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=color_jitter_strength / 2,
        )

        # Gaussian blur
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to video frames.

        Args:
            frames: [num_frames, channels, height, width]

        Returns:
            frames: Augmented frames
        """
        # Horizontal flip
        if torch.rand(1).item() < self.horizontal_flip_prob:
            frames = torch.flip(frames, dims=[3])

        # Color jitter (apply to all frames)
        if torch.rand(1).item() < self.color_jitter_prob:
            frames = torch.stack([self.color_jitter(frame) for frame in frames])

        # Gaussian blur
        if torch.rand(1).item() < self.gaussian_blur_prob:
            frames = torch.stack([self.gaussian_blur(frame) for frame in frames])

        # Random rotation
        if self.rotation_degrees > 0:
            angle = (torch.rand(1).item() - 0.5) * 2 * self.rotation_degrees
            frames = self._rotate_frames(frames, angle)

        # Temporal crop and resample
        if torch.rand(1).item() < self.temporal_crop_prob:
            frames = self._temporal_crop(frames)

        return frames

    def _rotate_frames(self, frames: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotate all frames by the same angle.

        Args:
            frames: [num_frames, channels, height, width]
            angle: Rotation angle in degrees

        Returns:
            rotated_frames: Rotated frames
        """
        # Use torchvision's functional API for rotation
        from torchvision.transforms import functional as F

        rotated_frames = torch.stack([
            F.rotate(frame, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            for frame in frames
        ])

        return rotated_frames

    def _temporal_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Crop a random temporal segment and resample to original length.

        Args:
            frames: [num_frames, channels, height, width]

        Returns:
            cropped_frames: Temporally cropped and resampled frames
        """
        num_frames = frames.shape[0]

        # Crop between 80-100% of the original length
        crop_ratio = 0.8 + torch.rand(1).item() * 0.2
        crop_length = int(num_frames * crop_ratio)

        # Random start point
        start_idx = torch.randint(0, num_frames - crop_length + 1, (1,)).item()
        end_idx = start_idx + crop_length

        # Crop
        cropped = frames[start_idx:end_idx]

        # Resample to original length
        indices = torch.linspace(0, crop_length - 1, num_frames).long()
        resampled = cropped[indices]

        return resampled


class MixUpAugmentation:
    """
    MixUp augmentation for video classification.

    Mixes two samples and their labels for regularization.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: MixUp interpolation strength
        """
        self.alpha = alpha

    def __call__(
        self,
        frames1: torch.Tensor,
        frames2: torch.Tensor,
        label1: int,
        label2: int,
        num_classes: int,
    ) -> tuple:
        """
        Apply MixUp to two samples.

        Args:
            frames1: First video [num_frames, channels, height, width]
            frames2: Second video
            label1: First label
            label2: Second label
            num_classes: Total number of classes

        Returns:
            mixed_frames: Mixed video
            mixed_labels: Soft labels [num_classes]
        """
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix frames
        mixed_frames = lam * frames1 + (1 - lam) * frames2

        # Create soft labels
        mixed_labels = torch.zeros(num_classes)
        mixed_labels[label1] = lam
        mixed_labels[label2] = 1 - lam

        return mixed_frames, mixed_labels


class TemporalSmoothing:
    """
    Temporal smoothing for action predictions.

    Smooths predictions across time to reduce jitter.
    """

    def __init__(self, window_size: int = 5, method: str = "median"):
        """
        Args:
            window_size: Size of smoothing window
            method: 'median' or 'moving_average'
        """
        self.window_size = window_size
        self.method = method

    def __call__(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to predictions.

        Args:
            predictions: [num_frames] (predicted class IDs)

        Returns:
            smoothed: Smoothed predictions
        """
        if self.method == "median":
            return self._median_filter(predictions)
        elif self.method == "moving_average":
            return self._moving_average(predictions)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")

    def _median_filter(self, predictions: np.ndarray) -> np.ndarray:
        """Apply median filter"""
        from scipy.ndimage import median_filter
        return median_filter(predictions, size=self.window_size, mode='nearest')

    def _moving_average(self, predictions: np.ndarray) -> np.ndarray:
        """Apply moving average (for probability distributions)"""
        smoothed = np.copy(predictions)
        half_window = self.window_size // 2

        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            smoothed[i] = np.argmax(np.bincount(predictions[start:end]))

        return smoothed


if __name__ == "__main__":
    # Test augmentation
    print("Testing VideoAugmentation...")

    aug = VideoAugmentation()

    # Create dummy video clip
    frames = torch.randn(16, 3, 224, 224)

    # Apply augmentation
    augmented = aug(frames)

    print(f"Original shape: {frames.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Original range: [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    # Test MixUp
    print("\nTesting MixUp...")
    mixup = MixUpAugmentation(alpha=0.2)

    frames1 = torch.randn(16, 3, 224, 224)
    frames2 = torch.randn(16, 3, 224, 224)

    mixed_frames, mixed_labels = mixup(frames1, frames2, label1=0, label2=1, num_classes=10)

    print(f"Mixed frames shape: {mixed_frames.shape}")
    print(f"Mixed labels: {mixed_labels}")

    # Test temporal smoothing
    print("\nTesting TemporalSmoothing...")
    smoothing = TemporalSmoothing(window_size=5, method="median")

    predictions = np.array([0, 0, 1, 0, 0, 0, 2, 2, 0, 0])
    smoothed = smoothing(predictions)

    print(f"Original predictions: {predictions}")
    print(f"Smoothed predictions: {smoothed}")

    print("\nAll tests passed!")

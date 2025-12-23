"""Frame buffering for video inference.

This module provides the FrameBuffer class that manages frame accumulation
and batch creation for video inference jobs.
"""

from dataclasses import dataclass, field

from PIL import Image


@dataclass
class FrameBuffer:
    """Manages frame buffering for video inference.

    Accumulates frames until buffer_size is reached, then provides
    a batch for inference while keeping overlap_frames for temporal continuity.
    """

    buffer_size: int = 8
    overlap_frames: int = 4
    _frames: list[Image.Image] = field(default_factory=list)

    def add_frame(self, frame: Image.Image) -> bool:
        """Add a frame to the buffer.

        Args:
            frame: PIL Image to add (will be copied)

        Returns:
            True if buffer is now full and ready for batch creation
        """
        self._frames.append(frame.copy())
        return len(self._frames) >= self.buffer_size

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for a batch.

        Returns:
            True if buffer has >= buffer_size frames
        """
        return len(self._frames) >= self.buffer_size

    def get_batch(self) -> list[Image.Image]:
        """Get the current frames for inference.

        Returns:
            Copy of the current frame buffer
        """
        return self._frames.copy()

    def slide_window(self) -> None:
        """Keep last N frames for temporal overlap.

        Should be called after get_batch() to prepare for the next batch.
        """
        self._frames = self._frames[-self.overlap_frames:]

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        self._frames.clear()

    @property
    def frames(self) -> list[Image.Image]:
        """Get current frames (read-only access)."""
        return self._frames

    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return len(self._frames)

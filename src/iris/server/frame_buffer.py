"""Frame buffering for video inference.

This module provides the FrameBuffer class that manages frame accumulation
and batch creation for video inference jobs.
"""

from dataclasses import dataclass, field

from PIL import Image


import logging
from collections import deque
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class FrameBuffer:
    """Thread-safe frame buffer with sliding window logic.

    Stores raw JPEG bytes to minimize memory footprint. Decoding happens
    only when a batch is processed by a worker.
    """

    def __init__(self, buffer_size: int = 8, overlap_frames: int = 4):
        self.buffer_size = buffer_size
        self.overlap_frames = overlap_frames
        # Store raw bytes
        self.buffer: deque[bytes] = deque(maxlen=buffer_size)

    def add_frame(self, frame_bytes: bytes) -> bool:
        """Add a frame (JPEG bytes) to the buffer.

        Returns:
            bool: True if buffer is full and ready for processing.
        """
        self.buffer.append(frame_bytes)
        return len(self.buffer) >= self.buffer_size

    def get_batch(self) -> list[bytes]:
        """Get current batch of frames as bytes.

        Returns:
            list[bytes]: List of JPEG byte strings.
        """
        return list(self.buffer)

    def slide_window(self) -> None:
        """Slide the window by removing old frames based on overlap."""
        # Calculate how many frames to keep (overlap)
        frames_to_keep = self.overlap_frames
        
        # Calculate how many to remove (stride)
        stride = len(self.buffer) - frames_to_keep
        
        # Remove old frames
        # If overlap is 0 or negative, clear everything
        if frames_to_keep <= 0:
            self.buffer.clear()
        else:
            # Remove 'stride' number of frames from the left
            for _ in range(max(0, stride)):
                if self.buffer:
                    self.buffer.popleft()

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    @property
    def current_size(self) -> int:
        return len(self.buffer)

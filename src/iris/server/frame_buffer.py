from collections import deque


class FrameBuffer:
    def __init__(self, buffer_size: int = 8, overlap_frames: int = 4):
        self.buffer_size = buffer_size
        self.overlap_frames = overlap_frames
        self.buffer: deque[bytes] = deque(maxlen=buffer_size)

    def add_frame(self, frame_bytes: bytes) -> bool:
        """Add a frame and return True if buffer is ready for processing."""
        self.buffer.append(frame_bytes)
        return self.is_ready()

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.buffer_size

    def get_batch(self) -> list[bytes]:
        return list(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def slide_window(self) -> None:
        stride = self.buffer_size - self.overlap_frames
        if stride <= 0:
            self.buffer.clear()
            return

        for _ in range(stride):
            if self.buffer:
                self.buffer.popleft()

    def __len__(self) -> int:
        return len(self.buffer)

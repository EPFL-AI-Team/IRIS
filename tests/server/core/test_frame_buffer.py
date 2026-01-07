"""Tests for FrameBuffer class.

Stores raw bytes instead of PIL Images.
"""

from iris.server.frame_buffer import FrameBuffer


class TestFrameBufferAddFrame:
    """Tests for add_frame method."""

    def test_add_frame_returns_false_until_full(self) -> None:
        """Test that add_frame returns False until buffer is full."""
        buffer = FrameBuffer(buffer_size=4)
        frame = b"fake_image_data"

        assert buffer.add_frame(frame) is False
        assert buffer.add_frame(frame) is False
        assert buffer.add_frame(frame) is False
        # Fourth frame should return True
        assert buffer.add_frame(frame) is True

    def test_add_frame_returns_true_when_full(self) -> None:
        """Test that add_frame returns True when reaching buffer_size."""
        buffer = FrameBuffer(buffer_size=2)
        frame = b"fake_image_data"

        buffer.add_frame(frame)
        result = buffer.add_frame(frame)

        assert result is True


class TestFrameBufferIsReady:
    """Tests for is_ready method."""

    def test_is_ready_false_when_empty(self) -> None:
        """Test is_ready returns False for empty buffer."""
        buffer = FrameBuffer(buffer_size=4)
        assert buffer.is_ready() is False

    def test_is_ready_false_when_partial(self) -> None:
        """Test is_ready returns False when buffer is partial."""
        buffer = FrameBuffer(buffer_size=4)
        frame = b"fake_image_data"
        buffer.add_frame(frame)
        buffer.add_frame(frame)

        assert buffer.is_ready() is False

    def test_is_ready_true_when_full(self) -> None:
        """Test is_ready returns True when buffer reaches size."""
        buffer = FrameBuffer(buffer_size=3)
        frame = b"fake_image_data"
        buffer.add_frame(frame)
        buffer.add_frame(frame)
        buffer.add_frame(frame)

        assert buffer.is_ready() is True

    def test_is_ready_true_when_overfull(self) -> None:
        """Test is_ready returns True when buffer exceeds size."""
        buffer = FrameBuffer(buffer_size=2)
        frame = b"fake_image_data"
        buffer.add_frame(frame)
        buffer.add_frame(frame)
        buffer.add_frame(frame)  # One extra

        assert buffer.is_ready() is True


class TestFrameBufferGetBatch:
    """Tests for get_batch method."""

    def test_get_batch_returns_all_frames(self) -> None:
        """Test get_batch returns all buffered frames."""
        buffer = FrameBuffer(buffer_size=3)
        frame = b"fake_image_data"
        buffer.add_frame(frame)
        buffer.add_frame(frame)
        buffer.add_frame(frame)

        batch = buffer.get_batch()

        assert len(batch) == 3
        assert all(isinstance(f, bytes) for f in batch)

    def test_get_batch_returns_copy(self) -> None:
        """Test get_batch returns a copy, not the internal list."""
        buffer = FrameBuffer(buffer_size=2)
        frame = b"fake_image_data"
        buffer.add_frame(frame)
        buffer.add_frame(frame)

        batch = buffer.get_batch()
        batch.clear()  # Modify the returned list

        # Internal buffer should be unchanged
        assert len(buffer) == 2


class TestFrameBufferSlideWindow:
    """Tests for slide_window method."""

    def test_slide_window_keeps_overlap(self) -> None:
        """Test slide_window keeps the correct number of overlap frames."""
        buffer = FrameBuffer(buffer_size=8, overlap_frames=3)
        frame = b"fake_image_data"

        # Fill buffer
        for _ in range(8):
            buffer.add_frame(frame)

        buffer.slide_window()

        assert len(buffer) == 3

    def test_slide_window_minimal_overlap(self) -> None:
        """Test slide_window with minimal overlap keeps 1 frame."""
        buffer = FrameBuffer(buffer_size=4, overlap_frames=1)
        frame = b"fake_image_data"

        for _ in range(4):
            buffer.add_frame(frame)

        buffer.slide_window()

        assert len(buffer) == 1

    def test_slide_window_zero_overlap(self) -> None:
        """Test slide_window with zero overlap clears the buffer completely."""
        buffer = FrameBuffer(buffer_size=4, overlap_frames=0)
        frame = b"fake_image_data"

        for _ in range(4):
            buffer.add_frame(frame)

        assert len(buffer) == 4
        buffer.slide_window()

        # With zero overlap, buffer should be empty
        assert len(buffer) == 0

    def test_slide_window_preserves_recent_frames(self) -> None:
        """Test that slide_window keeps the most recent frames."""
        buffer = FrameBuffer(buffer_size=4, overlap_frames=2)

        # Add frames with different bytes for identification
        frames = [b"frame1", b"frame2", b"frame3", b"frame4"]
        for f in frames:
            buffer.add_frame(f)

        buffer.slide_window()

        # Should have frame3 and frame4 (last 2)
        assert len(buffer) == 2
        current_batch = buffer.get_batch()
        assert current_batch[0] == b"frame3"
        assert current_batch[1] == b"frame4"


class TestFrameBufferClear:
    """Tests for clear method."""

    def test_clear_empties_buffer(self) -> None:
        """Test clear removes all frames."""
        buffer = FrameBuffer(buffer_size=4)
        frame = b"fake_image_data"
        buffer.add_frame(frame)
        buffer.add_frame(frame)

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.is_ready() is False


class TestFrameBufferConfiguration:
    """Tests for buffer configuration."""

    def test_custom_buffer_size(self) -> None:
        """Test buffer respects custom buffer_size."""
        buffer = FrameBuffer(buffer_size=5)
        frame = b"fake_image_data"

        for _ in range(4):
            assert buffer.add_frame(frame) is False

        assert buffer.add_frame(frame) is True

    def test_custom_overlap_frames(self) -> None:
        """Test buffer respects custom overlap_frames."""
        buffer = FrameBuffer(buffer_size=6, overlap_frames=4)
        frame = b"fake_image_data"

        for _ in range(6):
            buffer.add_frame(frame)

        buffer.slide_window()

        assert len(buffer) == 4

    def test_default_values(self) -> None:
        """Test default buffer configuration."""
        buffer = FrameBuffer()

        assert buffer.buffer_size == 8
        assert buffer.overlap_frames == 4

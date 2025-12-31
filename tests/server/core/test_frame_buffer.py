"""Tests for FrameBuffer class.

Fixtures used from tests/server/conftest.py:
- sample_image: A 100x100 red PIL Image
"""

from PIL import Image

from iris.server.frame_buffer import FrameBuffer


class TestFrameBufferAddFrame:
    """Tests for add_frame method."""

    def test_add_frame_returns_false_until_full(
        self, sample_image: Image.Image
    ) -> None:
        """Test that add_frame returns False until buffer is full."""
        buffer = FrameBuffer(buffer_size=4)

        assert buffer.add_frame(sample_image) is False
        assert buffer.add_frame(sample_image) is False
        assert buffer.add_frame(sample_image) is False
        # Fourth frame should return True
        assert buffer.add_frame(sample_image) is True

    def test_add_frame_returns_true_when_full(
        self, sample_image: Image.Image
    ) -> None:
        """Test that add_frame returns True when reaching buffer_size."""
        buffer = FrameBuffer(buffer_size=2)

        buffer.add_frame(sample_image)
        result = buffer.add_frame(sample_image)

        assert result is True

    def test_add_frame_copies_image(self, sample_image: Image.Image) -> None:
        """Test that frames are copied, not referenced."""
        buffer = FrameBuffer(buffer_size=4)
        buffer.add_frame(sample_image)

        # Modify original
        sample_image.putpixel((0, 0), (0, 0, 255))

        # Buffer frame should still be red
        assert buffer.frames[0].getpixel((0, 0)) == (255, 0, 0)


class TestFrameBufferIsReady:
    """Tests for is_ready method."""

    def test_is_ready_false_when_empty(self) -> None:
        """Test is_ready returns False for empty buffer."""
        buffer = FrameBuffer(buffer_size=4)
        assert buffer.is_ready() is False

    def test_is_ready_false_when_partial(
        self, sample_image: Image.Image
    ) -> None:
        """Test is_ready returns False when buffer is partial."""
        buffer = FrameBuffer(buffer_size=4)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)

        assert buffer.is_ready() is False

    def test_is_ready_true_when_full(self, sample_image: Image.Image) -> None:
        """Test is_ready returns True when buffer reaches size."""
        buffer = FrameBuffer(buffer_size=3)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)

        assert buffer.is_ready() is True

    def test_is_ready_true_when_overfull(
        self, sample_image: Image.Image
    ) -> None:
        """Test is_ready returns True when buffer exceeds size."""
        buffer = FrameBuffer(buffer_size=2)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)  # One extra

        assert buffer.is_ready() is True


class TestFrameBufferGetBatch:
    """Tests for get_batch method."""

    def test_get_batch_returns_all_frames(
        self, sample_image: Image.Image
    ) -> None:
        """Test get_batch returns all buffered frames."""
        buffer = FrameBuffer(buffer_size=3)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)

        batch = buffer.get_batch()

        assert len(batch) == 3

    def test_get_batch_returns_copy(self, sample_image: Image.Image) -> None:
        """Test get_batch returns a copy, not the internal list."""
        buffer = FrameBuffer(buffer_size=2)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)

        batch = buffer.get_batch()
        batch.clear()  # Modify the returned list

        # Internal buffer should be unchanged
        assert len(buffer) == 2


class TestFrameBufferSlideWindow:
    """Tests for slide_window method."""

    def test_slide_window_keeps_overlap(
        self, sample_image: Image.Image
    ) -> None:
        """Test slide_window keeps the correct number of overlap frames."""
        buffer = FrameBuffer(buffer_size=8, overlap_frames=3)

        # Fill buffer
        for _ in range(8):
            buffer.add_frame(sample_image)

        buffer.slide_window()

        assert len(buffer) == 3

    def test_slide_window_minimal_overlap(
        self, sample_image: Image.Image
    ) -> None:
        """Test slide_window with minimal overlap keeps 1 frame."""
        buffer = FrameBuffer(buffer_size=4, overlap_frames=1)

        for _ in range(4):
            buffer.add_frame(sample_image)

        buffer.slide_window()

        assert len(buffer) == 1

    def test_slide_window_zero_overlap(
        self, sample_image: Image.Image
    ) -> None:
        """Test slide_window with zero overlap clears the buffer completely."""
        buffer = FrameBuffer(buffer_size=4, overlap_frames=0)

        for _ in range(4):
            buffer.add_frame(sample_image)

        assert len(buffer) == 4
        buffer.slide_window()

        # With zero overlap, buffer should be empty (not retain all frames)
        assert len(buffer) == 0

    def test_slide_window_preserves_recent_frames(self) -> None:
        """Test that slide_window keeps the most recent frames."""
        buffer = FrameBuffer(buffer_size=4, overlap_frames=2)

        # Add frames with different colors for identification
        colors = ["red", "green", "blue", "yellow"]
        for color in colors:
            img = Image.new("RGB", (10, 10), color=color)
            buffer.add_frame(img)

        buffer.slide_window()

        # Should have blue and yellow (last 2)
        assert len(buffer) == 2
        # Check colors (blue is (0, 0, 255), yellow is (255, 255, 0))
        assert buffer.frames[0].getpixel((0, 0)) == (0, 0, 255)
        assert buffer.frames[1].getpixel((0, 0)) == (255, 255, 0)


class TestFrameBufferClear:
    """Tests for clear method."""

    def test_clear_empties_buffer(self, sample_image: Image.Image) -> None:
        """Test clear removes all frames."""
        buffer = FrameBuffer(buffer_size=4)
        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)

        buffer.clear()

        assert len(buffer) == 0
        assert buffer.is_ready() is False


class TestFrameBufferLen:
    """Tests for __len__ method."""

    def test_len_returns_frame_count(self, sample_image: Image.Image) -> None:
        """Test __len__ returns correct frame count."""
        buffer = FrameBuffer(buffer_size=10)

        assert len(buffer) == 0

        buffer.add_frame(sample_image)
        assert len(buffer) == 1

        buffer.add_frame(sample_image)
        buffer.add_frame(sample_image)
        assert len(buffer) == 3


class TestFrameBufferConfiguration:
    """Tests for buffer configuration."""

    def test_custom_buffer_size(self, sample_image: Image.Image) -> None:
        """Test buffer respects custom buffer_size."""
        buffer = FrameBuffer(buffer_size=5)

        for _ in range(4):
            assert buffer.add_frame(sample_image) is False

        assert buffer.add_frame(sample_image) is True

    def test_custom_overlap_frames(self, sample_image: Image.Image) -> None:
        """Test buffer respects custom overlap_frames."""
        buffer = FrameBuffer(buffer_size=6, overlap_frames=4)

        for _ in range(6):
            buffer.add_frame(sample_image)

        buffer.slide_window()

        assert len(buffer) == 4

    def test_default_values(self) -> None:
        """Test default buffer configuration."""
        buffer = FrameBuffer()

        assert buffer.buffer_size == 8
        assert buffer.overlap_frames == 4

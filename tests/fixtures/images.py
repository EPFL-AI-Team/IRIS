"""Image fixtures for server tests."""

import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a simple test image."""
    return Image.new("RGB", (100, 100), color="red")

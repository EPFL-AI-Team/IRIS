"""Root pytest configuration.

Fixtures are organized in tests/fixtures/ and imported by
tests/client/conftest.py and tests/server/conftest.py as needed.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

"""Database fixtures for client web tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def mock_db_path(temp_db_path: Path) -> Generator[Path, None, None]:
    """Patch DB_PATH to use temporary database."""
    with patch("iris.client.web.database.DB_PATH", temp_db_path):
        yield temp_db_path


@pytest.fixture
def initialized_db(mock_db_path: Path) -> Generator[Path, None, None]:
    """Initialize database with schema."""
    from iris.client.web.database import init_db

    init_db()
    yield mock_db_path

"""Server test fixtures.

Re-exports fixtures from tests/fixtures/ for server tests.
"""

import tempfile
from pathlib import Path

import pytest

# Import fixtures to make them available to server tests
from tests.fixtures.images import sample_image

# Re-export fixtures
__all__ = [
    "sample_image",
    "temp_metrics_dir",
    "metrics_collector",
    "persisted_metrics",
]


@pytest.fixture
def temp_metrics_dir() -> Path:
    """Create a temporary directory for metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metrics_collector(temp_metrics_dir: Path):
    """Create a MetricsCollector with persistence disabled."""
    from iris.server.metrics import MetricsCollector

    return MetricsCollector(
        persist=False,
        log_dir=str(temp_metrics_dir),
        collect_gpu_metrics=False,
    )


@pytest.fixture
def persisted_metrics(temp_metrics_dir: Path):
    """Create a MetricsCollector with persistence enabled."""
    from iris.server.metrics import MetricsCollector

    return MetricsCollector(
        persist=True,
        log_dir=str(temp_metrics_dir),
        collect_gpu_metrics=False,
    )

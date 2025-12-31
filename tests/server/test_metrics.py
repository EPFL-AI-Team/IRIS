"""Tests for MetricsCollector class."""

import tempfile
from pathlib import Path

import pytest

from iris.server.metrics import MetricsCollector, JobMetrics, SystemMetrics


@pytest.fixture
def temp_metrics_dir() -> Path:
    """Create a temporary directory for metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def metrics_collector(temp_metrics_dir: Path) -> MetricsCollector:
    """Create a MetricsCollector with persistence disabled."""
    return MetricsCollector(
        persist=False,
        log_dir=str(temp_metrics_dir),
        collect_gpu_metrics=False,
    )


@pytest.fixture
def persisted_metrics(temp_metrics_dir: Path) -> MetricsCollector:
    """Create a MetricsCollector with persistence enabled."""
    return MetricsCollector(
        persist=True,
        log_dir=str(temp_metrics_dir),
        collect_gpu_metrics=False,
    )


class TestMetricsCollectorRecordJob:
    """Tests for record_job method."""

    def test_record_job_increments_total(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test record_job increments total_jobs."""
        assert metrics_collector.total_jobs == 0

        metrics_collector.record_job(
            job_id="job-1",
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=0,
        )

        assert metrics_collector.total_jobs == 1

    def test_record_job_increments_completed(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test record_job increments completed_jobs for completed status."""
        metrics_collector.record_job(
            job_id="job-1",
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=0,
        )

        assert metrics_collector.completed_jobs == 1
        assert metrics_collector.failed_jobs == 0

    def test_record_job_increments_failed(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test record_job increments failed_jobs for failed status."""
        metrics_collector.record_job(
            job_id="job-1",
            inference_time=0.0,
            total_latency=0.0,
            status="failed",
            queue_depth=0,
        )

        assert metrics_collector.completed_jobs == 0
        assert metrics_collector.failed_jobs == 1

    def test_record_job_tracks_inference_times(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test record_job stores inference times for completed jobs."""
        metrics_collector.record_job(
            job_id="job-1",
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=0,
        )
        metrics_collector.record_job(
            job_id="job-2",
            inference_time=200.0,
            total_latency=250.0,
            status="completed",
            queue_depth=0,
        )

        assert len(metrics_collector.inference_times) == 2
        assert 100.0 in metrics_collector.inference_times
        assert 200.0 in metrics_collector.inference_times

    def test_record_job_stores_job_history(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test record_job adds to job_history."""
        metrics_collector.record_job(
            job_id="job-1",
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=2,
        )

        assert len(metrics_collector.job_history) == 1
        job = metrics_collector.job_history[0]
        assert job.job_id == "job-1"
        assert job.inference_time == 100.0
        assert job.queue_depth == 2


class TestMetricsCollectorGetStats:
    """Tests for get_stats method."""

    def test_get_stats_returns_system_metrics(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test get_stats returns a dictionary with system metrics."""
        stats = metrics_collector.get_stats()

        assert "total_jobs" in stats
        assert "completed_jobs" in stats
        assert "failed_jobs" in stats
        assert "dropped_frames" in stats
        assert "avg_inference_time" in stats
        assert "avg_latency" in stats
        assert "p50_latency" in stats
        assert "p95_latency" in stats
        assert "p99_latency" in stats

    def test_get_stats_reflects_recorded_jobs(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test get_stats reflects recorded job data."""
        metrics_collector.record_job(
            job_id="job-1",
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=0,
        )
        metrics_collector.record_job(
            job_id="job-2",
            inference_time=200.0,
            total_latency=250.0,
            status="failed",
            queue_depth=0,
        )

        stats = metrics_collector.get_stats()

        assert stats["total_jobs"] == 2
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1


class TestMetricsCollectorCalculations:
    """Tests for statistical calculations."""

    def test_mean_empty_list(self, metrics_collector: MetricsCollector) -> None:
        """Test _mean returns 0 for empty list."""
        result = metrics_collector._mean([])
        assert result == 0.0

    def test_mean_single_value(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test _mean with single value."""
        result = metrics_collector._mean([42.0])
        assert result == 42.0

    def test_mean_multiple_values(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test _mean with multiple values."""
        result = metrics_collector._mean([10.0, 20.0, 30.0])
        assert result == 20.0

    def test_percentile_empty_list(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test _percentile returns 0 for empty list."""
        result = metrics_collector._percentile([], 50)
        assert result == 0.0

    def test_percentile_p50(self, metrics_collector: MetricsCollector) -> None:
        """Test _percentile p50 calculation."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = metrics_collector._percentile(values, 50)
        # Index = 5 * 0.5 = 2.5 -> 2 -> values[2] = 30.0
        assert result == 30.0

    def test_percentile_p95(self, metrics_collector: MetricsCollector) -> None:
        """Test _percentile p95 calculation."""
        values = list(range(1, 101))  # 1 to 100
        result = metrics_collector._percentile(values, 95)
        # Index = 100 * 0.95 = 95 -> values[95] = 96
        assert result == 96


class TestMetricsCollectorReset:
    """Tests for reset method."""

    def test_reset_clears_counters(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test reset clears all counters."""
        metrics_collector.record_job(
            job_id="job-1",
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=0,
        )
        metrics_collector.record_dropped_frame()

        metrics_collector.reset(clear_files=False)

        assert metrics_collector.total_jobs == 0
        assert metrics_collector.completed_jobs == 0
        assert metrics_collector.failed_jobs == 0
        assert metrics_collector.dropped_frames == 0

    def test_reset_clears_history(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test reset clears job history and timing lists."""
        for i in range(5):
            metrics_collector.record_job(
                job_id=f"job-{i}",
                inference_time=100.0,
                total_latency=150.0,
                status="completed",
                queue_depth=0,
            )

        metrics_collector.reset(clear_files=False)

        assert len(metrics_collector.inference_times) == 0
        assert len(metrics_collector.latencies) == 0
        assert len(metrics_collector.job_history) == 0

    def test_reset_returns_previous_totals(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test reset returns previous totals in result."""
        for i in range(3):
            metrics_collector.record_job(
                job_id=f"job-{i}",
                inference_time=100.0,
                total_latency=150.0,
                status="completed",
                queue_depth=0,
            )

        result = metrics_collector.reset(clear_files=False)

        assert result["previous_totals"]["total_jobs"] == 3
        assert result["previous_totals"]["completed_jobs"] == 3

    def test_reset_updates_session_id(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test reset updates session_id and returns it in result."""
        result = metrics_collector.reset(clear_files=False)

        # Session ID should be in the expected format (YYYYMMDD_HHMMSS)
        assert len(result["new_session_id"]) == 15
        assert "_" in result["new_session_id"]
        # Collector should have the new session ID
        assert metrics_collector.session_id == result["new_session_id"]


class TestMetricsCollectorRecentJobs:
    """Tests for get_recent_jobs method."""

    def test_get_recent_jobs_returns_list(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test get_recent_jobs returns list of job dicts."""
        metrics_collector.record_job(
            job_id="job-1",
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=0,
        )

        jobs = metrics_collector.get_recent_jobs()

        assert isinstance(jobs, list)
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "job-1"

    def test_get_recent_jobs_respects_limit(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test get_recent_jobs respects limit parameter."""
        for i in range(10):
            metrics_collector.record_job(
                job_id=f"job-{i}",
                inference_time=100.0,
                total_latency=150.0,
                status="completed",
                queue_depth=0,
            )

        jobs = metrics_collector.get_recent_jobs(limit=3)

        assert len(jobs) == 3
        # Should be most recent jobs
        assert jobs[0]["job_id"] == "job-7"
        assert jobs[1]["job_id"] == "job-8"
        assert jobs[2]["job_id"] == "job-9"


class TestMetricsCollectorHistoryLimits:
    """Tests for history size limits."""

    def test_inference_times_limited(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test inference_times list is limited to 1000 entries."""
        for i in range(1100):
            metrics_collector.record_job(
                job_id=f"job-{i}",
                inference_time=float(i),
                total_latency=float(i),
                status="completed",
                queue_depth=0,
            )

        assert len(metrics_collector.inference_times) == 1000
        # Should have most recent values
        assert metrics_collector.inference_times[0] == 100.0
        assert metrics_collector.inference_times[-1] == 1099.0

    def test_job_history_limited(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test job_history list is limited to 1000 entries."""
        for i in range(1100):
            metrics_collector.record_job(
                job_id=f"job-{i}",
                inference_time=float(i),
                total_latency=float(i),
                status="completed",
                queue_depth=0,
            )

        assert len(metrics_collector.job_history) == 1000


class TestMetricsCollectorDroppedFrames:
    """Tests for dropped frame tracking."""

    def test_record_dropped_frame(
        self, metrics_collector: MetricsCollector
    ) -> None:
        """Test record_dropped_frame increments counter."""
        assert metrics_collector.dropped_frames == 0

        metrics_collector.record_dropped_frame()
        assert metrics_collector.dropped_frames == 1

        metrics_collector.record_dropped_frame()
        metrics_collector.record_dropped_frame()
        assert metrics_collector.dropped_frames == 3


class TestJobMetricsDataclass:
    """Tests for JobMetrics dataclass."""

    def test_job_metrics_fields(self) -> None:
        """Test JobMetrics has expected fields."""
        job = JobMetrics(
            job_id="test-job",
            timestamp=1234567890.0,
            inference_time=100.0,
            total_latency=150.0,
            status="completed",
            queue_depth=5,
        )

        assert job.job_id == "test-job"
        assert job.timestamp == 1234567890.0
        assert job.inference_time == 100.0
        assert job.total_latency == 150.0
        assert job.status == "completed"
        assert job.queue_depth == 5


class TestSystemMetricsDataclass:
    """Tests for SystemMetrics dataclass."""

    def test_system_metrics_fields(self) -> None:
        """Test SystemMetrics has expected fields."""
        metrics = SystemMetrics(
            timestamp=1234567890.0,
            total_jobs=100,
            completed_jobs=95,
            failed_jobs=5,
            dropped_frames=10,
            avg_inference_time=150.0,
            avg_latency=200.0,
            p50_latency=180.0,
            p95_latency=250.0,
            p99_latency=300.0,
        )

        assert metrics.total_jobs == 100
        assert metrics.completed_jobs == 95
        assert metrics.failed_jobs == 5
        assert metrics.dropped_frames == 10

    def test_system_metrics_optional_gpu_fields(self) -> None:
        """Test SystemMetrics GPU fields are optional."""
        metrics = SystemMetrics(
            timestamp=1234567890.0,
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            dropped_frames=0,
            avg_inference_time=0.0,
            avg_latency=0.0,
            p50_latency=0.0,
            p95_latency=0.0,
            p99_latency=0.0,
        )

        assert metrics.gpu_utilization is None
        assert metrics.gpu_memory_used_mb is None
        assert metrics.gpu_memory_total_mb is None
        assert metrics.gpu_temperature_c is None

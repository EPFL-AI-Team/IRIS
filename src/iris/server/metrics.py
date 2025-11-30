"""Metrics collection and persistence for IRIS server."""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JobMetrics:
    """Metrics for a single inference job."""
    job_id: str
    timestamp: float
    inference_time: float
    total_latency: float
    status: str
    queue_depth: int


@dataclass
class SystemMetrics:
    """System-level metrics snapshot."""
    timestamp: float
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    dropped_frames: int
    avg_inference_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    gpu_utilization: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_temperature_c: float | None = None


class MetricsCollector:
    """Collects and persists operational metrics."""

    def __init__(
        self,
        persist: bool = True,
        log_dir: str = "logs/metrics",
        collect_gpu_metrics: bool = True,
    ):
        """Initialize the metrics collector.

        Args:
            persist: Whether to persist metrics to disk
            log_dir: Directory to store metrics logs
            collect_gpu_metrics: Whether to collect GPU metrics (requires pynvml)
        """
        self.persist = persist
        self.log_dir = Path(log_dir)
        self.collect_gpu_metrics = collect_gpu_metrics

        # Metrics storage
        self.total_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.dropped_frames = 0
        self.inference_times: list[float] = []
        self.latencies: list[float] = []
        self.job_history: list[JobMetrics] = []

        # Session info
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # GPU monitoring
        self.nvml_available = False
        if collect_gpu_metrics:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_available = True
                self.nvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
                logger.info("GPU metrics collection enabled")
            except (ImportError, Exception) as e:
                logger.warning("GPU metrics not available: %s", e)
                self.nvml_available = False

        # Create log directory and session file
        if self.persist:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = self.log_dir / f"session_{self.session_id}.jsonl"
            logger.info("Metrics will be persisted to: %s", self.metrics_file)

            # Write session header
            self._write_metrics({
                "type": "session_start",
                "session_id": self.session_id,
                "timestamp": self.session_start,
                "datetime": datetime.fromtimestamp(self.session_start).isoformat(),
            })

    def record_job(
        self,
        job_id: str,
        inference_time: float,
        total_latency: float,
        status: str,
        queue_depth: int,
    ) -> None:
        """Record metrics for a completed job."""
        self.total_jobs += 1
        if status == "completed":
            self.completed_jobs += 1
            self.inference_times.append(inference_time)
            self.latencies.append(total_latency)
        elif status == "failed":
            self.failed_jobs += 1

        # Keep only last 1000 entries to limit memory
        if len(self.inference_times) > 1000:
            self.inference_times = self.inference_times[-1000:]
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]

        # Store job metrics
        job_metrics = JobMetrics(
            job_id=job_id,
            timestamp=time.time(),
            inference_time=inference_time,
            total_latency=total_latency,
            status=status,
            queue_depth=queue_depth,
        )
        self.job_history.append(job_metrics)

        # Limit job history
        if len(self.job_history) > 1000:
            self.job_history = self.job_history[-1000:]

        # Persist job metrics
        if self.persist:
            self._write_metrics({
                "type": "job",
                **asdict(job_metrics),
            })

    def record_dropped_frame(self) -> None:
        """Record a dropped frame."""
        self.dropped_frames += 1

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        stats = SystemMetrics(
            timestamp=time.time(),
            total_jobs=self.total_jobs,
            completed_jobs=self.completed_jobs,
            failed_jobs=self.failed_jobs,
            dropped_frames=self.dropped_frames,
            avg_inference_time=self._mean(self.inference_times),
            avg_latency=self._mean(self.latencies),
            p50_latency=self._percentile(self.latencies, 50),
            p95_latency=self._percentile(self.latencies, 95),
            p99_latency=self._percentile(self.latencies, 99),
        )

        # Add GPU metrics if available
        if self.nvml_available:
            try:
                gpu_metrics = self._get_gpu_metrics()
                stats.gpu_utilization = gpu_metrics.get("utilization")
                stats.gpu_memory_used_mb = gpu_metrics.get("memory_used_mb")
                stats.gpu_memory_total_mb = gpu_metrics.get("memory_total_mb")
                stats.gpu_temperature_c = gpu_metrics.get("temperature_c")
            except Exception as e:
                logger.warning("Failed to collect GPU metrics: %s", e)

        return asdict(stats)

    def get_recent_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent job metrics."""
        return [asdict(job) for job in self.job_history[-limit:]]

    def _get_gpu_metrics(self) -> dict[str, float]:
        """Collect GPU metrics using NVML."""
        if not self.nvml_available:
            return {}

        try:
            # GPU utilization
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

            # Memory info
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)

            # Temperature
            temperature = self.nvml.nvmlDeviceGetTemperature(
                self.gpu_handle, self.nvml.NVML_TEMPERATURE_GPU
            )

            return {
                "utilization": utilization.gpu,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024),
                "temperature_c": temperature,
            }
        except Exception as e:
            logger.warning("Failed to get GPU metrics: %s", e)
            return {}

    def snapshot(self) -> None:
        """Take a snapshot of current metrics and persist."""
        if self.persist:
            stats = self.get_stats()
            self._write_metrics({
                "type": "snapshot",
                **stats,
            })

    def _write_metrics(self, data: dict[str, Any]) -> None:
        """Write metrics to JSONL file."""
        if not self.persist:
            return

        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error("Failed to write metrics: %s", e)

    def _mean(self, values: list[float]) -> float:
        """Calculate mean of values."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def close(self) -> None:
        """Close metrics collector and finalize session."""
        if self.persist:
            self._write_metrics({
                "type": "session_end",
                "session_id": self.session_id,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.session_start,
                "total_jobs": self.total_jobs,
                "completed_jobs": self.completed_jobs,
                "failed_jobs": self.failed_jobs,
                "dropped_frames": self.dropped_frames,
            })
            logger.info("Metrics session closed: %s", self.session_id)

        # Cleanup NVML
        if self.nvml_available:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

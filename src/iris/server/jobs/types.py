"""Shared types for job system - no dependencies to avoid circular imports."""

from enum import Enum


class TriggerMode(str, Enum):
    """Trigger modes for video jobs."""

    PERIODIC = "periodic"  # Automatic (buffer_size frames)
    MANUAL = "manual"      # API call to /jobs/{id}/trigger
    DISABLED = "disabled"  # Buffering only, never triggers


class JobType(str, Enum):
    """Available job types in the system."""

    SINGLE_FRAME = "single_frame"
    VIDEO = "video"

"""
Pydantic message types for WebSocket communication.

This module defines strongly-typed message schemas for:
- Frontend → Backend messages (via /ws/client)
- Backend → Frontend messages (via /ws/client)
"""

from pydantic import BaseModel
from typing import Literal, Optional, Any


# =============================================================================
# Frontend → Backend Messages
# =============================================================================


class StartMessage(BaseModel):
    """Start live inference streaming."""

    type: Literal["start"]
    config: dict[str, Any]  # {frames_per_segment, overlap_frames}


class StopMessage(BaseModel):
    """Stop live inference streaming."""

    type: Literal["stop"]


class ClearQueueMessage(BaseModel):
    """Clear the inference job queue."""

    type: Literal["clear_queue"]


class ResetSessionMessage(BaseModel):
    """Reset session - generate new session_id, clear history."""

    type: Literal["reset_session"]


class StartAnalysisMessage(BaseModel):
    """Start video file analysis."""

    type: Literal["start_analysis"]
    video_path: str
    annotation_path: Optional[str] = None
    config: dict[str, Any]


# =============================================================================
# Backend → Frontend Messages
# =============================================================================


class SessionInfoMessage(BaseModel):
    """Session information sent on connect and after reset."""

    type: Literal["session_info"] = "session_info"
    session_id: str
    config: dict[str, Any]


class PreviewFrameMessage(BaseModel):
    """Preview frame from USB camera."""

    type: Literal["preview_frame"] = "preview_frame"
    frame: str  # base64 JPEG
    timestamp: float


class ResultMessage(BaseModel):
    """Inference result from the inference server."""

    type: Literal["result"] = "result"
    job_id: str
    result: str
    frames_processed: int
    inference_time: float
    timestamp: float


class MetricsMessage(BaseModel):
    """Session metrics update during streaming."""

    type: Literal["metrics"] = "metrics"
    elapsed_seconds: float
    segments_processed: int
    segments_total: Optional[int] = None
    queue_depth: int
    processing_rate: float
    frames_received: Optional[int] = None


class ServerStatusMessage(BaseModel):
    """Inference server health status."""

    type: Literal["server_status"] = "server_status"
    alive: bool
    queue_depth: Optional[int] = None


class ErrorMessage(BaseModel):
    """Error message from backend."""

    type: Literal["error"] = "error"
    message: str


class LogMessage(BaseModel):
    """Log message for frontend display."""

    type: Literal["log"] = "log"
    message: str
    level: Literal["INFO", "WARNING", "ERROR", "DEBUG"] = "INFO"
    job_id: Optional[str] = None
    timestamp: Optional[float] = None


class AnalysisProgressMessage(BaseModel):
    """Analysis progress update."""

    type: Literal["analysis_progress"] = "analysis_progress"
    current_frame: int
    total_frames: int
    progress_percent: float
    current_chunk: Optional[int] = None
    total_chunks: Optional[int] = None
    estimated_time_remaining: Optional[float] = None


# =============================================================================
# Type Unions for Parsing
# =============================================================================

# All messages that can be received from frontend
ClientToBackendMessage = (
    StartMessage
    | StopMessage
    | ClearQueueMessage
    | ResetSessionMessage
    | StartAnalysisMessage
)

# All messages that can be sent to frontend
BackendToClientMessage = (
    SessionInfoMessage
    | PreviewFrameMessage
    | ResultMessage
    | MetricsMessage
    | ServerStatusMessage
    | ErrorMessage
    | LogMessage
    | AnalysisProgressMessage
)

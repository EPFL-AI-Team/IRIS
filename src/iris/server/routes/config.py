"""Configuration and queue management routes for IRIS Inference Server."""

import asyncio
import gc
import logging

from fastapi import APIRouter

from iris.config import _yaml_config
from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state

logger = logging.getLogger(__name__)
config = ServerConfig()
router = APIRouter()


@router.get("/api/config/defaults")
async def get_config_defaults() -> dict:
    """Return configuration defaults for frontend initialization.

    Returns:
        Dictionary with server, video, and segment configuration defaults.
    """
    return {
        "server": {
            "host": config.host,
            "port": config.port,
            "model_id": config.model_id,
        },
        "video": {
            "width": _yaml_config.get("client", {}).get("video", {}).get("width", 640),
            "height": _yaml_config.get("client", {})
            .get("video", {})
            .get("height", 480),
            "capture_fps": _yaml_config.get("client", {})
            .get("video", {})
            .get("capture_fps", 10),
            "jpeg_quality": _yaml_config.get("client", {})
            .get("video", {})
            .get("jpeg_quality", 80),
            "camera_index": _yaml_config.get("client", {})
            .get("video", {})
            .get("camera_index", 0),
        },
        "segment": {
            "segment_time": _yaml_config.get("client", {})
            .get("segment", {})
            .get("segment_time", 1.0),
            "frames_per_segment": _yaml_config.get("client", {})
            .get("segment", {})
            .get("frames_per_segment", 8),
            "overlap_frames": _yaml_config.get("client", {})
            .get("segment", {})
            .get("overlap_frames", 4),
        },
    }


@router.post("/api/queue/clear")
async def clear_queue() -> dict:
    """Clear all pending inference jobs and free GPU memory.

    This endpoint clears the inference queue and triggers garbage collection
    to free any associated GPU memory. Use this when you want to stop
    processing and start fresh without waiting for queued jobs to complete.

    Returns:
        Dictionary with cleared count and status.
    """
    state = get_server_state()

    # Clear the job queue
    cleared_count = 0
    if state.queue and state.queue.queue:
        while not state.queue.queue.empty():
            try:
                state.queue.queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break

    # Force garbage collection to free memory
    gc.collect()

    # Clear GPU cache if torch/CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except ImportError:
        pass  # torch not available

    logger.info(f"Cleared {cleared_count} jobs from queue")
    return {"cleared": cleared_count, "status": "ok"}

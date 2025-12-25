import logging
import time
from typing import Any

from fastapi import APIRouter

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state

logger = logging.getLogger(__name__)
config = ServerConfig()
router = APIRouter()

@router.get("/health")
async def health() -> dict[str, str | bool]:
    """Health check endpoint."""
    state = get_server_state()
    return {
        "status": "healthy" if state.model_loaded else "loading",
        "model_loaded": state.model_loaded,
    }


@router.get("/metrics")
async def metrics_endpoint() -> dict[str, Any]:
    """Get current metrics and statistics."""
    state = get_server_state()

    if not config.enable_metrics or state.metrics is None:
        return {
            "error": "Metrics collection is disabled",
            "enable_metrics": False,
        }

    return {
        "enable_metrics": True,
        "stats": state.metrics.get_stats(),
        "recent_jobs": state.metrics.get_recent_jobs(limit=20),
    }


@router.delete("/system/clear")
async def clear_system(
    clear_logs: bool = True,
    stop_active_jobs: bool = True,
) -> dict[str, Any]:
    """Clear server state including queue, jobs, and metrics.

    This endpoint provides a way to reset the server state without restarting.
    Useful when switching between different experiments or data sources.

    Args:
        clear_logs: Whether to delete metrics files from disk (default: True)
        stop_active_jobs: Whether to stop running jobs (default: True)

    Returns:
        Dictionary with detailed status of clearing operations
    """
    state = get_server_state()

    result = {
        "status": "success",
        "cleared": {},
        "errors": [],
        "timestamp": time.time(),
    }

    # 1. Stop all active jobs (if requested)
    if stop_active_jobs and state.job_manager:
        try:
            logger.info("Stopping all active jobs...")
            job_result = await state.job_manager.stop_all_jobs()
            result["cleared"]["active_jobs"] = job_result["stopped_count"]
            if job_result["errors"]:
                result["errors"].extend(job_result["errors"])
        except Exception as e:
            error_msg = f"Failed to stop active jobs: {e}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            result["cleared"]["active_jobs"] = "error"
    else:
        result["cleared"]["active_jobs"] = "skipped"

    # 2. Clear pending jobs from queue
    if state.queue:
        try:
            logger.info("Clearing job queue...")
            cleared_count = await state.queue.clear_queue()
            result["cleared"]["pending_jobs"] = cleared_count
        except Exception as e:
            error_msg = f"Failed to clear queue: {e}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            result["cleared"]["pending_jobs"] = "error"
    else:
        result["cleared"]["pending_jobs"] = "queue_not_initialized"

    # 3. Reset metrics (and optionally delete files)
    if state.metrics:
        try:
            logger.info(f"Resetting metrics (clear_files={clear_logs})...")
            metrics_result = state.metrics.reset(clear_files=clear_logs)
            result["cleared"]["metrics"] = {
                "previous_totals": metrics_result["previous_totals"],
                "files_deleted": metrics_result["files_deleted"],
                "new_session_id": metrics_result["new_session_id"],
            }
            if metrics_result["errors"]:
                result["errors"].extend(metrics_result["errors"])
        except Exception as e:
            error_msg = f"Failed to reset metrics: {e}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)
            result["cleared"]["metrics"] = "error"
    else:
        result["cleared"]["metrics"] = "metrics_disabled"

    # 4. Determine overall status
    if result["errors"]:
        result["status"] = "partial_success"
        logger.warning(f"System clear completed with {len(result['errors'])} errors")
    else:
        logger.info("System clear completed successfully")

    return result

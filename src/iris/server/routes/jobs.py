import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from iris.server.dependencies import get_server_state
from iris.server.jobs.config import JobConfig

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/jobs/start")
async def start_job(config: JobConfig) -> dict[str, Any]:
    """Start a new job with specified configuration.

    Args:
        config: Job configuration (validated Pydantic model)

    Returns:
        Dictionary with job_id, status, job_type, and config
    """
    state = get_server_state()

    if state.shutting_down:
        logger.warning(
            "Rejecting start_job request during shutdown: job_type=%s", config.job_type
        )
        raise HTTPException(status_code=503, detail="Server shutting down")

    try:
        job_id = await state.job_manager.start_job(config)
        return {
            "job_id": job_id,
            "status": "started",
            "job_type": config.job_type.value,
            "config": config.model_dump(),
        }
    except Exception as e:
        logger.error(f"Failed to start job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/jobs/{job_id}/stop")
async def stop_job(job_id: str) -> dict[str, Any]:
    """Stop a running job.

    Args:
        job_id: Job identifier

    Returns:
        Dictionary with job_id, status, and message
    """
    state = get_server_state()

    try:
        success = await state.job_manager.stop_job(job_id)
        if success:
            return {
                "job_id": job_id,
                "status": "stopped",
                "message": "Job stopped successfully",
            }
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a specific job.

    Args:
        job_id: Job identifier

    Returns:
        Dictionary with job status details
    """
    state = get_server_state()

    status = await state.job_manager.get_job_status(job_id)
    if status:
        return status
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@router.post("/jobs/{job_id}/trigger")
async def trigger_job(job_id: str) -> dict[str, str]:
    """Manually trigger inference for a VideoJob.

    Args:
        job_id: Job identifier

    Returns:
        Dictionary with status message

    Raises:
        HTTPException: 404 if job not found, 400 if job doesn't support manual triggering
    """
    state = get_server_state()

    # Get job from active jobs
    async with state.job_manager.lock:
        job = state.job_manager.active_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not hasattr(job, "trigger_inference"):
        raise HTTPException(
            status_code=400, detail="Job does not support manual triggering"
        )

    await job.trigger_inference()
    return {"status": "ok", "message": "Inference triggered"}


@router.get("/jobs/active")
async def list_active_jobs() -> dict[str, Any]:
    """List all active jobs.

    Returns:
        Dictionary with list of active jobs
    """
    state = get_server_state()

    jobs = await state.job_manager.list_active_jobs()
    return {"active_jobs": jobs}

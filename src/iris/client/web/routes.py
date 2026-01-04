"""API routes for the IRIS client web interface."""

import asyncio
import base64
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import websockets
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from iris.client.capture.camera import CameraCapture
from iris.client.capture.video_file import VideoFileCapture
from iris.client.config import ServerConfig
from iris.client.streaming.websocket_client import StreamingClient
from iris.client.web.dependencies import get_app_state

logger = logging.getLogger(__name__)
api_router = APIRouter(prefix="/api")  # HTTP endpoints
ws_router = APIRouter(prefix="/ws")  # WebSocket endpoints


class StartRequest(BaseModel):
    frames_per_segment: int = 8
    overlap_frames: int = 4


class SessionConfig(BaseModel):
    frames_per_segment: int = 8
    overlap_frames: int = 4
    # camera_mode removed - server-only camera selection


# Global store for active sessions (Simple in-memory for demo)
# Map session_id -> {"config": dict, "created_at": float}
session_store: dict[str, dict] = {}

# Global store for Gemini API key (in-memory, optional alternative to env vars)
gemini_api_key_store: str | None = None


@api_router.post("/session/init")
async def initialize_session(config: SessionConfig) -> dict[str, Any]:
    """Initialize a session with configuration before streaming starts."""
    session_id = str(uuid.uuid4())

    # Store config
    session_store[session_id] = {
        "config": config.model_dump(),
        "created_at": time.time(),
    }

    # If using SERVER camera, we can start the process here immediately (optional, or kept in /start)
    # For now, we'll keep /start for server camera logic to minimize disruption,
    # but strictly speaking, this is where we establish the intent.

    logger.info(f"Initialized session {session_id} with config {config}")

    return {"session_id": session_id, "status": "ready"}


@api_router.get("/status")
async def get_status() -> dict[str, Any]:
    """Get current status."""
    state = get_app_state()

    # Determine streaming server connection status
    if state.streaming_client:
        streaming_server_status = state.streaming_client.connection_state
    else:
        streaming_server_status = "disconnected"

    return {
        "camera_active": state.camera is not None and state.camera.cap.isOpened(),
        "streaming_active": state.streaming_client is not None
        and state.streaming_client.running,
        "streaming_server_status": streaming_server_status,
        "config": state.config.model_dump(),
        "fps": state.streaming_client.get_fps() if state.streaming_client else 0.0,
    }


@api_router.post("/config")
async def update_config(new_config: ServerConfig) -> dict[str, Any]:
    """Update server configuration."""
    state = get_app_state()
    state.config.server = new_config
    return {"status": "ok", "config": state.config.server.model_dump()}


@api_router.get("/config/gemini-key")
async def get_gemini_key() -> dict[str, Any]:
    """Get whether Gemini API key is configured (without exposing the actual key)."""
    global gemini_api_key_store
    import os

    has_env_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    has_stored_key = bool(gemini_api_key_store)

    return {
        "configured": has_env_key or has_stored_key,
        "source": "environment" if has_env_key else ("stored" if has_stored_key else "none")
    }


@api_router.post("/config/gemini-key")
async def set_gemini_key(request: dict[str, Any]) -> dict[str, Any]:
    """Set Gemini API key for this session."""
    global gemini_api_key_store

    api_key = request.get("api_key", "").strip()
    if not api_key:
        gemini_api_key_store = None
        return {"status": "ok", "message": "API key cleared"}

    gemini_api_key_store = api_key
    return {"status": "ok", "message": "API key stored"}


@api_router.get("/config/defaults")
async def get_config_defaults() -> dict[str, Any]:
    """Get default configuration values from config.yaml.

    Returns server config, video config, and segment config defaults.
    Used by frontend to initialize settings on load.
    """
    state = get_app_state()
    return {
        "server": state.config.server.model_dump(),
        "video": state.config.video.model_dump(),
        "segment": state.config.segment.model_dump(),
    }


@api_router.post("/start")
async def start_streaming(request: StartRequest | None = None) -> dict[str, Any]:
    """Start camera and streaming."""
    state = get_app_state()
    # Use defaults if no body provided
    if request is None:
        request = StartRequest()

    # Start camera
    if state.camera is None:
        state.camera = CameraCapture(
            camera_index=state.config.video.camera_index,
            width=state.config.video.width,
            height=state.config.video.height,
            fps=state.config.video.capture_fps,
        )
        if not state.camera.start():
            return {"status": "error", "message": "Failed to start camera"}

    # Start streaming
    def store_result(result: dict[str, Any]) -> None:
        """Callback to store inference results."""
        state.results_history.append(result)
        if len(state.results_history) > state.max_results_history:
            state.results_history.pop(0)

    state.streaming_client = StreamingClient(
        state.config.server.ws_url,
        state.camera,
        result_callback=store_result,
        session_config={
            "frames_per_segment": request.frames_per_segment,
            "overlap_frames": request.overlap_frames,
        },
    )

    task = asyncio.create_task(state.streaming_client.stream())
    state.streaming_task = task

    return {"status": "ok", "message": "Streaming started"}


@api_router.post("/stop")
async def stop_streaming() -> dict[str, str]:
    """Stop streaming and camera."""
    state = get_app_state()

    if state.streaming_client:
        state.streaming_client.stop()
        state.streaming_client = None

    if state.camera:
        state.camera.stop()
        state.camera = None

    # Clear session state
    state.current_session = None
    state.session_id = None

    return {"status": "ok", "message": "Stopped"}


@api_router.get("/cameras")
async def list_cameras() -> dict[str, Any]:
    """List available camera devices on server."""
    import cv2

    cameras = []
    for i in range(10):
        # Suppress OpenCV warnings temporarily
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({
                "index": i,
                "name": f"Camera {i}",
                "resolution": f"{width}x{height}",
            })
            cap.release()
        else:
            # Immediately release if it fails
            cap.release()
    return {"cameras": cameras}


@api_router.post("/camera/select")
async def select_camera(request: dict[str, int]) -> dict[str, Any]:
    """Switch server to different camera device."""
    state = get_app_state()
    camera_index = request["camera_index"]

    # Stop current camera if running
    if state.camera:
        state.camera.stop()
        state.camera = None

    # Update config and start new camera
    state.config.video.camera_index = camera_index
    state.camera = CameraCapture(
        camera_index=camera_index,
        width=state.config.video.width,
        height=state.config.video.height,
        fps=state.config.video.capture_fps,
    )

    if not state.camera.start():
        state.camera = None
        return {"status": "error", "message": f"Failed to open camera {camera_index}"}

    return {"status": "ok", "camera_index": camera_index}


@api_router.get("/results/history")
async def get_results_history() -> dict[str, Any]:
    """Get all stored inference results."""
    state = get_app_state()
    return {
        "count": len(state.results_history),
        "results": state.results_history,
    }


@api_router.post("/results/clear")
async def clear_results_history() -> dict[str, str]:
    """Clear stored inference results history."""
    state = get_app_state()
    state.results_history.clear()
    return {"status": "ok", "message": "Results history cleared"}


# ============================================================================
# Analysis & Benchmark Endpoints
# ============================================================================


@api_router.get("/datasets")
async def get_datasets() -> dict[str, Any]:
    """List available video and annotation files in static/videos/ directory."""
    videos_dir = Path(__file__).parent / "static" / "videos"

    # Create directory if it doesn't exist
    videos_dir.mkdir(parents=True, exist_ok=True)

    videos = []
    for video_path in videos_dir.glob("*.mp4"):
        try:
            # Use cv2.VideoCapture to read metadata
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration_sec = frame_count / fps if fps > 0 else 0.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                videos.append({
                    "filename": video_path.name,
                    "path": f"/static/videos/{video_path.name}",
                    "size_mb": video_path.stat().st_size / (1024 * 1024),
                    "duration_sec": duration_sec,
                    "resolution": f"{width}x{height}",
                    "fps": fps,
                    "frame_count": frame_count,
                })
                cap.release()
        except Exception as e:
            logger.warning(f"Failed to read video metadata for {video_path.name}: {e}")

    annotations = []
    for jsonl_path in videos_dir.glob("*.jsonl"):
        try:
            line_count = sum(1 for _ in jsonl_path.open())
            annotations.append({
                "filename": jsonl_path.name,
                "path": f"/static/videos/{jsonl_path.name}",
                "size_kb": jsonl_path.stat().st_size / 1024,
                "line_count": line_count,
            })
        except Exception as e:
            logger.warning(f"Failed to read annotation file {jsonl_path.name}: {e}")

    return {"videos": videos, "annotations": annotations}


@api_router.get("/videos/{filename}")
async def get_video(filename: str) -> FileResponse:
    """Serve video files with proper headers for HTML5 video playback.

    Provides Accept-Ranges header for seeking support in video players.
    """
    videos_dir = Path(__file__).parent / "static" / "videos"
    video_path = videos_dir / filename

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Ensure file is within videos directory (security)
    if not video_path.resolve().is_relative_to(videos_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


def parse_openai_chat_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    r"""Parse OpenAI Chat format JSONL to extract ground truth annotations.

    JSONL format:
    {
      "messages": [
        {
          "role": "user",
          "content": [
            {"type": "image", "image": "/frames/VideoID_58.6_59.6/frame_00.jpg"},
            ...
          ]
        },
        {
          "role": "assistant",
          "content": [{"type": "text", "text": "{\"action\": \"eject\", \"tool\": \"pipette\", ...}"}]
        }
      ]
    }

    Extract start/end from image path regex: _(\d+\.\d+)_(\d+\.\d+)/
    """
    annotations = []
    for line_num, line in enumerate(jsonl_path.open(), start=1):
        try:
            entry = json.loads(line)
            messages = entry.get("messages", [])

            # Find user message with images
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if not user_msg:
                continue

            # Extract timestamps from first image path
            content = user_msg.get("content", [])
            first_image = next((c for c in content if c.get("type") == "image"), None)
            if not first_image:
                continue

            image_path = first_image.get("image", "")
            # Regex: _(digits.digits)_(digits.digits)/
            match = re.search(r"_(\d+\.\d+)_(\d+\.\d+)/", image_path)
            if not match:
                continue

            start_sec = float(match.group(1))
            end_sec = float(match.group(2))

            # Find assistant message with labels
            assistant_msg = next(
                (m for m in messages if m.get("role") == "assistant"), None
            )
            if not assistant_msg:
                continue

            # Parse JSON from assistant text
            assistant_content = assistant_msg.get("content", [])
            text_item = next(
                (c for c in assistant_content if c.get("type") == "text"), None
            )
            if not text_item:
                continue

            text = text_item.get("text", "{}")
            labels = json.loads(text)

            # Create standardized annotation
            annotations.append({
                "start_ms": int(start_sec * 1000),
                "end_ms": int(end_sec * 1000),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "action": labels.get("action", "unknown"),
                "tool": labels.get("tool", "unknown"),
                "target": labels.get("target", "unknown"),
                "context": labels.get("context", "unknown"),
            })
        except Exception as e:
            logger.warning(
                f"Failed to parse annotation line {line_num} in {jsonl_path.name}: {e}"
            )
            continue

    return sorted(annotations, key=lambda x: x["start_sec"])


@api_router.post("/analysis/start")
async def start_analysis(request: dict[str, Any]) -> dict[str, Any]:
    """Start video analysis job.

    Request body:
        video_filename: str - Name of video file in static/videos/
        annotation_filename: str | None - Name of annotation JSONL file
        segment_time: float - Duration of each segment in seconds (T)
        frames_per_segment: int - Number of frames per segment (s)
        overlap_frames: int - Number of frames to overlap between segments
        simulation_fps: float (deprecated) - Falls back to this if segment params not provided
    """
    state = get_app_state()

    video_filename = request.get("video_filename")
    if not video_filename:
        return {"status": "error", "message": "video_filename is required"}

    annotation_filename = request.get("annotation_filename")

    # Extract segment configuration (new params) or fall back to simulation_fps
    segment_time = request.get("segment_time")
    frames_per_segment = request.get("frames_per_segment")
    overlap_frames = request.get("overlap_frames", 4)

    if segment_time is not None and frames_per_segment is not None:
        # New segment-based configuration
        segment_time = float(segment_time)
        frames_per_segment = int(frames_per_segment)
        overlap_frames = int(overlap_frames)
        # Derive FPS from segment config: FPS = s / T
        simulation_fps = frames_per_segment / segment_time if segment_time > 0 else 5.0
    else:
        # Fall back to old simulation_fps parameter
        simulation_fps = request.get("simulation_fps", 5.0)
        # Infer segment config from fps (assume T=1s)
        segment_time = 1.0
        frames_per_segment = int(simulation_fps)
        overlap_frames = max(0, frames_per_segment // 2)

    # Validate video file exists
    video_path = Path(__file__).parent / "static" / "videos" / video_filename
    if not video_path.exists():
        return {
            "status": "error",
            "message": f"Video file not found: {video_filename}",
        }

    # Clear inference server queue to prevent "ghost results" from previous sessions
    inference_server_url = (
        f"http://{state.config.server.host}:{state.config.server.port}"
    )

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{inference_server_url}/api/queue/clear")
            if response.status_code == 200:
                result = response.json()
                logger.info(
                    f"Cleared inference queue: {result.get('cleared_jobs', 0)} jobs, "
                    f"{result.get('freed_mb', 0):.1f}MB freed"
                )
            else:
                logger.warning(
                    f"Failed to clear inference queue: HTTP {response.status_code}"
                )
    except httpx.TimeoutException:
        logger.warning("Timeout clearing inference queue (server slow)")
    except httpx.ConnectError:
        logger.warning("Could not connect to inference server to clear queue")
    except Exception as e:
        # Non-fatal: log warning but allow analysis to proceed
        logger.warning(f"Error clearing inference queue: {e}")

    # Create VideoFileCapture
    state.analysis_video_capture = VideoFileCapture(
        video_path=str(video_path),
        width=state.config.video.width,
        height=state.config.video.height,
        simulation_fps=simulation_fps,
    )
    if not state.analysis_video_capture.start():
        state.analysis_video_capture = None
        return {"status": "error", "message": "Failed to open video file"}

    # Load and parse annotations if provided (OpenAI Chat format)
    if annotation_filename:
        annotation_path = (
            Path(__file__).parent / "static" / "videos" / annotation_filename
        )
        if annotation_path.exists():
            state.analysis_annotations = parse_openai_chat_jsonl(annotation_path)
            logger.info(
                f"Loaded {len(state.analysis_annotations)} annotations from {annotation_filename}"
            )
        else:
            logger.warning(f"Annotation file not found: {annotation_filename}")
            state.analysis_annotations = []
    else:
        state.analysis_annotations = []

    # [FIX] Define total_frames (native count)
    total_frames = state.analysis_video_capture.total_frames

    # Calculate effective frames based on simulation FPS (sampling rate)
    duration_sec = state.analysis_video_capture.get_duration_ms() / 1000.0
    effective_total_frames = int(duration_sec * simulation_fps)

    # Calculate total chunks for progress tracking
    frames_per_chunk = max(1, frames_per_segment - overlap_frames)
    total_chunks = (
        max(1, (effective_total_frames + frames_per_chunk - 1) // frames_per_chunk)
        if frames_per_chunk > 0
        else 1
    )

    # Create job metadata
    job_id = f"analysis_{uuid.uuid4().hex[:8]}"
    state.active_analysis_job = {
        "job_id": job_id,
        "video_file": video_filename,
        "annotation_file": annotation_filename,
        "start_time": time.time(),
        # Segment configuration
        "segment_time": segment_time,
        "frames_per_segment": frames_per_segment,
        "overlap_frames": overlap_frames,
        "simulation_fps": simulation_fps,  # Derived FPS
        # Video metadata
        "annotation_count": len(state.analysis_annotations),
        "total_frames": total_frames, # Now this variable exists again
        "effective_total_frames": effective_total_frames, # Sampled frames
        "total_chunks": total_chunks,
        "duration_sec": duration_sec,
    }

    # Clear previous analysis results
    state.analysis_results.clear()

    # Create StreamingClient (will be connected via WebSocket)
    def store_analysis_result(result: dict[str, Any]) -> None:
        state.analysis_results.append(result)

    state.analysis_streaming_client = StreamingClient(
        state.config.server.ws_url,
        state.analysis_video_capture,
        result_callback=store_analysis_result,
    )

    return {
        "status": "ok",
        "job_id": job_id,
        "message": "Analysis ready to start",
        "annotation_count": len(state.analysis_annotations),
        "total_frames": total_frames,
        "total_chunks": total_chunks,
        "duration_sec": state.analysis_video_capture.get_duration_ms() / 1000.0,
        # Echo back segment config
        "segment_config": {
            "segment_time": segment_time,
            "frames_per_segment": frames_per_segment,
            "overlap_frames": overlap_frames,
            "derived_fps": simulation_fps,
        },
    }


@api_router.post("/analysis/stop")
async def stop_analysis() -> dict[str, str]:
    """Stop ongoing analysis."""
    state = get_app_state()

    if state.analysis_streaming_client:
        state.analysis_streaming_client.stop()
        state.analysis_streaming_client = None

    if state.analysis_streaming_task:
        state.analysis_streaming_task.cancel()
        state.analysis_streaming_task = None

    if state.analysis_video_capture:
        state.analysis_video_capture.stop()
        state.analysis_video_capture = None

    state.active_analysis_job = None

    return {"status": "ok", "message": "Analysis stopped"}


@ws_router.websocket("/analysis")
async def analysis_websocket(websocket: WebSocket) -> None:
    """WebSocket for analysis streaming.

    Protocol:
        1. Client sends session_config to inference server with segment params
        2. Inference server responds with session_ack
        3. Frames are sent, session_metrics broadcast every 500ms
        4. Results are forwarded to frontend

    Streams progress updates and inference results to the frontend
    while processing a video file through the inference server.
    """
    state = get_app_state()
    await websocket.accept()

    if not state.analysis_video_capture or not state.analysis_streaming_client:
        await websocket.send_json({
            "type": "error",
            "message": "No active analysis job. Call /api/analysis/start first.",
        })
        await websocket.close()
        return

    if not state.active_analysis_job:
        await websocket.send_json({
            "type": "error",
            "message": "No active analysis job metadata found.",
        })
        await websocket.close()
        return

    # Get video metadata
    total_frames = state.analysis_video_capture.total_frames
    duration_ms = state.analysis_video_capture.get_duration_ms()
    duration_sec = duration_ms / 1000.0
    simulation_fps = state.active_analysis_job["simulation_fps"]
    job_id = state.active_analysis_job["job_id"]
    frames_per_segment = state.active_analysis_job.get("frames_per_segment", 8)
    overlap_frames = state.active_analysis_job.get("overlap_frames", 4)
    segment_time = state.active_analysis_job.get("segment_time", 1.0)
    total_chunks = state.active_analysis_job.get("total_chunks", 1)

    # Get inference server URL
    inference_ws_url = state.config.server.ws_url
    logger.info(
        f"Analysis WebSocket connecting to inference server at {inference_ws_url}"
    )

    # Track progress
    frame_count = 0
    results_sent = 0
    start_time = time.time()
    frames_per_chunk = max(1, frames_per_segment - overlap_frames)

    async def send_frames(server_ws: websockets.WebSocketClientProtocol) -> None:
        """Send video frames to inference server at simulation FPS."""
        nonlocal frame_count

        # Progress throttling to prevent UI flooding in Turbo Mode
        last_progress_update_time = 0.0
        PROGRESS_UPDATE_INTERVAL = 0.1  # 100ms = max 10 updates/second

        try:
            native_fps = state.analysis_video_capture.native_fps if state.analysis_video_capture.native_fps > 0 else 30.0
            total_frames_to_send = int(state.analysis_video_capture.get_duration_ms() / 1000 * simulation_fps)
            estimated_time_remaining = 0

            while True:
                # Defensive check: ensure video_capture still exists
                if not state.analysis_video_capture:
                    logger.info("Analysis video capture was stopped, ending frame transmission")
                    # CRITICAL: Send completion signal to prevent server deadlock
                    logger.info("Sending completion signal to inference server...")
                    await server_ws.send(json.dumps({"type": "complete"}))

                    # Notify frontend
                    duration_sec = time.time() - start_time
                    await websocket.send_json({
                        "type": "complete",
                        "job_id": job_id,
                        "total_frames": frame_count,
                        "total_results": len(state.analysis_results),
                        "duration_sec": duration_sec,
                        "actual_send_fps": frame_count / duration_sec if duration_sec > 0 else 0,
                        "speedup_factor": (frame_count / simulation_fps) / duration_sec if duration_sec > 0 else 0,
                    })
                    break

                # Calculate the logical time for the frame we want to send
                logical_time_sec = frame_count / simulation_fps
                
                # Convert this logical time into a physical frame index in the video file
                target_frame_idx = int(logical_time_sec * native_fps)
                
                if target_frame_idx >= state.analysis_video_capture.total_frames:
                    # End of video
                    logger.info(
                        f"Analysis complete: {frame_count} frames sent in "
                        f"{time.time() - start_time:.1f}s"
                    )
                    # [FIX] Send completion signal to INFERENCE SERVER
                    logger.info("Sending completion signal to inference server...")
                    await server_ws.send(json.dumps({"type": "complete"}))
                    
                    # Notify frontend (keep this)
                    duration_sec = time.time() - start_time
                    await websocket.send_json({
                        "type": "complete",
                        "job_id": job_id,
                        "total_frames": frame_count,
                        "total_results": len(state.analysis_results),
                        "duration_sec": duration_sec,
                        "actual_send_fps": frame_count / duration_sec if duration_sec > 0 else 0,
                        "speedup_factor": (frame_count / simulation_fps) / duration_sec if duration_sec > 0 else 0,
                    })
                    break

                # Seek to the correct frame
                state.analysis_video_capture.seek(target_frame_idx)

                frame_jpeg = state.analysis_video_capture.get_frame_jpeg(
                    quality=state.config.video.jpeg_quality
                )

                if frame_jpeg is None:
                    # Safety fallback: end of video reached unexpectedly
                    logger.info(
                        f"Analysis complete (frame read failed): {frame_count} frames sent in "
                        f"{time.time() - start_time:.1f}s"
                    )
                    # CRITICAL: Send completion signal to inference server
                    logger.info("Sending completion signal to inference server...")
                    await server_ws.send(json.dumps({"type": "complete"}))

                    # Notify frontend
                    duration_sec = time.time() - start_time
                    await websocket.send_json({
                        "type": "complete",
                        "job_id": job_id,
                        "total_frames": frame_count,
                        "total_results": len(state.analysis_results),
                        "duration_sec": duration_sec,
                        "actual_send_fps": frame_count / duration_sec if duration_sec > 0 else 0,
                        "speedup_factor": (frame_count / simulation_fps) / duration_sec if duration_sec > 0 else 0,
                    })
                    break

                # The video timestamp is the logical time we calculated
                video_timestamp = logical_time_sec

                # Send frame to inference server
                message = {
                    "frame": base64.b64encode(frame_jpeg).decode(),
                    "frame_id": frame_count,
                    "timestamp": video_timestamp,
                    "fps": simulation_fps,
                }
                await server_ws.send(json.dumps(message))

                # Send progress to frontend (throttled to avoid flooding in Turbo Mode)
                current_time = time.time()
                should_send_progress = (
                    frame_count == 0 or  # Always send first frame
                    (current_time - last_progress_update_time) >= PROGRESS_UPDATE_INTERVAL or
                    target_frame_idx >= state.analysis_video_capture.total_frames - 1  # Always send last frame
                )

                if should_send_progress:
                    progress_percent = (target_frame_idx / total_frames) * 100
                    current_position_ms = video_timestamp * 1000.0

                    # Calculate chunk progress and estimated time
                    current_chunk = frame_count // frames_per_chunk + 1
                    elapsed_time = time.time() - start_time
                    if frame_count > 0 and elapsed_time > 0:
                        frames_per_second_sent = frame_count / elapsed_time
                        remaining_frames_to_send = total_frames_to_send - frame_count
                        estimated_time_remaining = (
                            remaining_frames_to_send / frames_per_second_sent
                            if frames_per_second_sent > 0
                            else 0
                        )

                    await websocket.send_json({
                        "type": "progress",
                        "job_id": job_id,
                        "current_frame": frame_count,
                        "total_frames": total_frames_to_send,
                        "progress_percent": progress_percent,
                        "position_ms": current_position_ms,
                        "current_chunk": current_chunk,
                        "total_chunks": total_chunks,
                        "estimated_time_remaining": estimated_time_remaining,
                    })
                    last_progress_update_time = current_time

                frame_count += 1
                # No sleep - send frames as fast as possible (Turbo Mode)

        except WebSocketDisconnect:
            logger.info("Frontend disconnected during frame sending")
            raise
        except Exception as e:
            logger.error(f"Error sending frames: {e}")
            raise

    async def receive_results(server_ws: websockets.WebSocketClientProtocol) -> None:
        """Receive results from inference server and forward to frontend."""
        nonlocal results_sent
        try:
            while True:
                response = await server_ws.recv()
                message = json.loads(response)
                msg_type = message.get("type")

                # Forward session messages directly to frontend
                if msg_type in ("session_ack", "session_metrics"):
                    await websocket.send_json(message)
                    if msg_type == "session_ack":
                        logger.info(
                            "Analysis session established: id=%s config=%s",
                            message.get("session_id"),
                            message.get("config"),
                        )
                    continue

                # [FIX] Handle processing completion
                if msg_type == "processing_complete":
                    logger.info("Inference server finished all processing.")
                    # Optionally notify frontend if needed, or just exit to close cleanly
                    break

                # Augment result messages with frame/timestamp ranges
                if msg_type == "result":
                    # Defensive check: ensure video_capture still exists
                    if not state.analysis_video_capture:
                        logger.info("Analysis video capture was stopped, ending result reception")
                        break

                    # Trust server's video_time_ms (calculated when segment was created)
                    # This eliminates race condition in Turbo Mode where frame_count
                    # advances to video end before results arrive
                    video_time_ms = message.get("video_time_ms", 0.0)
                    frames_processed = message.get("frames_processed", 8)

                    # Calculate frame range from video time (for display/logging)
                    native_fps = (
                        state.analysis_video_capture.native_fps
                        if state.analysis_video_capture.native_fps > 0
                        else 30.0
                    )
                    start_frame_approx = int((video_time_ms / 1000.0) * native_fps)
                    end_frame_approx = start_frame_approx + frames_processed - 1

                    # Calculate end timestamp from segment duration
                    segment_time = state.active_analysis_job.get("segment_time", 1.0)
                    end_time_ms = video_time_ms + (segment_time * 1000.0)

                    # Augment message with accurate timestamps
                    message["frame_range"] = [start_frame_approx, end_frame_approx]
                    message["timestamp_range_ms"] = [int(video_time_ms), int(end_time_ms)]
                    message["job_id"] = job_id
                    results_sent += 1

                # Forward to frontend
                await websocket.send_json(message)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Inference server connection closed during analysis")
            raise
        except WebSocketDisconnect:
            logger.info("Frontend disconnected while receiving results")
            raise
        except Exception as e:
            logger.error(f"Error receiving results: {e}")
            raise

    try:
        # Connect to inference server
        async with websockets.connect(
            inference_ws_url,
            ping_interval=20,
            ping_timeout=60,
            close_timeout=30.0,
            max_queue=None,
        ) as server_ws:
            logger.info("Connected to inference server for analysis")

            # Send session_config as first message to establish session
            config_message = {
                "type": "session_config",
                "config": {
                    "frames_per_segment": frames_per_segment,
                    "overlap_frames": overlap_frames,
                    "segment_time": segment_time,
                },
                "mode": "analysis",
                "duration_sec": duration_sec,
            }
            await server_ws.send(json.dumps(config_message))
            logger.info(
                "Sent session_config to inference server: "
                "frames_per_segment=%d, overlap=%d, segment_time=%.2f, duration_sec=%.1f",
                frames_per_segment,
                overlap_frames,
                segment_time,
                duration_sec,
            )

            # Run tasks concurrently.
            # send_task finishes when the video ends and it sends the "complete" signal.
            # recv_task finishes when it receives the "processing_complete" signal from the server.
            send_task = asyncio.create_task(send_frames(server_ws))
            recv_task = asyncio.create_task(receive_results(server_ws))

            await asyncio.gather(send_task, recv_task)

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error(f"Inference server returned invalid status: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to connect to inference server: {e}",
        })
        await websocket.close()
    except (ConnectionRefusedError, OSError, TimeoutError) as e:
        logger.error(f"Inference server connection failed: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Cannot connect to inference server: {type(e).__name__}",
        })
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("Analysis WebSocket disconnected")
    except Exception as e:
        logger.error(f"Analysis WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Unexpected error: {e}",
            })
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# Session Management Endpoints
# ============================================================================


@api_router.get("/sessions")
async def list_sessions() -> dict[str, Any]:
    """List all analysis sessions."""
    from iris.client.web.repositories import session_repo

    sessions = session_repo.list_sessions(limit=50)
    return {"sessions": sessions}


@api_router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get a specific session by ID."""
    from iris.client.web.repositories import session_repo

    session = session_repo.get(session_id)
    if not session:
        return {"error": "Session not found", "session_id": session_id}
    return {"session": session}


@api_router.get("/sessions/{session_id}/results")
async def get_session_results(session_id: str) -> dict[str, Any]:
    """Get all inference results for a session."""
    from iris.client.web.repositories import results_repo

    results = results_repo.get_by_session(session_id)
    return {"session_id": session_id, "count": len(results), "results": results}


@api_router.get("/sessions/{session_id}/logs")
async def get_session_logs(session_id: str, limit: int = 1000) -> dict[str, Any]:
    """Get logs for a session."""
    from iris.client.web.repositories import logs_repo

    logs = logs_repo.get_by_session(session_id, limit=limit)
    return {"session_id": session_id, "count": len(logs), "logs": logs}


@api_router.delete("/sessions/{session_id}/logs")
async def clear_session_logs(session_id: str) -> dict[str, Any]:
    """Clear all logs for a session."""
    from iris.client.web.repositories import logs_repo

    deleted = logs_repo.clear_session(session_id)
    return {"status": "ok", "deleted": deleted}


@api_router.delete("/sessions/{session_id}/results")
async def clear_session_results(session_id: str) -> dict[str, Any]:
    """Clear all results for a session."""
    from iris.client.web.repositories import results_repo

    deleted = results_repo.clear_session(session_id)
    return {"status": "ok", "deleted": deleted}


@api_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session and all related data."""
    from iris.client.web.repositories import session_repo

    deleted = session_repo.delete(session_id)
    return {"status": "ok" if deleted else "not_found", "session_id": session_id}


# ============================================================================
# Report Generation Endpoints
# ============================================================================


@api_router.post("/report/generate")
async def generate_report(request: dict[str, Any]) -> Any:
    """Generate Gemini-powered report with streaming.

    Works for both live view and analysis view sessions.
    For live sessions, uses in-memory results_history.
    For analysis sessions, uses database results.

    Request body:
        session_id: Session to generate report for

    Returns:
        StreamingResponse with Markdown content
    """
    from fastapi.responses import StreamingResponse

    from iris.client.web.report_generator import (
        generate_fallback_report,
        generate_report_stream,
    )
    from iris.client.web.repositories import reports_repo, results_repo, session_repo

    session_id = request.get("session_id")
    force_regenerate = request.get("force_regenerate", False)

    if not session_id:
        return {"error": "session_id is required"}

    # Check if report already exists (unless force_regenerate is True)
    if not force_regenerate:
        existing_report = reports_repo.get_latest_by_session(session_id)
        if existing_report:
            logger.info(f"Returning cached report for session {session_id}")
            return {
                "report": existing_report["content"],
                "provider": existing_report.get("provider", "gemini"),
                "cached": True,
                "created_at": existing_report.get("created_at"),
            }

    # Try to get session from database
    session = session_repo.get(session_id)
    state = get_app_state()

    # If no session in DB, this might be a live session
    # Use in-memory data from AppState
    if not session:
        # Create minimal session dict from in-memory state
        session = {
            "id": session_id,
            "status": "running" if state.is_streaming else "idle",
            "created_at": time.time(),
            "config": state.session_config,
            "video_file": None,  # Live sessions don't have video files
            "annotation_file": None,
        }
        # Use in-memory results history for live sessions
        results = state.results_history
    else:
        # Analysis session - get results from database
        results = results_repo.get_by_session(session_id)

    annotations = state.analysis_annotations if state.analysis_annotations else None

    # Check for Gemini API key (stored in UI or environment variables)
    from iris.client.web.report_generator import get_gemini_api_key

    has_gemini = bool(get_gemini_api_key())

    if not has_gemini:
        # Return fallback report
        report = generate_fallback_report(session, results, annotations)
        return {"report": report, "provider": "fallback"}

    # Stream and store report
    start_time = time.time()
    accumulated = []

    async def stream_and_store():
        nonlocal accumulated
        async for chunk in generate_report_stream(session, results, annotations):
            accumulated.append(chunk)
            yield chunk

        # Store complete report (only if session exists in DB)
        if session_repo.get(session_id):
            full_content = "".join(accumulated)
            duration = time.time() - start_time
            try:
                reports_repo.store(
                    session_id=session_id,
                    provider="gemini",
                    content=full_content,
                    generation_duration_sec=duration,
                )
                logger.info(f"Stored report for {session_id} ({duration:.2f}s)")
            except Exception as e:
                logger.error(f"Failed to store report: {e}")

    return StreamingResponse(
        stream_and_store(),
        media_type="text/markdown",
        headers={"X-Report-Provider": "gemini"},
    )


@api_router.get("/report/fallback/{session_id}")
async def get_fallback_report(session_id: str) -> dict[str, Any]:
    """Get a basic statistics report without LLM.

    Useful when no API key is configured.
    """
    from iris.client.web.report_generator import generate_fallback_report
    from iris.client.web.repositories import results_repo, session_repo

    session = session_repo.get(session_id)
    if not session:
        return {"error": "Session not found", "session_id": session_id}

    results = results_repo.get_by_session(session_id)

    state = get_app_state()
    annotations = state.analysis_annotations if state.analysis_annotations else None

    report = generate_fallback_report(session, results, annotations)
    return {"report": report, "provider": "fallback"}


@api_router.get("/report/{session_id}")
async def get_stored_report(session_id: str) -> dict[str, Any]:
    """Get the latest stored report for a session.

    Args:
        session_id: Session identifier.

    Returns:
        Report data or error message.
    """
    from iris.client.web.repositories import reports_repo

    report = reports_repo.get_latest_by_session(session_id)
    if not report:
        return {"error": "No report found", "session_id": session_id}

    return {"session_id": session_id, "report": report}


@api_router.get("/session/{session_id}/data")
async def get_session_data(session_id: str) -> dict[str, Any]:
    """Get session data including logs and results for restoration.

    Args:
        session_id: Session identifier.

    Returns:
        Dictionary with session info, logs, and results.
    """
    from iris.client.web.repositories import (
        logs_repo,
        results_repo,
        session_repo,
    )

    # Check if session exists
    session = session_repo.get(session_id)
    if not session:
        return {"exists": False, "session_id": session_id}

    # Fetch logs and results
    logs = logs_repo.get_by_session(session_id)
    results = results_repo.get_by_session(session_id)

    return {
        "exists": True,
        "session_id": session_id,
        "session": session,
        "logs": logs,
        "results": results,
    }


# ============================================================================
# Simplified Communication Endpoints (New Architecture)
# ============================================================================


@api_router.post("/session/reset")
async def reset_session() -> dict[str, Any]:
    """Reset session and clear all state (client + server)."""
    from iris.client.web.repositories import session_repo, logs_repo, results_repo
    import httpx

    state = get_app_state()

    # Step 1: Generate new session_id and clear local state
    old_session_id = state.session_id
    new_session_id = state.reset_session()

    # Clear database logs and results for old session
    try:
        if logs_repo:
            logs_repo.clear_logs(old_session_id)
        if results_repo:
            results_repo.clear_results(old_session_id)
    except Exception as e:
        logger.warning(f"Failed to clear old session data from database: {e}")

    # Step 2 (CRITICAL): Clear inference server queue to prevent ghost results
    # This prevents buffered results from previous session from arriving
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{state.config.server_url}/api/queue/clear",
                timeout=5.0
            )
            logger.info(f"Cleared inference server queue: {response.status_code}")
    except Exception as e:
        logger.warning(f"Failed to clear inference server queue: {e}")
        # Don't fail reset if server unreachable - allow client reset to proceed

    # Create new session in database
    try:
        session_repo.create(
            session_id=new_session_id,
            config=state.session_config,
        )
        logger.info(f"Created new session in DB after reset: {new_session_id}")
    except Exception as e:
        logger.error(f"Failed to create session in DB after reset: {e}")

    logger.info(
        f"Session reset complete: {old_session_id} → {new_session_id} "
        f"(client state cleared + server queue flushed)"
    )

    return {
        "status": "success",
        "session_id": new_session_id,
        "message": "Session reset (client state + server queue cleared)"
    }


@api_router.post("/queue/clear")
async def clear_queue() -> dict[str, Any]:
    """Clear inference queue by calling the inference server endpoint."""
    import aiohttp

    state = get_app_state()

    # Build the inference server URL for queue clear
    server_url = f"http://{state.config.server.host}:{state.config.server.port}/api/queue/clear"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(server_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Cleared inference queue: {data}")
                    return {"status": "ok", "cleared": data.get("cleared", 0)}
                else:
                    return {"status": "error", "message": f"Server returned {response.status}"}
    except Exception as e:
        logger.error(f"Failed to clear inference queue: {e}")
        return {"status": "error", "message": str(e)}


@ws_router.websocket("/client")
async def client_websocket(websocket: WebSocket) -> None:
    """Unified WebSocket for all frontend communication.

    This is the single WebSocket endpoint for the new simplified architecture.
    Handles:
    - Preview frames (always streaming from USB camera)
    - Start/stop inference commands
    - Session info and reset
    - Results forwarding from inference server
    - Metrics updates

    Simple pattern (from FastAPI docs):
    1. Accept connection
    2. Send initial session info
    3. Start preview streaming task
    4. Loop: receive messages, handle them
    5. Catch WebSocketDisconnect for cleanup
    """
    from iris.client.web.messages import (
        SessionInfoMessage,
        PreviewFrameMessage,
        ServerStatusMessage,
        ErrorMessage,
    )

    state = get_app_state()
    await websocket.accept()
    logger.info(f"Client WebSocket connected, session_id={state.session_id}")

    # Create or restore session in database
    from iris.client.web.repositories import logs_repo, session_repo

    db_session = session_repo.get(state.session_id)
    if not db_session:
        # Create new session in database
        session_repo.create(
            session_id=state.session_id,
            config=state.session_config,
        )
        logger.info(f"Created new session in DB: {state.session_id}")
    else:
        # Session exists - restored from database
        logger.info(f"Restored existing session from DB: {state.session_id}")

    # Helper function to persist logs to database
    def persist_log(level: str, message: str) -> None:
        """Persist log to database if session exists."""
        if state.session_id:
            try:
                logs_repo.append(
                    session_id=state.session_id,
                    level=level,
                    message=message,
                )
            except Exception as e:
                logger.error(f"Failed to persist log: {e}")

    # Log session connection
    persist_log("INFO", f"Session {state.session_id} connected")

    # Send session info immediately on connect
    session_info = SessionInfoMessage(
        session_id=state.session_id,
        config=state.session_config,
    ).model_dump()
    logger.info(f"Sending to frontend: type=session_info")
    await websocket.send_json(session_info)

    # Track active tasks
    preview_task: asyncio.Task | None = None
    health_check_task: asyncio.Task | None = None

    async def stream_preview_frames() -> None:
        """Stream USB camera frames to frontend continuously."""
        try:
            # Initialize camera if needed
            if state.camera is None:
                state.camera = CameraCapture(
                    camera_index=state.config.video.camera_index,
                    width=state.config.video.width,
                    height=state.config.video.height,
                    fps=state.config.video.capture_fps,
                )
                if not state.camera.start():
                    logger.error("Failed to start camera for preview")
                    error_msg = ErrorMessage(message="Failed to start camera").model_dump()
                    logger.info(f"Sending error to frontend: {error_msg}")
                    await websocket.send_json(error_msg)
                    state.camera = None
                    return

            preview_fps = min(10, state.config.video.capture_fps)  # Cap at 10 FPS
            frame_counter = 0
            while True:
                if state.camera:
                    frame_jpeg = state.camera.get_frame_jpeg(quality=70)
                    if frame_jpeg:
                        frame_counter += 1
                        await websocket.send_json(
                            PreviewFrameMessage(
                                frame=base64.b64encode(frame_jpeg).decode(),
                                timestamp=time.time(),
                            ).model_dump()
                        )
                        # Log every 30 frames to avoid spam (3 seconds at 10 FPS)
                        if frame_counter % 30 == 0:
                            logger.debug(f"Sent preview frame #{frame_counter} to frontend")
                await asyncio.sleep(1.0 / preview_fps)
        except asyncio.CancelledError:
            logger.debug("Preview streaming cancelled")
        except Exception as e:
            logger.error(f"Preview streaming error: {e}")

    async def check_inference_server() -> None:
        """Periodically check if inference server is alive."""
        import aiohttp

        server_url = f"http://{state.config.server.host}:{state.config.server.port}/health"
        check_interval = 5.0  # seconds

        while True:
            queue_depth = None
            try:
                async with aiohttp.ClientSession() as http_session:
                    async with http_session.get(
                        server_url, timeout=aiohttp.ClientTimeout(total=2)
                    ) as response:
                        state.inference_server_alive = response.status == 200
                        if response.status == 200:
                            data = await response.json()
                            queue_depth = data.get("queue_depth", 0)
            except Exception:
                state.inference_server_alive = False

            # Send status to frontend
            try:
                status_msg = ServerStatusMessage(
                    alive=state.inference_server_alive,
                    queue_depth=queue_depth,
                ).model_dump()
                logger.debug(f"Sending server status to frontend: alive={state.inference_server_alive}, queue_depth={queue_depth}")
                await websocket.send_json(status_msg)
            except Exception:
                break  # WebSocket closed

            await asyncio.sleep(check_interval)

    # Start background tasks
    preview_task = asyncio.create_task(stream_preview_frames())
    health_check_task = asyncio.create_task(check_inference_server())

    try:
        while True:
            # Simple receive - no timeout, no complex retry logic
            data = await websocket.receive_json()
            msg_type = data.get("type")
            logger.info(f"Received from frontend: type={msg_type}")

            if msg_type == "start":
                # Start inference streaming
                config = data.get("config", {})
                state.session_config = config
                state.is_streaming = True
                logger.info(f"Starting inference with config: {config}")

                # Update session status in database
                session_repo.update_status(
                    session_id=state.session_id,
                    status="running",
                    started_at=time.time(),
                )

                # Log inference start
                persist_log("INFO", f"Inference started with config: {config}")

                # Start streaming to inference server
                try:
                    if state.camera is None:
                        state.camera = CameraCapture(
                            camera_index=state.config.video.camera_index,
                            width=state.config.video.width,
                            height=state.config.video.height,
                            fps=state.config.video.capture_fps,
                        )
                        if not state.camera.start():
                            error_msg = ErrorMessage(message="Failed to start camera").model_dump()
                            logger.error("Failed to start camera for inference")
                            persist_log("ERROR", "Failed to start camera for inference")
                            logger.info(f"Sending error to frontend: {error_msg}")
                            await websocket.send_json(error_msg)
                            state.is_streaming = False
                            continue

                    def store_result(result: dict[str, Any]) -> None:
                        msg_type = result.get('type')
                        logger.info(f"StreamingClient received from inference server: type={msg_type}, job_id={result.get('job_id')}")

                        # Store in memory
                        state.results_history.append(result)
                        if len(state.results_history) > state.max_results_history:
                            state.results_history.pop(0)

                        # Persist result to database
                        if msg_type == "result" and state.session_id:
                            try:
                                from iris.client.web.repositories import results_repo
                                results_repo.store(
                                    session_id=state.session_id,
                                    job_id=result.get("job_id", "unknown"),
                                    video_time_ms=int(result.get("timestamp", 0) * 1000),  # Convert to ms
                                    inference_start_ms=result.get("timestamp", 0) * 1000,
                                    inference_end_ms=(result.get("timestamp", 0) + result.get("inference_time", 0)) * 1000,
                                    frame_start=0,  # Would need frame info from result
                                    frame_end=result.get("frames_processed", 0),
                                    result={"raw": result.get("result", "")},
                                )
                                logger.debug(f"Persisted result to DB: {result.get('job_id')}")
                            except Exception as e:
                                logger.error(f"Failed to persist result to DB: {e}")

                        # Forward result to frontend via WebSocket
                        logger.info(f"Forwarding to frontend: type={msg_type}, job_id={result.get('job_id')}")
                        asyncio.create_task(websocket.send_json(result))

                    # Calculate streaming FPS from segment configuration
                    # FPS = s/T (frames_per_segment / segment_time)
                    frames_per_segment = config.get("frames_per_segment", 2)
                    segment_time = config.get("segment_time", 1.0)
                    streaming_fps = frames_per_segment / segment_time if segment_time > 0 else state.config.video.capture_fps

                    logger.info(f"Starting streaming with calculated FPS: {streaming_fps:.2f} (frames={frames_per_segment}, time={segment_time}s)")

                    state.streaming_client = StreamingClient(
                        state.config.server.ws_url,
                        state.camera,
                        result_callback=store_result,
                        session_config={
                            "frames_per_segment": frames_per_segment,
                            "overlap_frames": config.get("overlap_frames", 0),
                            # Client session_id NEVER sent to inference server (strict architecture boundary)
                        },
                        streaming_fps=streaming_fps,
                    )
                    state.streaming_task = asyncio.create_task(
                        state.streaming_client.stream()
                    )
                    logger.info("Inference streaming started")
                except Exception as e:
                    logger.error(f"Failed to start inference: {e}")
                    persist_log("ERROR", f"Failed to start inference: {e}")
                    error_msg = ErrorMessage(message=f"Failed to start: {e}").model_dump()
                    logger.info(f"Sending error to frontend: {error_msg}")
                    await websocket.send_json(error_msg)
                    state.is_streaming = False

            elif msg_type == "stop":
                # Stop inference streaming
                logger.info("Stopping inference streaming")
                state.is_streaming = False
                if state.streaming_client:
                    state.streaming_client.stop()
                    state.streaming_client = None

                # Update session status in database
                session_repo.update_status(
                    session_id=state.session_id,
                    status="paused",
                    completed_at=time.time(),
                )

                # Log inference stop
                persist_log("INFO", "Inference stopped")

                logger.info("Inference streaming stopped")

            elif msg_type == "clear_queue":
                # Clear inference queue
                logger.info("Clearing inference queue")
                result = await clear_queue()
                response = {"type": "queue_cleared", **result}
                logger.info(f"Sending to frontend: type=queue_cleared, result={result}")
                await websocket.send_json(response)

            elif msg_type == "reset_session":
                # Reset session
                logger.info("Resetting session")
                new_session_id = state.reset_session()

                # Create new session in database immediately
                try:
                    session_repo.create(
                        session_id=new_session_id,
                        config=state.session_config,
                    )
                    logger.info(f"Created new session in DB after reset: {new_session_id}")
                    persist_log("INFO", f"Session reset: new session {new_session_id}")
                except Exception as e:
                    logger.error(f"Failed to create session in DB after reset: {e}")

                session_msg = SessionInfoMessage(
                    session_id=new_session_id,
                    config=state.session_config,
                ).model_dump()
                logger.info(f"Sending to frontend: type=session_info, new_session_id={new_session_id}")
                await websocket.send_json(session_msg)
                logger.info(f"Session reset to: {new_session_id}")

            elif msg_type == "start_analysis":
                # Handle analysis start
                logger.info(f"Start analysis requested: {data}")
                error_msg = ErrorMessage(
                    message="Analysis via WebSocket not yet implemented"
                ).model_dump()
                logger.info(f"Sending error to frontend: {error_msg}")
                await websocket.send_json(error_msg)

    except WebSocketDisconnect:
        logger.info("Client WebSocket disconnected")
    except Exception as e:
        logger.error(f"Client WebSocket error: {e}")
    finally:
        logger.info("Starting WebSocket cleanup...")
        # Cleanup
        if preview_task:
            preview_task.cancel()
            try:
                await preview_task
            except asyncio.CancelledError:
                pass
            logger.debug("Preview task cancelled")

        if health_check_task:
            health_check_task.cancel()
            try:
                await health_check_task
            except asyncio.CancelledError:
                pass
            logger.debug("Health check task cancelled")

        # Stop inference if running
        if state.streaming_client:
            state.streaming_client.stop()
            state.streaming_client = None
            logger.info("StreamingClient stopped")
        state.is_streaming = False

        # Stop camera if no longer needed
        if state.camera and not state.is_streaming:
            state.camera.stop()
            state.camera = None
            logger.info("Camera stopped")

        logger.info("Client WebSocket cleanup complete")

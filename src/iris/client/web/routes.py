"""API routes for the IRIS client web interface."""

import asyncio
import base64
import logging
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from iris.client.capture.camera import CameraCapture
from iris.client.config import ServerConfig
from iris.client.streaming.websocket_client import StreamingClient
from iris.client.web.dependencies import get_app_state

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    """Serve web UI."""
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_file.read_text())


@router.get("/status")
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


@router.post("/config")
async def update_config(new_config: ServerConfig) -> dict[str, Any]:
    """Update server configuration."""
    state = get_app_state()
    state.config.server = new_config
    return {"status": "ok", "config": state.config.server.model_dump()}


@router.post("/tunnel/config")
async def update_tunnel_config(request: dict[str, str]) -> dict[str, Any]:
    """Update SSH tunnel remote hostname."""
    state = get_app_state()

    if "remote_host" in request:
        state.config.ssh_tunnel.remote_host = request["remote_host"]

    return {"status": "ok", "remote_host": state.config.ssh_tunnel.remote_host}


@router.post("/start")
async def start_streaming() -> dict[str, Any]:
    """Start camera and streaming."""
    state = get_app_state()

    # Start SSH tunnel if enabled
    if state.config.ssh_tunnel.enabled and state.config.ssh_tunnel.remote_host:
        ssh_key = Path(state.config.ssh_tunnel.ssh_key_path).expanduser()

        cmd = [
            "ssh",
            "-N",
            "-L",
            f"8001:{state.config.ssh_tunnel.remote_host}:8001",
            "-i",
            str(ssh_key),
            f"{state.config.ssh_tunnel.ssh_user}@{state.config.ssh_tunnel.ssh_host}",
        ]

        state.tunnel_process = subprocess.Popen(cmd)
        logger.info("Started SSH tunnel to %s", state.config.ssh_tunnel.remote_host)

    # Start camera
    if state.camera is None:
        state.camera = CameraCapture(
            camera_index=state.config.video.camera_index,
            width=state.config.video.width,
            height=state.config.video.height,
            fps=state.config.video.fps,
        )
        if not state.camera.start():
            return {"status": "error", "message": "Failed to start camera"}

    # Start streaming
    def store_result(result: dict[str, Any]) -> None:
        """Callback to store inference results."""
        state.results_history.append(result)
        # Keep only the most recent results
        if len(state.results_history) > state.max_results_history:
            state.results_history.pop(0)

    state.streaming_client = StreamingClient(
        state.config.server.ws_url, state.camera, result_callback=store_result
    )
    task = asyncio.create_task(state.streaming_client.stream())
    # Store task reference to prevent it from being garbage collected
    state.streaming_task = task

    return {"status": "ok", "message": "Streaming started"}


@router.post("/stop")
async def stop_streaming() -> dict[str, str]:
    """Stop streaming and camera."""
    state = get_app_state()

    if state.streaming_client:
        state.streaming_client.stop()
        state.streaming_client = None

    if state.camera:
        state.camera.stop()
        state.camera = None

    # Stop SSH tunnel
    if state.tunnel_process:
        state.tunnel_process.terminate()
        state.tunnel_process = None
        logger.info("Stopped SSH tunnel")

    return {"status": "ok", "message": "Stopped"}


@router.get("/cameras")
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


@router.post("/camera/select")
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
        fps=state.config.video.fps,
    )

    if not state.camera.start():
        state.camera = None
        return {"status": "error", "message": f"Failed to open camera {camera_index}"}

    return {"status": "ok", "camera_index": camera_index}


@router.websocket("/preview")
async def preview_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for local video preview."""
    state = get_app_state()
    await websocket.accept()

    # Initialize camera if not already running
    if state.camera is None:
        state.camera = CameraCapture(
            camera_index=state.config.video.camera_index,
            width=state.config.video.width,
            height=state.config.video.height,
            fps=state.config.video.fps,
        )
        if not state.camera.start():
            state.camera = None
            await websocket.close()
            return

    try:
        while True:
            if state.camera:
                frame_jpeg = state.camera.get_frame_jpeg(quality=70)
                if frame_jpeg:
                    await websocket.send_text(base64.b64encode(frame_jpeg).decode())
            await asyncio.sleep(0.05)  # 20 FPS preview
    except WebSocketDisconnect:
        pass
    finally:
        # Stop camera if we're not streaming to server
        if state.camera and (
            state.streaming_client is None or not state.streaming_client.running
        ):
            state.camera.stop()
            state.camera = None


@router.websocket("/results")
async def results_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for inference results."""
    state = get_app_state()
    await websocket.accept()

    last_sent_index = 0

    try:
        while True:
            # Send all new results since last check
            current_count = len(state.results_history)
            if current_count > last_sent_index:
                # Send all new results
                for i, result in enumerate(
                    state.results_history[last_sent_index:], start=last_sent_index
                ):
                    try:
                        await websocket.send_json(result)
                        logger.debug(
                            f"Sent result {i + 1}/{current_count}: {result.get('job_id')}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send result {i}: {e}")
                        raise  # Re-raise to close connection
                last_sent_index = current_count
            await asyncio.sleep(0.1)  # Check for new results 10 times per second
    except WebSocketDisconnect:
        logger.info("Results WebSocket disconnected")
    except Exception as e:
        logger.error(f"Results WebSocket error: {e}")


@router.get("/results/history")
async def get_results_history() -> dict[str, Any]:
    """Get all stored inference results."""
    state = get_app_state()
    return {
        "count": len(state.results_history),
        "results": state.results_history,
    }


@router.post("/results/clear")
async def clear_results_history() -> dict[str, str]:
    """Clear stored inference results history."""
    state = get_app_state()
    state.results_history.clear()
    return {"status": "ok", "message": "Results history cleared"}

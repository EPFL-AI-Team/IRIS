"""API routes for the IRIS client web interface."""

import asyncio
import base64
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from iris.client.capture.camera import CameraCapture
from iris.client.config import ServerConfig
from iris.client.streaming.websocket_client import StreamingClient
from iris.client.web.dependencies import get_app_state

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
    return {
        "camera_active": state.camera is not None and state.camera.cap.isOpened(),
        "streaming_active": state.streaming_client is not None
        and state.streaming_client.running,
        "config": state.config.dict(),
        "fps": state.streaming_client.get_fps() if state.streaming_client else 0.0,
    }


@router.post("/config")
async def update_config(new_config: ServerConfig) -> dict[str, Any]:
    """Update server configuration."""
    state = get_app_state()
    state.config.server = new_config
    return {"status": "ok", "config": state.config.server.dict()}


@router.post("/start")
async def start_streaming() -> dict[str, Any]:
    """Start camera and streaming."""
    state = get_app_state()

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
    state.streaming_client = StreamingClient(state.config.server.ws_url, state.camera)
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

    return {"status": "ok", "message": "Stopped"}


@router.websocket("/preview")
async def preview_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for local video preview."""
    state = get_app_state()
    await websocket.accept()

    try:
        while True:
            if state.camera:
                frame_jpeg = state.camera.get_frame_jpeg(quality=70)
                if frame_jpeg:
                    await websocket.send_text(base64.b64encode(frame_jpeg).decode())
            await asyncio.sleep(0.05)  # 20 FPS preview
    except WebSocketDisconnect:
        pass

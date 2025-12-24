"""API routes for the IRIS client web interface."""

import asyncio
import base64
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from iris.client.capture.camera import CameraCapture
from iris.client.config import ServerConfig
from iris.client.streaming.websocket_client import StreamingClient
from iris.client.web.dependencies import get_app_state

logger = logging.getLogger(__name__)
api_router = APIRouter(prefix="/api")  # HTTP endpoints
ws_router = APIRouter(prefix="/ws")  # WebSocket endpoints


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


@api_router.post("/tunnel/config")
async def update_tunnel_config(request: dict[str, str]) -> dict[str, Any]:
    """Update SSH tunnel remote hostname."""
    state = get_app_state()

    if "remote_host" in request:
        state.config.ssh_tunnel.remote_host = request["remote_host"]

    return {"status": "ok", "remote_host": state.config.ssh_tunnel.remote_host}


@api_router.post("/start")
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
            fps=state.config.video.capture_fps,
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

    # Stop SSH tunnel
    if state.tunnel_process:
        state.tunnel_process.terminate()
        state.tunnel_process = None
        logger.info("Stopped SSH tunnel")

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


@ws_router.websocket("/preview")
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
            fps=state.config.video.capture_fps,
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
    except Exception as e:
        logger.error(f"Preview WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        # Stop camera if we're not streaming to server
        if state.camera and (
            state.streaming_client is None or not state.streaming_client.running
        ):
            state.camera.stop()
            state.camera = None


@ws_router.websocket("/results")
async def results_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for inference results and status updates."""
    state = get_app_state()
    await websocket.accept()

    last_sent_index = 0
    loop = asyncio.get_event_loop()
    last_keepalive = loop.time()
    last_status_update = loop.time()
    keepalive_interval = 15.0  # seconds
    status_update_interval = 1.0  # seconds - send status every second

    try:
        while True:
            now = loop.time()

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
                    except WebSocketDisconnect:
                        logger.info("Results WebSocket disconnected during send")
                        raise
                    except Exception as e:
                        logger.error(f"Failed to send result {i}: {e}")
                        raise  # Re-raise to close connection
                last_sent_index = current_count
                last_keepalive = now  # Reset timer when data sent

            # Send periodic status updates (replacing HTTP polling)
            if now - last_status_update > status_update_interval:
                try:
                    # Determine streaming server connection status (with defensive checks)
                    streaming_server_status = "disconnected"
                    streaming_active = False
                    current_fps = 0.0
                    camera_active = False

                    try:
                        if state.streaming_client:
                            streaming_server_status = state.streaming_client.connection_state
                            streaming_active = state.streaming_client.running
                            current_fps = state.streaming_client.get_fps()
                    except Exception:
                        pass  # Use defaults on error

                    try:
                        if state.camera and state.camera.cap:
                            camera_active = state.camera.cap.isOpened()
                    except Exception:
                        pass  # Use default on error

                    status_message = {
                        "type": "status_update",
                        "camera_active": camera_active,
                        "streaming_active": streaming_active,
                        "streaming_server_status": streaming_server_status,
                        "fps": current_fps,
                        "config": {
                            "server": {
                                "host": state.config.server.host,
                                "port": state.config.server.port,
                                "endpoint": state.config.server.endpoint,
                            },
                            "ssh_tunnel": {
                                "remote_host": state.config.ssh_tunnel.remote_host,
                            },
                        },
                        "timestamp": now,
                    }
                    await websocket.send_json(status_message)
                    last_status_update = now
                    last_keepalive = now  # Status update counts as keepalive
                except WebSocketDisconnect:
                    logger.info("Results WebSocket disconnected during status update")
                    raise
                except Exception as e:
                    logger.error(f"Failed to send status update: {e}")
                    # Don't raise - continue trying

            # Keepalive ping when idle (fallback if status updates fail)
            elif now - last_keepalive > keepalive_interval:
                try:
                    await websocket.send_json({"type": "keepalive", "timestamp": now})
                    last_keepalive = now
                except WebSocketDisconnect:
                    logger.info("Results WebSocket disconnected during keepalive")
                    raise

            await asyncio.sleep(0.1)  # Check for new results 10 times per second
    except asyncio.CancelledError:
        logger.debug("Results WebSocket cancelled during shutdown")
        # Don't raise - let it close gracefully
    except WebSocketDisconnect:
        logger.info("Results WebSocket disconnected")
    except Exception as e:
        logger.error(f"Results WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


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


@ws_router.websocket("/browser-stream")
async def browser_stream_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for browser camera frame streaming.

    Accepts frames from the browser camera and forwards them to the inference server.
    Results are sent back to the browser via this same connection.
    """
    state = get_app_state()
    await websocket.accept()

    # Get inference server URL from config
    inference_ws_url = state.config.server.ws_url
    logger.info("Browser stream connecting to inference server at %s", inference_ws_url)

    # Track connection state
    frame_count = 0
    start_time = time.time()
    results_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def forward_to_inference(
        server_ws: websockets.WebSocketClientProtocol,
    ) -> None:
        """Receive frames from browser and forward to inference server."""
        nonlocal frame_count
        try:
            while True:
                # Receive frame from browser
                data = await websocket.receive_text()
                frame_data = json.loads(data)

                # Forward to inference server with same format
                message = {
                    "frame": frame_data["frame"],
                    "frame_id": frame_data.get("frame_id", frame_count),
                    "timestamp": frame_data.get("timestamp", time.time()),
                    "fps": frame_data.get("fps", 5.0),
                    "measured_fps": frame_data.get("measured_fps", 0.0),
                }
                await server_ws.send(json.dumps(message))
                frame_count += 1
        except WebSocketDisconnect:
            logger.info("Browser disconnected from browser-stream")
            raise
        except Exception as e:
            logger.error("Error forwarding frame to inference server: %s", e)
            raise

    async def receive_from_inference(
        server_ws: websockets.WebSocketClientProtocol,
    ) -> None:
        """Receive results from inference server and queue for browser."""
        try:
            while True:
                response = await server_ws.recv()
                message = json.loads(response)

                # Store result in history (same as StreamingClient)
                if message.get("type") == "result":
                    state.results_history.append(message)
                    if len(state.results_history) > state.max_results_history:
                        state.results_history.pop(0)

                # Check for batch submission in log messages
                elif message.get("type") == "log":
                    log_text = message.get("message", "")
                    log_job = message.get("job_id")

                    if "Submitted video_job_" in log_text:
                        match = re.search(
                            r"Submitted (video_job_\w+_batch_\d+)", log_text
                        )
                        if match:
                            job_id = match.group(1)
                            batch_msg = {
                                "type": "batch_submitted",
                                "job_id": job_id,
                                "timestamp": time.time(),
                                "status": "processing",
                            }
                            state.results_history.append(batch_msg)
                            await results_queue.put(batch_msg)

                    if "Starting inference:" in log_text:
                        log_msg = {
                            "type": "log",
                            "message": log_text,
                            "job_id": log_job,
                            "timestamp": time.time(),
                        }
                        state.results_history.append(log_msg)
                        await results_queue.put(log_msg)

                # Queue result to send back to browser
                await results_queue.put(message)

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(
                "Inference server connection closed: code=%s reason=%s",
                getattr(e, "code", None),
                getattr(e, "reason", None),
            )
            raise
        except Exception as e:
            logger.error("Error receiving from inference server: %s", e)
            raise

    async def send_to_browser() -> None:
        """Send queued results back to browser."""
        try:
            while True:
                result = await results_queue.get()
                await websocket.send_json(result)
        except WebSocketDisconnect:
            logger.info("Browser disconnected while sending results")
            raise
        except Exception as e:
            logger.error("Error sending result to browser: %s", e)
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
            logger.info("Connected to inference server for browser stream")

            # Run all three tasks concurrently
            forward_task = asyncio.create_task(forward_to_inference(server_ws))
            receive_task = asyncio.create_task(receive_from_inference(server_ws))
            send_task = asyncio.create_task(send_to_browser())

            try:
                _, pending = await asyncio.wait(
                    [forward_task, receive_task, send_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            except Exception as e:
                forward_task.cancel()
                receive_task.cancel()
                send_task.cancel()
                raise e

    except websockets.exceptions.InvalidStatusCode as e:
        logger.error("Inference server returned invalid status: %s", e)
        await websocket.send_json({
            "type": "error",
            "message": f"Failed to connect to inference server: {e}",
        })
        await websocket.close()
    except (ConnectionRefusedError, OSError, TimeoutError) as e:
        logger.error("Inference server connection failed at %s: %s", inference_ws_url, e)
        await websocket.send_json({
            "type": "error",
            "message": f"Cannot connect to inference server. Is it running? ({type(e).__name__})",
        })
        await websocket.close()
    except WebSocketDisconnect:
        logger.info("Browser stream WebSocket disconnected")
    except Exception as e:
        logger.error("Browser stream error: %s", e)
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Unexpected error: {e}",
            })
            await websocket.close()
        except Exception:
            pass  # WebSocket may already be closed
    finally:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Browser stream ended: %d frames in %.1fs (%.1f fps)",
            frame_count,
            elapsed,
            fps,
        )

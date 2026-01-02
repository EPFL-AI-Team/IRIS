"""API routes for the IRIS client web interface."""

import asyncio
import base64
import json
import logging
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import websockets
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from iris.client.capture.camera import CameraCapture
from iris.client.capture.video_file import VideoFileCapture
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
            # Match preview rate to configured capture FPS
            await asyncio.sleep(1.0 / state.config.video.capture_fps)
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
    # Send an initial status update immediately on connect.
    last_status_update = 0.0
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
                            streaming_server_status = (
                                state.streaming_client.connection_state
                            )
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
                            "video": {
                                "width": state.config.video.width,
                                "height": state.config.video.height,
                                "capture_fps": state.config.video.capture_fps,
                                "jpeg_quality": state.config.video.jpeg_quality,
                                "camera_index": state.config.video.camera_index,
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

    Protocol:
        1. Frontend sends segment_config as FIRST message
        2. Client connects to inference server with that config
        3. Inference server responds with session_ack
        4. Frames are forwarded, results streamed back
        5. session_metrics broadcast every 500ms

    Accepts frames from the browser camera and forwards them to the inference server.
    Results are sent back to the browser via this same connection.
    """
    state = get_app_state()
    await websocket.accept()

    # Get inference server URL from config
    inference_ws_url = state.config.server.ws_url
    logger.info("Browser stream connecting to inference server at %s", inference_ws_url)

    # Wait for segment_config from frontend FIRST before connecting to inference server
    logger.info("Waiting for segment_config from frontend...")

    try:
        initial_data = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        initial_msg = json.loads(initial_data)

        if initial_msg.get("type") != "segment_config":
            await websocket.send_json({
                "type": "error",
                "message": "Expected segment_config as first message",
            })
            await websocket.close()
            return

        # Use frontend-provided config
        session_config = {
            "frames_per_segment": initial_msg.get(
                "frames_per_segment", state.config.video.capture_fps
            ),
            "overlap_frames": initial_msg.get(
                "overlap_frames", max(1, state.config.video.capture_fps // 2)
            ),
        }
        logger.info(
            "Received segment_config from frontend: frames_per_segment=%d, overlap=%d",
            session_config["frames_per_segment"],
            session_config["overlap_frames"],
        )

    except TimeoutError:
        await websocket.send_json({
            "type": "error",
            "message": "Timeout waiting for segment_config",
        })
        await websocket.close()
        return

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
                msg_type = message.get("type")

                # Forward session messages directly to browser
                if msg_type in ("session_ack", "session_metrics"):
                    await results_queue.put(message)
                    if msg_type == "session_ack":
                        logger.info(
                            "Session established: id=%s config=%s",
                            message.get("session_id"),
                            message.get("config"),
                        )
                    continue

                # Store result in history (same as StreamingClient)
                if msg_type == "result":
                    state.results_history.append(message)
                    if len(state.results_history) > state.max_results_history:
                        state.results_history.pop(0)

                # Check for batch submission in log messages
                elif msg_type == "log":
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

            # Send session_config as first message to establish session
            config_message = {
                "type": "session_config",
                "config": session_config,
                "mode": "live",
                "total_frames": None,  # Unknown for live streaming
            }
            await server_ws.send(json.dumps(config_message))
            logger.info(
                "Sent session_config to inference server: frames_per_segment=%d, overlap=%d",
                session_config["frames_per_segment"],
                session_config["overlap_frames"],
            )

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
        logger.error(
            "Inference server connection failed at %s: %s", inference_ws_url, e
        )
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
    """Parse OpenAI Chat format JSONL to extract ground truth annotations.

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
        annotation_path = Path(__file__).parent / "static" / "videos" / annotation_filename
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

    # Calculate total chunks for progress tracking
    total_frames = state.analysis_video_capture.total_frames
    frames_per_chunk = frames_per_segment - overlap_frames
    total_chunks = max(1, (total_frames + frames_per_chunk - 1) // frames_per_chunk) if frames_per_chunk > 0 else 1

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
        "total_frames": total_frames,
        "total_chunks": total_chunks,
        "duration_sec": state.analysis_video_capture.get_duration_ms() / 1000.0,
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
    simulation_fps = state.active_analysis_job["simulation_fps"]
    job_id = state.active_analysis_job["job_id"]
    frames_per_segment = state.active_analysis_job.get("frames_per_segment", 8)
    overlap_frames = state.active_analysis_job.get("overlap_frames", 4)
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
        try:
            while True:
                frame_jpeg = state.analysis_video_capture.get_frame_jpeg(
                    quality=state.config.video.jpeg_quality
                )
                if frame_jpeg is None:
                    # End of video
                    logger.info(
                        f"Analysis complete: {frame_count} frames processed in "
                        f"{time.time() - start_time:.1f}s"
                    )
                    await websocket.send_json({
                        "type": "complete",
                        "job_id": job_id,
                        "total_frames": frame_count,
                        "total_results": len(state.analysis_results),
                        "duration_sec": time.time() - start_time,
                    })
                    break

                # Send frame to inference server
                message = {
                    "frame": base64.b64encode(frame_jpeg).decode(),
                    "frame_id": frame_count,
                    "timestamp": time.time(),
                    "fps": simulation_fps,
                }
                await server_ws.send(json.dumps(message))

                # Send progress to frontend
                progress_percent = (frame_count / total_frames) * 100
                current_position_ms = state.analysis_video_capture.get_position_ms()

                # Calculate chunk progress and estimated time
                current_chunk = frame_count // frames_per_chunk + 1
                elapsed_time = time.time() - start_time
                if frame_count > 0 and elapsed_time > 0:
                    frames_per_second = frame_count / elapsed_time
                    remaining_frames = total_frames - frame_count
                    estimated_time_remaining = remaining_frames / frames_per_second if frames_per_second > 0 else 0
                else:
                    estimated_time_remaining = 0

                await websocket.send_json({
                    "type": "progress",
                    "job_id": job_id,
                    "current_frame": frame_count,
                    "total_frames": total_frames,
                    "progress_percent": progress_percent,
                    "position_ms": current_position_ms,
                    "current_chunk": current_chunk,
                    "total_chunks": total_chunks,
                    "estimated_time_remaining": estimated_time_remaining,
                })

                frame_count += 1
                await asyncio.sleep(1.0 / simulation_fps)

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

                # Augment result messages with frame/timestamp ranges
                if msg_type == "result":
                    # Add metadata about frame range
                    frames_processed = message.get("frames_processed", 8)
                    current_frame = frame_count
                    start_frame = max(0, current_frame - frames_processed)

                    message["frame_range"] = [start_frame, current_frame - 1]
                    message["timestamp_range_ms"] = [
                        int(start_frame / state.analysis_video_capture.native_fps * 1000)
                        if state.analysis_video_capture.native_fps > 0
                        else 0,
                        int(
                            (current_frame - 1)
                            / state.analysis_video_capture.native_fps
                            * 1000
                        )
                        if state.analysis_video_capture.native_fps > 0
                        else 0,
                    ]
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
                },
                "mode": "analysis",
                "total_frames": total_frames,
            }
            await server_ws.send(json.dumps(config_message))
            logger.info(
                "Sent session_config to inference server: "
                "frames_per_segment=%d, overlap=%d, total_frames=%d",
                frames_per_segment,
                overlap_frames,
                total_frames,
            )

            # Run send and receive tasks concurrently
            send_task = asyncio.create_task(send_frames(server_ws))
            recv_task = asyncio.create_task(receive_results(server_ws))

            try:
                _, pending = await asyncio.wait(
                    [send_task, recv_task],
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
                send_task.cancel()
                recv_task.cancel()
                raise e

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
    """Generate an LLM-powered analysis report.

    Request body:
        session_id: Session to generate report for
        provider: Optional LLM provider ("anthropic" or "openai")

    Returns:
        StreamingResponse with Markdown content
    """
    from fastapi.responses import StreamingResponse

    from iris.client.web.report_generator import (
        generate_fallback_report,
        generate_report_stream,
    )
    from iris.client.web.repositories import results_repo, session_repo

    session_id = request.get("session_id")
    provider = request.get("provider", "anthropic")

    if not session_id:
        return {"error": "session_id is required"}

    # Get session and results
    session = session_repo.get(session_id)
    if not session:
        return {"error": "Session not found", "session_id": session_id}

    results = results_repo.get_by_session(session_id)

    # Load annotations if available
    state = get_app_state()
    annotations = state.analysis_annotations if state.analysis_annotations else None

    # Check if LLM API key is available
    import os

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        # Return fallback report
        report = generate_fallback_report(session, results, annotations)
        return {"report": report, "provider": "fallback"}

    # Stream LLM-generated report
    async def stream_generator():
        async for chunk in generate_report_stream(
            session, results, annotations, provider
        ):
            yield chunk

    return StreamingResponse(
        stream_generator(),
        media_type="text/markdown",
        headers={"X-Report-Provider": provider},
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

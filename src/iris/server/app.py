"""IRIS Inference Server - receives frames, runs VLM inference."""

import asyncio
import base64
import logging
import signal
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.vlm.inference.queue.jobs import SingleFrameJob
from iris.server.logging_handler import WebSocketLogHandler
from iris.vlm.inference.queue.queue import InferenceQueue
from iris.vlm.models import load_model_and_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ServerConfig()

# Initialize log streaming handler if enabled
log_streaming_handler: WebSocketLogHandler | None = None
if config.enable_log_streaming:
    log_streaming_handler = WebSocketLogHandler(min_level=config.log_streaming_min_level)
    # Add to root logger to capture all logs
    logging.getLogger().addHandler(log_streaming_handler)
    logger.info("Log streaming enabled (min level: %s)", config.log_streaming_min_level)

# Graceful shutdown management
shutdown_event = asyncio.Event()
force_shutdown_event = asyncio.Event()
shutdown_count = 0


def handle_shutdown_signal(signum: int, frame: any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global shutdown_count
    shutdown_count += 1

    if shutdown_count == 1:
        logger.info(
            "Received signal %s. Initiating graceful shutdown (waiting for in-flight jobs, timeout: %.1fs). "
            "Send signal again to force shutdown.",
            signal.Signals(signum).name,
            config.graceful_shutdown_timeout,
        )
        shutdown_event.set()
    else:
        logger.warning(
            "Received signal %s again. Force shutdown initiated. Queued jobs will be terminated.",
            signal.Signals(signum).name,
        )
        force_shutdown_event.set()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown."""
    # Startup
    state = get_server_state()

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    logger.info("Signal handlers registered for graceful shutdown")

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown."""
    # Startup
    state = get_server_state()

    logger.info("Loading model...")

    state.model, state.processor = load_model_and_processor(config.model_key)

    logger.info("Starting inference queue...")
    state.queue = InferenceQueue(
        max_queue_size=config.max_queue_size, num_workers=config.num_workers
    )
    await state.queue.start()

    # Initialize metrics collector if enabled
    if config.enable_metrics:
        from iris.server.metrics import MetricsCollector
        state.metrics = MetricsCollector(
            persist=True,
            log_dir="logs/metrics",
            collect_gpu_metrics=True,
        )
        logger.info("Metrics collection enabled")

    state.model_loaded = True
    logger.info("Server ready!")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Close metrics collector
    if state.metrics:
        state.metrics.close()
        logger.info("Metrics collector closed")

    if state.queue:
        if force_shutdown_event.is_set():
            logger.warning("Force shutdown - terminating all jobs immediately")
            await state.queue.stop()
        else:
            logger.info("Graceful shutdown - waiting for in-flight jobs (timeout: %.1fs)", config.graceful_shutdown_timeout)
            try:
                await asyncio.wait_for(
                    state.queue.stop(),
                    timeout=config.graceful_shutdown_timeout
                )
                logger.info("All in-flight jobs completed successfully")
            except TimeoutError:
                logger.warning(
                    "Graceful shutdown timeout (%.1fs) exceeded. Some jobs may be incomplete.",
                    config.graceful_shutdown_timeout
                )
    logger.info("Server stopped.")


app = FastAPI(title="IRIS Inference Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str | bool]:
    """Health check endpoint."""
    state = get_server_state()
    return {
        "status": "healthy" if state.model_loaded else "loading",
        "model_loaded": state.model_loaded,
    }


@app.websocket("/ws/stream")
async def inference_endpoint(websocket: WebSocket) -> None:
    """Receive frames and return inference results."""
    await websocket.accept()
    state = get_server_state()
    logger.info("Client connected")

    # --- CONFIGURATION ---
    video_batch_size = 16  # How many frames to accumulate before running inference
    frame_skip = 60  # Only keep every Nth frame to save memory/compute
    # ---------------------

    # Producer loop which receives frames and pushes them to the queue
    async def receive_loop() -> None:
        frame_buffer: list[Image.Image] = []
        buffer_start_time = 0.0
        frame_counter = 0
        try:
            while True:
                data = await websocket.receive_json()
                arrival_time = time.time()

                frame_b64 = data["frame"]
                frame_id = data["frame_id"]

                image_data = base64.b64decode(frame_b64)
                image = Image.open(BytesIO(image_data))

                # Accumulation logic
                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    continue  # Skip this frame

                if not frame_buffer:
                    buffer_start_time = arrival_time  # noqa: F841

                frame_buffer.append(image)

                if len(frame_buffer) >= video_batch_size:
                    logger.info(
                        "Accumulated %d frames. Submitting job", len(frame_buffer)
                    )

                    curr_prompt = "Describe what you see in one sentence. Describe the colors, and the expressions of people in detail"

                    job = SingleFrameJob(
                        job_id=f"frame-{frame_id}",
                        frame=image,
                        model=state.model,
                        processor=state.processor,
                        prompt=curr_prompt,
                        executor=state.queue.executor,
                        received_at=arrival_time,
                    )

                    frame_buffer.clear()

                    # Submit without waiting for the result
                    submitted = await state.queue.submit(job)
                    if not submitted:
                        logger.warning("Dropped frame %s, queue full", frame_id)

        except WebSocketDisconnect:
            logger.info("Client disconnected (Receive Loop)")
        except Exception as e:
            logger.error("Receive loop error: %s", e, exc_info=True)

    # Consumer loop which watches queue result and sends them to the client
    async def send_loop() -> None:
        try:
            while True:
                # Wait specifically for the NEXT available result
                # accessing the internal results queue directly
                result_job = await state.queue.results.get()

                response = {
                    "job_id": result_job.job_id,
                    "status": result_job.status.value,
                    "result": result_job.result,
                    "metrics": {
                        "inference_time": result_job.processing_time,
                        "total_latency": result_job.total_latency,
                        "received_at": result_job.received_at,
                        # "frames_processed": getattr(result_job, "frames", 1)
                        # if isinstance(result, SingleFrameJob) else len(result.frames)
                    },
                }

                await websocket.send_json(response)
                state.queue.results.task_done()

                # Record job metrics (if applicable)
                if state.metrics and hasattr(result_job, 'processing_time'):
                    total_latency = getattr(result_job, 'total_latency', result_job.processing_time)
                    state.metrics.record_job(
                        job_id=result_job.job_id,
                        inference_time=result_job.processing_time,
                        total_latency=total_latency,
                        status=result_job.status.value,
                        queue_depth=state.queue.queue.qsize(),
                    )

        except WebSocketDisconnect:
            logger.info("Client disconnected (Send Loop)")
        except Exception as e:
            logger.error("Send loop error: %s", e, exc_info=True)

    # Run both functions concurrently
    # This runs until one of them finishes (usually the receive loop on disconnect)
    await asyncio.gather(receive_loop(), send_loop())


@app.websocket("/ws/logs")
async def log_streaming_endpoint(websocket: WebSocket) -> None:
    """Stream server logs to connected clients."""
    if not config.enable_log_streaming or log_streaming_handler is None:
        await websocket.close(code=1008, reason="Log streaming is disabled")
        return

    await websocket.accept()
    log_streaming_handler.add_connection(websocket)
    logger.info("Log streaming client connected (total: %d)", log_streaming_handler.get_connection_count())

    try:
        # Keep connection alive and wait for client disconnect
        while True:
            # Receive ping/pong to detect disconnection
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text('{"type": "ping"}')
    except WebSocketDisconnect:
        logger.info("Log streaming client disconnected")
    except Exception as e:
        logger.error("Log streaming error: %s", e, exc_info=True)
    finally:
        log_streaming_handler.remove_connection(websocket)
        logger.info("Log streaming client removed (remaining: %d)", log_streaming_handler.get_connection_count())


@app.get("/metrics")
async def metrics_endpoint() -> dict[str, any]:
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


def main() -> None:
    """Entry point for server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

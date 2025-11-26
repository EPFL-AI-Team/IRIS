"""IRIS Inference Server - receives frames, runs VLM inference."""

import base64
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from iris.server.config import ServerConfig
from iris.server.dependencies import get_server_state
from iris.vlm.inference.queue.jobs import SingleFrameJob
from iris.vlm.inference.queue.queue import InferenceQueue
from iris.vlm.models import load_model_and_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = ServerConfig()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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

    state.model_loaded = True
    logger.info("Server ready!")

    yield

    # Shutdown
    if state.queue:
        await state.queue.stop()
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

    try:
        while True:
            data = await websocket.receive_json()
            received_at = time.time()

            frame_b64 = data["frame"]
            frame_id = data["frame_id"]

            image_data = base64.b64decode(frame_b64)
            image = Image.open(BytesIO(image_data))

            job = SingleFrameJob(
                job_id=f"frame-{frame_id}",
                frame=image,
                model=state.model,
                processor=state.processor,
                prompt="Describe what you see in one sentence.",
                executor=state.queue.executor,
            )

            await state.queue.submit(job)
            result_job = await state.queue.get_result(timeout=30.0)

            if result_job:
                total_latency = time.time() - received_at
                await websocket.send_json({
                    "job_id": result_job.job_id,
                    "status": result_job.status.value,
                    "result": result_job.result,
                    "frame": frame_b64,
                    "metrics": {
                        "inference_time": result_job.processing_time,
                        "total_latency": total_latency,
                        "received_at": received_at,
                    },
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def main() -> None:
    """Entry point for server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

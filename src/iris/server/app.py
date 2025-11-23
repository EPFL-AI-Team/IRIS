"""IRIS Inference Server - receives frames, runs VLM inference."""

import base64
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from io import BytesIO
import time
import asyncio

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
    
    # --- CONFIGURATION ---
    VIDEO_BATCH_SIZE = 16  # How many frames to accumulate before running inference
    FRAME_SKIP = 60         # Only keep every Nth frame to save memory/compute
    # ---------------------
    
    # Producer loop which receives frames and pushes them to the queue
    async def receive_loop():
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
                if frame_counter % FRAME_SKIP != 0:
                    continue # Skip this frame

                if not frame_buffer:
                    buffer_start_time = arrival_time
                
                frame_buffer.append(image)
                
                if len(frame_buffer) >= VIDEO_BATCH_SIZE:
                    logger.info(f"Accumulated {len(frame_buffer)} frames. Submitting job")
                
                
                
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
                        logger.warning(f"Dropped frame {frame_id}, queue full")
        
        except WebSocketDisconnect:
            logger.info("Client disconnected (Receive Loop)")
        except Exception as e:
            logger.error(f"Receive loop error: {e}", exc_info=True)

    # Consumer loop which watches queue result and sends them to the client
    async def send_loop():
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
                        # "frames_processed": getattr(result_job, "frames", 1) if isinstance(result, SingleFrameJob) else len(result.frames)
                    }
                }
                
                await websocket.send_json(response)
                state.queue.results.task_done()
                
        except WebSocketDisconnect:
            logger.info("Client disconnected (Send Loop)")
        except Exception as e:
            logger.error(f"Send loop error: {e}", exc_info=True)
    
    # Run both functions concurrently
    # This runs until one of them finishes (usually the receive loop on disconnect)
    await asyncio.gather(receive_loop(), send_loop())


def main() -> None:
    """Entry point for server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()

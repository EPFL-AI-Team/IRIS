import asyncio
import logging
import time
from typing import NoReturn

import cv2
from PIL import Image

from iris.server.inference.executor import InferenceExecutor
from iris.server.inference.jobs import Job, JobStatus, SingleFrameJob
from iris.vlm.models import load_model_and_processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set to DEBUG to see device info, INFO for normal operation
logger = logging.getLogger(__name__)

# CURRENT_MODEL_KEY = "smolvlm"
CURRENT_MODEL_KEY = "smolvlm2"
# CURRENT_MODEL_KEY = "qwen3-2b"
BATCH_SIZE = 8  # How many frames to grab for a job
INFERENCE_INTERVAL = 120  # Frames to wait before triggering


def load_camera_source(device_id: int = 1) -> cv2.VideoCapture:
    """Initialize video capture from camera."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise OSError("Cannot open webcam")
    return cap


async def result_consumer(queue: InferenceExecutor) -> NoReturn:
    """
    A separate, concurrent task that just waits for results and logs them.
    It also inspects the job object to provide formatted logging.
    """
    logger.info("Result consumer started.")
    while True:
        # Wait for a result to appear in the 'outbox'
        job: Job | None = await queue.get_result()
        if not job:
            continue

        if job.status == JobStatus.COMPLETED:
            # Ask the job object to format its own result
            formatted_output = job.format_result()
            logger.info(formatted_output)
            with open("inference_results.txt", "a", encoding="utf-8") as f:
                f.write(f"{formatted_output}\n\n")

        elif job.status == JobStatus.FAILED:
            # We can still have a standard format for failures
            log_message = (
                f"\n\nJob Failed: {job.job_id} ({job.job_type})\n"
                f"----------------------------------------\n"
                f"  - Error: {job.error}\n"
                f"----------------------------------------\n"
            )
            with open("inference_results.txt", "a", encoding="utf-8") as f:
                f.write(f"{log_message}\n\n")
            logger.error(log_message)

        # Give control back to the event loop
        await asyncio.sleep(0.01)


# async def main() -> None:
async def main() -> None:
    """Main execution loop for live video analysis."""
    logger.info("Loading model...")
    model, processor = load_model_and_processor(CURRENT_MODEL_KEY)

    # Executor initializing
    queue: InferenceExecutor = InferenceExecutor(max_queue_size=10, num_workers=1)
    await queue.start()

    # Start result consumer
    result_task = asyncio.create_task(result_consumer(queue))

    cap = load_camera_source()
    frame_count = 0
    PROMPT = """Describe what you see in the foreground in one sentence.
If you see a person with glasses, respond EXACTLY with:
TOOL_CALL: {"tool_name": "person_item_detector", "params": {"item": "water_bottle", "person_present": true}}
Otherwise, describe normally.
"""
    # PROMPT = "Describe what you see in the foreground in one sentence"
    frame_buffer: list[Image.Image] = []

    logger.info("Starting video stream...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            cv2.imshow(
                "IRIS Live Stream - Press Q to quit", frame
            )  # Display live video feed

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_buffer.append(pil_image)

            # Non-blocking job submission (to the queue)
            if frame_count % INFERENCE_INTERVAL == 0 and frame_buffer:
                # if frame_count == INFERENCE_INTERVAL and frame_buffer:
                logger.info(f"MAIN: Triggering job for frame {frame_count}")

                # Get the most recent frame for analysis
                image_for_job = frame_buffer[-1]

                job = SingleFrameJob(
                    job_id=f"frame-{frame_count}",
                    frame=image_for_job,
                    model=model,
                    processor=processor,
                    prompt=PROMPT,
                    executor=queue.executor,
                    received_at=time.time(),
                )

                # Non-blocking, it returns immediately
                await queue.submit(job)
                logger.info(f"MAIN: Job {job.job_id} submitted to queue.")

                frame_buffer.clear()

            # Give asyncio a chance to run other tasks (like the worker)
            await asyncio.sleep(0.001)
    except KeyboardInterrupt:
        logger.info("User pressed Ctrl+C. Shutting down.")
    finally:
        # Graceful shutdown
        logger.info("Cleaning up...")
        result_task.cancel()  # Stop the result logger
        await queue.stop()  # Stop the inference queue
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


def cli() -> None:
    """CLI entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program stopped by user.")


if __name__ == "__main__":
    cli()

"""Video inference job.

This module contains the VideoJob class for processing batches of video frames
through VLM inference. VideoJob is a server-side construct that:

- Takes pre-buffered frames at construction time (one-shot processing)
- Runs inference in a ThreadPoolExecutor thread to avoid blocking
- Uses callbacks (log_callback, result_callback) to communicate results
  back to the WebSocket handler

The job supports both Qwen and SmolVLM model families with automatic
detection and appropriate inference paths.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from PIL import Image

from iris.server.inference.jobs.base import Job, JobStatus

logger = logging.getLogger(__name__)


class VideoJob(Job):
    """One-shot batch processor for video inference.

    Processes a pre-buffered batch of frames through VLM inference and completes.
    Each VideoJob handles one batch (typically 8 frames) and then exits.
    Frame buffering is handled by the WebSocket handler or FrameBuffer, not by VideoJob.
    """

    def __init__(
        self,
        job_id: str,
        model: Any | None,
        processor: Any | None,
        executor: ThreadPoolExecutor,
        frames: list[Image.Image],
        prompt: str = """<video>\nAnalyze this video segment of a biological experiment. Output a valid JSON object with the following keys:\n- action: The specific movement (e.g., streaking, inspecting).\n- tool: The active instrument.\n- target: The object being acted upon.\n- context: The protocol step.\n- hand: Active hand (left/right).\n- sample_id: Any handwritten text or labels visible on the object (or "none").\n- notes: Visual details like agar color or colony presence.""",
        buffer_size: int = 8,
        overlap_frames: int = 4,
        default_fps: float = 5.0,
        max_new_tokens: int = 128,
        client_fps: float | None = None,
    ):
        super().__init__(job_id)
        self.model = model  # May be None until worker injects
        self.processor = processor  # May be None until worker injects
        self.executor = executor
        self.prompt = prompt
        self._process_vision_info = None  # Lazy-loaded when Qwen is detected

        # Configuration (kept for status reporting)
        self.buffer_size = buffer_size
        self.overlap_frames = overlap_frames
        self.default_fps = float(default_fps)
        self.client_fps = (
            float(client_fps) if client_fps is not None else self.default_fps
        )
        self.max_new_tokens = max_new_tokens

        # Pre-buffered frames to process (passed in at construction)
        self.frame_buffer: list[Image.Image] = frames.copy() if frames else []

        # Callbacks for server-side communication
        self.log_callback: Any = None  # Callable[[dict], None]
        self.result_callback: Any = (
            None  # Callable[[dict], None] - for real-time result broadcasting
        )

    async def execute(self) -> None:
        """One-shot execution: process the buffered frames and complete.

        This job processes the frames passed in at construction time,
        runs inference once, and then completes. No infinite loop.
        """
        self.status = JobStatus.RUNNING
        self.started_at = time.time()

        # Validate that model and processor have been injected by worker
        if self.model is None or self.processor is None:
            error_msg = f"[{self.job_id}] Model/processor not injected by worker. This indicates a configuration error."
            logger.error(error_msg)
            self.status = JobStatus.FAILED
            self.error = error_msg
            raise RuntimeError(error_msg)

        frame_count = len(self.frame_buffer)
        logger.info(f"[{self.job_id}] VideoJob starting with {frame_count} frames")

        # Lazy-load Qwen utils if needed (now that model is available)
        if self._is_qwen_model() and self._process_vision_info is None:
            try:
                from qwen_vl_utils import process_vision_info

                self._process_vision_info = process_vision_info
            except ModuleNotFoundError as exc:
                error_msg = "qwen_vl_utils (and its torchvision dependency) is required when using Qwen models."
                logger.error(f"[{self.job_id}] {error_msg}")
                self.status = JobStatus.FAILED
                self.error = error_msg
                raise ModuleNotFoundError(error_msg) from exc

        # Process frames
        await self._run_inference()

        # Mark as completed
        self.status = JobStatus.COMPLETED
        logger.info(f"[{self.job_id}] VideoJob completed")

    async def _run_inference(self) -> None:
        """Run inference on buffered frames in ThreadPoolExecutor."""
        if not self.frame_buffer:
            return

        frames_to_process = self.frame_buffer.copy()
        frame_count = len(frames_to_process)

        # Log the batch details
        self._send_log(
            f"Starting inference: {frame_count} frames | buffer={self.buffer_size} overlap={self.overlap_frames}"
        )

        # Track timing
        start_time = time.time()

        logger.info(f"[{self.job_id}] Running inference on {frame_count} frames")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._sync_inference, frames_to_process, self.prompt
        )

        # Calculate timing
        inference_time = time.time() - start_time

        # Update state
        self.result = result
        self.processing_time = inference_time

        frames_per_second = frame_count / inference_time if inference_time > 0 else 0.0
        summary = (
            f"Inference complete | frames={frame_count} "
            f"time={inference_time:.3f}s fps={frames_per_second:.2f}"
        )

        logger.info(f"[{self.job_id}] {summary}")
        self._send_log(summary)

        # Broadcast result immediately via callback
        if self.result_callback:
            result_data = {
                "type": "result",
                "job_id": self.job_id,
                "job_type": self.job_type,
                "status": "completed",
                "result": result,
                "frames_processed": frame_count,
                "inference_time": inference_time,
                "buffer_size": self.buffer_size,
                "overlap_frames": self.overlap_frames,
                "client_fps": self.client_fps,
                "sample_fps": self.client_fps,
                "timestamp": time.time(),
            }
            self.result_callback(result_data)

    def _sync_inference(self, frames: list[Image.Image], prompt: str) -> str:
        """Blocking GPU inference (runs in ThreadPoolExecutor)."""
        if not frames:
            return "No frames to process"

        # Simple dispatch based on model type
        if self._is_qwen_model():
            return self._inference_qwen(frames, prompt)
        else:
            return self._inference_smolvlm(frames, prompt)

    def _is_qwen_model(self) -> bool:
        """Simple check to see if we are using a Qwen model."""
        if self.model is None:
            return False
        model_type = getattr(self.model.config, "model_type", "").lower()
        return "qwen" in model_type

    def _inference_qwen(self, frames: list[Image.Image], prompt: str) -> str:
        """Original Qwen 2.5 VL logic."""
        logger.info(f"WORKER: Running Qwen inference on {len(frames)} frames")

        if not self._process_vision_info:
            raise ModuleNotFoundError(
                "qwen_vl_utils (and its torchvision dependency) is required when using Qwen models."
            )

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                            "sample_fps": self.client_fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Qwen-specific utils
            image_inputs, video_inputs, video_kwargs = self._process_vision_info(
                messages, return_video_kwargs=True
            )

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                **(video_kwargs or {}),
            ).to(self.model.device) # pyright: ignore[reportOptionalCall]

            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids = outputs[0][len(inputs.input_ids[0]) :]
            result = self.processor.decode(generated_ids, skip_special_tokens=True)

        return result

    def _inference_smolvlm(self, frames: list[Image.Image], prompt: str) -> str:
        """SmolVLM2 inference using VideoMetadata (avoids temp file and dtype issues)."""
        from transformers.video_utils import VideoMetadata

        logger.info(f"WORKER: Running SmolVLM inference on {len(frames)} frames")

        if not frames:
            return "No frames to process"

        # Create metadata for our pre-sampled frames
        video_metadata = VideoMetadata(
            total_num_frames=len(frames),
            fps=self.client_fps,
            duration=len(frames) / self.client_fps,
        )

        # Pass PIL Images directly (not a video path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                messages,
                video_metadata=[video_metadata],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device=self.model.device, dtype=self.model.dtype)

            # Generate
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

            # Decode
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return result

    def _send_log(self, message: str) -> None:
        """Send WebSocket log message."""
        if self.log_callback:
            self.log_callback({
                "type": "log",
                "job_id": self.job_id,
                "message": message,
                "timestamp": time.time(),
            })

    def get_status_dict(self) -> dict:
        """Return custom status information."""
        return {
            "frames_buffered": len(self.frame_buffer),
            "buffer_size": self.buffer_size,
            "overlap_frames": self.overlap_frames,
            "client_fps": self.client_fps,
            "default_fps": self.default_fps,
        }

    def to_response_dict(self) -> dict:
        """Serialize VideoJob data for WebSocket response."""
        response = {
            "result": self.result,
            "frames_processed": len(self.frame_buffer),
        }

        # Add metrics if available (after inference has run)
        if self.processing_time is not None:
            response["metrics"] = {
                "inference_time": self.processing_time,
                "total_latency": self.processing_time,
            }
            if self.started_at:
                response["metrics"]["received_at"] = self.started_at

        return response

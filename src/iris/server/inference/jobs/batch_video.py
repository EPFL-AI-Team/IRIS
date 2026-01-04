"""Batch video inference job - processes multiple segments in one GPU call."""

import asyncio
import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any

import torch
from PIL import Image

from iris.server.inference.jobs.base import Job, JobStatus

logger = logging.getLogger(__name__)


class BatchVideoJob(Job):
    """Process multiple video segments in a single GPU batch.

    Each segment has:
    - frames: List of frame bytes for that segment
    - segment_id: Unique identifier for tracking results
    - prompt: Inference prompt (usually same across all segments)
    """

    def __init__(
        self,
        job_id: str,
        model: Any | None,
        processor: Any | None,
        executor: ThreadPoolExecutor,
        segments: list[dict],  # [{"frames": list[bytes], "segment_id": str, "prompt": str, "client_fps": float}, ...]
        max_new_tokens: int = 128,
    ):
        super().__init__(job_id)
        self.model = model
        self.processor = processor
        self.executor = executor
        self.segments = segments
        self.max_new_tokens = max_new_tokens
        self._process_vision_info = None  # Lazy-loaded for Qwen

        self.log_callback: Any = None
        self.result_callback: Any = None

    async def execute(self) -> None:
        """Execute batch inference on all segments."""
        self.status = JobStatus.RUNNING
        self.started_at = time.time()

        if self.model is None or self.processor is None:
            error_msg = f"[{self.job_id}] Model/processor not injected"
            logger.error(error_msg)
            self.status = JobStatus.FAILED
            self.error = error_msg
            raise RuntimeError(error_msg)

        num_segments = len(self.segments)
        total_frames = sum(len(seg["frames"]) for seg in self.segments)
        logger.info(
            f"[{self.job_id}] BatchVideoJob starting: {num_segments} segments, {total_frames} total frames"
        )

        # Lazy-load Qwen utils
        if self._is_qwen_model() and self._process_vision_info is None:
            try:
                from qwen_vl_utils import process_vision_info

                self._process_vision_info = process_vision_info
            except ModuleNotFoundError as exc:
                error_msg = "qwen_vl_utils required for Qwen models"
                logger.error(f"[{self.job_id}] {error_msg}")
                self.status = JobStatus.FAILED
                self.error = error_msg
                raise ModuleNotFoundError(error_msg) from exc

        await self._run_batch_inference()

        self.status = JobStatus.COMPLETED
        logger.info(f"[{self.job_id}] BatchVideoJob completed")

    async def _run_batch_inference(self) -> None:
        """Run batched inference on all segments."""
        self._send_log(f"Starting batch inference: {len(self.segments)} segments")

        start_time = time.time()

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor, self._sync_batch_inference
        )

        inference_time = time.time() - start_time

        # Memory cleanup
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
            gc.collect()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        else:
            gc.collect()

        summary = f"Batch inference complete | segments={len(self.segments)} time={inference_time:.3f}s"
        logger.info(f"[{self.job_id}] {summary}")
        self._send_log(summary)

        # Log each individual segment result for comparison
        for i, (segment, result_text) in enumerate(zip(self.segments, results)):
            logger.info(
                f"[{self.job_id}] Segment {i+1}/{len(self.segments)}: "
                f"frames={len(segment['frames'])}, "
                f"inference_time={inference_time/len(self.segments):.3f}s (share of batch), "
                f"batch_total_time={inference_time:.3f}s"
            )

        # Send individual results via callback
        if self.result_callback:
            for segment, result_text in zip(self.segments, results):
                result_data = {
                    "type": "result",
                    "job_id": f"{self.job_id}_{segment['segment_id']}",
                    "job_type": "BatchVideoJob",
                    "status": "completed",
                    "result": result_text,
                    "frames_processed": len(segment["frames"]),
                    "inference_time": inference_time / len(self.segments),  # Per-segment share
                    "batch_size": len(self.segments),
                    "batch_inference_time": inference_time,  # Total batch time
                    "segment_inference_time": inference_time / len(self.segments),
                    "segment_id": segment["segment_id"],
                    "client_fps": segment.get("client_fps", 5.0),
                    "sample_fps": segment.get("client_fps", 5.0),
                    "timestamp": time.time(),
                }
                self.result_callback(result_data)

    def _sync_batch_inference(self) -> list[str]:
        """Blocking batch GPU inference (runs in ThreadPoolExecutor)."""
        # Decode all frames for all segments
        all_frames = []
        for seg in self.segments:
            frames = []
            for frame_bytes in seg["frames"]:
                try:
                    img = Image.open(BytesIO(frame_bytes)).convert("RGB")
                    img.thumbnail((640, 640))
                    frames.append(img)
                except Exception as e:
                    logger.error(f"Failed to decode frame: {e}")
                    frames.append(None)
            all_frames.append([f for f in frames if f is not None])

        # Dispatch based on model
        if self._is_qwen_model():
            return self._batch_inference_qwen(all_frames)
        else:
            return self._batch_inference_smolvlm(all_frames)

    def _is_qwen_model(self) -> bool:
        """Check if using Qwen model."""
        if self.model is None:
            return False
        model_type = getattr(self.model.config, "model_type", "").lower()
        return "qwen" in model_type

    def _batch_inference_qwen(self, all_frames: list[list[Image.Image]]) -> list[str]:
        """Qwen batch inference using official syntax."""
        logger.info(f"WORKER: Running Qwen batch inference on {len(all_frames)} segments")

        if not self._process_vision_info:
            raise ModuleNotFoundError("qwen_vl_utils required for Qwen")

        # Build messages list - one message per segment
        messages = []
        for frames, seg in zip(all_frames, self.segments):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames,
                            "sample_fps": seg.get("client_fps", 5.0),
                        },
                        {"type": "text", "text": seg["prompt"]},
                    ],
                }
            ])

        with torch.no_grad():
            # Apply chat template to each message
            texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                for msg in messages
            ]

            # Process vision info for batch
            image_inputs, video_inputs = self._process_vision_info(messages)

            # Batch processing with padding
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,  # Critical for batching
                return_tensors="pt",
            ).to(self.model.device)

            # Single forward pass for all segments
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            # Trim and decode each output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return output_texts

    def _batch_inference_smolvlm(self, all_frames: list[list[Image.Image]]) -> list[str]:
        """SmolVLM batch inference (treats videos as image sequences)."""
        logger.info(f"WORKER: Running SmolVLM batch inference on {len(all_frames)} segments")

        messages = []
        all_images = []
        for frames, seg in zip(all_frames, self.segments):
            # Build content with image markers
            content = [{"type": "image"} for _ in frames]
            content.append({"type": "text", "text": seg["prompt"]})
            messages.append([{"role": "user", "content": content}])
            all_images.extend(frames)

        with torch.no_grad():
            # Apply chat template
            texts = [
                self.processor.apply_chat_template(msg, add_generation_prompt=True)
                for msg in messages
            ]

            # Process batch
            inputs = self.processor(
                text=texts,
                images=all_images,
                padding=True,
                return_tensors="pt",
            ).to(device=self.model.device, dtype=self.model.dtype)

            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

            # Decode
            output_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)

        return output_texts

    def _send_log(self, message: str) -> None:
        """Send log via callback."""
        if self.log_callback:
            self.log_callback({
                "type": "log",
                "job_id": self.job_id,
                "message": message,
                "timestamp": time.time(),
            })

    def to_response_dict(self) -> dict:
        """Serialize BatchVideoJob data for WebSocket response."""
        return {
            "batch_size": len(self.segments),
            "segments_processed": len(self.segments),
        }

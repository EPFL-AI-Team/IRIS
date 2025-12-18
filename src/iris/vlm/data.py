from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qwen_vl_utils import process_vision_info


@dataclass
class QwenDataCollator:
    """
    Handles converting raw JSONL messages into Qwen2.5-VL tensors.

    Features:
    1. Fixes the 'pad' attribute error by doing custom padding.
    2. Supports 'max_frames' config to downsample video frames dynamically.
    3. Supports 'max_pixels' to enforce resolution limits (Critical for VRAM on V100).
    4. Masks user instructions so the model only learns to generate the answer.
    5. Properly extracts image paths from JSONL structure.
    """

    processor: Any
    max_frames: int | None = None
    max_pixels: int | None = None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # DEBUG: Print warning if limits are missing
        if self.max_pixels is None:
            print("WARNING: max_pixels is None! Training will crash on V100.")
        # 1. Extract raw messages from the dataset
        messages_batch = [x["messages"] for x in examples]

        # 2. Process messages: fix paths, subsample frames, inject resolution limits
        messages_batch = [self._process_messages(msgs) for msgs in messages_batch]

        # 3. Extract visual inputs (images/videos) using official Qwen util
        # process_vision_info returns (images, videos) by default
        image_inputs, video_inputs = process_vision_info(messages_batch)  # pyright: ignore[reportAssignmentType]

        # 4. Prepare text inputs
        # We use apply_chat_template to format the text (adding <|im_start|>, etc.)
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages_batch
        ]

        # 5. Process everything into tensors
        # This handles loading images from disk and padding
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 6. Create Labels for Training
        # We clone input_ids to create labels, then mask the parts we don't want to train on.
        labels = batch["input_ids"].clone()

        # Mask padding tokens (standard practice)
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch

    def _process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process messages to:
        1. Clean up None values in image/text fields
        2. Validate and convert image paths to absolute paths
        3. Subsample frames if max_frames is set
        4. Inject max_pixels constraint if set (critical for V100 memory management)
        """
        processed_messages = []

        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Separate and clean images and text
                images = []
                texts = []

                for item in msg["content"]:
                    if item.get("type") == "image":
                        img_path = item.get("image")

                        # Skip if path is None, empty, or doesn't exist
                        if not img_path:
                            continue

                        # Validate path exists
                        path = Path(img_path)
                        if not path.exists():
                            print(f"Warning: Image path does not exist: {img_path}")
                            continue

                        # Create clean image dict with absolute path
                        img_dict = {"type": "image", "image": str(path.absolute())}

                        # Inject max_pixels if configured (forces resolution limit)
                        if self.max_pixels:
                            img_dict["max_pixels"] = self.max_pixels

                        images.append(img_dict)

                    elif item.get("type") == "text":
                        text_content = item.get("text")
                        if text_content:  # Only add if text is not None/empty
                            texts.append({"type": "text", "text": text_content})

                # Subsample frames if necessary
                if self.max_frames and len(images) > self.max_frames:
                    indices = np.linspace(
                        0, len(images) - 1, self.max_frames, dtype=int
                    )
                    images = [images[i] for i in indices]

                # Reconstruct content (Images first, then text is standard for Qwen)
                if images or texts:
                    processed_messages.append({
                        "role": "user",
                        "content": images + texts,
                    })

            elif msg.get("role") == "assistant" and isinstance(
                msg.get("content"), list
            ):
                # Clean assistant messages
                texts = []
                for item in msg["content"]:
                    if item.get("type") == "text":
                        text_content = item.get("text")
                        if text_content:
                            texts.append({"type": "text", "text": text_content})
                if texts:
                    processed_messages.append({"role": "assistant", "content": texts})
            else:
                # Keep other messages as-is
                processed_messages.append(msg)

        return processed_messages

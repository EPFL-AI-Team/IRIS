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
        
        messages_batch = [x["messages"] for x in examples]
        messages_batch = [self._process_messages(msgs) for msgs in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False
            )
            for msg in messages_batch
        ]
        
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 5. Create labels the OFFICIAL way: start masked, unmask assistant

        IGNORE_INDEX = -100
        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX)  # All masked by default

        # Token IDs (use Qwen's official values, not dynamic lookup)
        ASSISTANT_TOKEN = 77091  # <|im_start|>assistant in Qwen2.5-VL tokenizer
        EOS_TOKEN = 151645  # <|im_end|> in Qwen2.5-VL tokenizer

        # Unmask assistant responses only
        for i, seq in enumerate(input_ids):
            seq_list = seq.tolist()
            pos = 0
            while pos < len(seq_list):
                # Find assistant marker
                if seq_list[pos] == ASSISTANT_TOKEN:
                    # Assistant response starts 2 tokens after <|im_start|>assistant
                    ans_start = pos + 2
                    ans_end = ans_start

                    # Find end of assistant response (next <|im_end|>)
                    while ans_end < len(seq_list) and seq_list[ans_end] != EOS_TOKEN:
                        ans_end += 1

                    # Unmask assistant response INCLUDING the EOS token
                    if ans_end < len(seq_list):
                        labels[i, ans_start : ans_end + 1] = input_ids[
                            i, ans_start : ans_end + 1
                        ]
                        pos = ans_end
                pos += 1

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
                            img_dict["max_pixels"] = self.max_pixels  # pyright: ignore[reportArgumentType]

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

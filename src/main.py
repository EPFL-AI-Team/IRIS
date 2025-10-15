import asyncio

import cv2
import torch
from PIL import Image
from transformers import PreTrainedModel, ProcessorMixin

from inference.model_loader import load_model_and_processor

CURRENT_MODEL_KEY = "smolvlm"


def load_camera_source(device_id: int = 1) -> cv2.VideoCapture:
    """Initialize video capture from camera."""
    cap = cv2.VideoCapture(device_id)  # 0 is the default webcam
    if not cap.isOpened():
        raise OSError("Cannot open webcam")
    return cap


def analyze_frame(
    model: PreTrainedModel, processor: ProcessorMixin, image: Image.Image, prompt: str
) -> str:
    """Run vision-language model inference on a frame."""
    # Prepare the input for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_text: str = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
    return generated_text


async def main() -> None:
    """Main execution loop for live video analysis."""
    print("Loading model...")
    model, processor = load_model_and_processor(CURRENT_MODEL_KEY)

    # queue = InferenceQueue(
    #     model=model
    # )

    cap = load_camera_source()

    frame_count = 0
    INFERENCE_INTERVAL = 120
    PROMPT = "Describe what you see in the foreground in one sentence"
    frame_buffer: list[Image.Image] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Display live video feed
        cv2.imshow("IRIS Live Stream - Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_buffer.append(pil_image)

        # Run inference periodically
        if frame_count % INFERENCE_INTERVAL == 0:
            recent_frames = frame_buffer[-8:]

            print(f"Analyzing frame {frame_count} ({CURRENT_MODEL_KEY})")

            output_text = analyze_frame(model, processor, pil_image, PROMPT)

            print(f"{CURRENT_MODEL_KEY} output: {output_text}")

            frame_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()
    print("\nStream ended")


if __name__ == "__main__":
    asyncio.run(main())

import torch
from transformers import AutoProcessor
from iris.vlm.data import QwenDataCollator

# Use the same model as your config
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def test_masking():
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=600000
    )

    # Initialize your patched collator
    collator = QwenDataCollator(processor=processor, max_frames=4, max_pixels=600000)

    # Mock Data: A simple user message + image placeholder
    # Note: We don't need a real image for this logic test, just the structure
    examples = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image."},
                        # We use a dummy image path that must exist or be handled by your code
                        # Ideally point to a real small image or disable image check for this test
                        {"type": "text", "text": "\n(Image Placeholder)"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"verb": "pipette"}'}],
                },
            ]
        }
    ]

    print("Running collator...")
    # We bypass the image loading part of your collator for this dry run
    # or you can point to a real image in 'examples' above.
    # meaningful_batch = collator(examples)

    # Actually, let's manually construct the input_ids to test purely the masking logic
    # This ensures we don't need real images to verify the math.

    # 1. Create a fake batch tensor simulating: <|im_start|>user...<|im_start|>assistant...
    im_start = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant = processor.tokenizer.convert_tokens_to_ids("assistant")

    # Construct: [Start, User, ..., Start, Assistant, Answer, EOS]
    # Let's say: Start(1) + User(2) + ... + Start(1) + Assistant(1) + Answer(3)
    fake_input_ids = torch.tensor([
        [im_start, 100, 101, im_start, assistant, 200, 201, 202],  # Sample 1
        [im_start, 500, im_start, assistant, 600, 601, 0, 0],  # Sample 2 (padded)
    ])

    # Mock the batch dictionary
    batch = {"input_ids": fake_input_ids}

    # 2. RUN YOUR MASKING LOGIC (Copy-paste the specific logic block here or import if modular)
    # Since we can't easily import just the logic block, let's instantiate the collator
    # and override the batch processing to just run the masking part if possible.
    # Alternatively, just verify the logic matches:

    labels = batch["input_ids"].clone()
    IGNORE_INDEX = -100

    for i, seq in enumerate(fake_input_ids):
        # --- REPLICATING YOUR PATCHED LOGIC ---
        start_indices = (seq == im_start).nonzero(as_tuple=True)[0]
        assistant_start_index = None
        for idx in reversed(start_indices):
            if idx + 1 < len(seq) and seq[idx + 1] == assistant:
                assistant_start_index = idx + 2
                break

        if assistant_start_index is not None:
            labels[i, :assistant_start_index] = IGNORE_INDEX
        else:
            labels[i, :] = IGNORE_INDEX
        # --------------------------------------

    print("\n--- Verification Results ---")
    for i, label_seq in enumerate(labels):
        print(f"\nSample {i}:")
        decoded_visible = []
        for token_id in label_seq:
            if token_id == IGNORE_INDEX:
                decoded_visible.append("[MASK]")
            else:
                decoded_visible.append(str(token_id.item()))

        print("Masked Sequence:", " ".join(decoded_visible))

        # Validation
        if i == 0:
            # Expected: Mask Mask Mask Mask Mask 200 201 202
            # Indices:  0    1    2    3    4    5   6   7
            # "assistant" is at index 4. Masking should stop at 4 (inclusive) or 5?
            # Your code says: assistant_start_index = idx + 2
            # idx=3 (start). idx+1=4 (assistant). idx+2=5.
            # labels[:5] are masked. So indices 0,1,2,3,4 are MASKED.
            # Index 5 (Token 200) is VISIBLE.

            if labels[i][5] != IGNORE_INDEX and labels[i][4] == IGNORE_INDEX:
                print("✅ SUCCESS: User prompt is masked, Assistant answer is visible.")
            else:
                print("❌ FAILURE: Incorrect masking boundary.")


if __name__ == "__main__":
    test_masking()

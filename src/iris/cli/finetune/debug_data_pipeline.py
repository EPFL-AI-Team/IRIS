"""Test script to verify data loading and collation on CPU."""
import sys
from pathlib import Path

# Add src to path so we can import iris modules
sys.path.append(str(Path(__file__).parent / "src"))

import json
import torch
from datasets import load_dataset
from transformers import AutoProcessor

from iris.vlm.config import load_config
from iris.vlm.data import QwenDataCollator

def main():
    print(">>> 1. Loading Config")
    # Load your train config to get paths and model name
    cfg = load_config("train")
    model_name = cfg["model"]["name"]
    train_path = cfg["data"]["train_path"]
    max_frames = cfg["data"].get("max_frames", 8) # Default to 8 for test

    print(f"    Model: {model_name}")
    print(f"    Data: {train_path}")
    print(f"    Max Frames: {max_frames}")

    print("\n>>> 1.5. Inspecting First JSONL Entry")
    # Let's see what the actual data looks like
    with open(train_path, 'r') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(f"    Sample keys: {list(sample.keys())}")
        if "messages" in sample:
            print(f"    Number of messages: {len(sample['messages'])}")
            for i, msg in enumerate(sample['messages']):
                print(f"    Message {i}: role={msg.get('role')}, content_items={len(msg.get('content', []))}")
                if msg.get('role') == 'user':
                    for j, item in enumerate(msg.get('content', [])):
                        if item.get('type') == 'image':
                            img_path = item.get('image')
                            print(f"      Image {j}: path={img_path}, exists={Path(img_path).exists() if img_path else 'N/A'}")

    print("\n>>> 2. Loading Processor (This might take a minute)")
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Fix missing pad token if necessary (common Qwen issue)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = "<|endoftext|>"
        print("    Added missing pad_token to tokenizer")

    print("\n>>> 3. Loading Dataset (First 2 examples)")
    # Load just the first 2 lines to test batching
    dataset = load_dataset("json", data_files=train_path, split="train").select(range(2))
    print(f"    Loaded {len(dataset)} examples")

    print("\n>>> 4. Testing Collator")
    collator = QwenDataCollator(processor=processor, max_frames=max_frames)
    
    # Run the collator on the raw samples
    try:
        batch = collator([dataset[0], dataset[1]])
    except Exception as e:
        print(f"    [ERROR] Collator failed: {e}")
        print(f"\n    First example messages:")
        for msg in dataset[0]['messages']:
            print(f"      Role: {msg['role']}")
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    print(f"        - {item}")
        raise

    print("\n>>> 5. Inspecting Batch Tensors")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape} | dtype: {value.dtype}")
        else:
            print(f"    {key}: {type(value)}")

    # Specific checks
    print("\n>>> 6. Verification Checks")
    
    # Check pixel_values (Shape: [batch, num_patches, hidden_dim] or similar depending on Qwen version)
    if "pixel_values" in batch:
        print("    [PASS] pixel_values present (Images processed)")
    else:
        print("    [FAIL] No pixel_values found! Images were not processed.")

    # Check input_ids
    if "input_ids" in batch:
        print(f"    [PASS] input_ids shape: {batch['input_ids'].shape}")
    
    # Check labels masking
    labels = batch["labels"]
    masked_count = (labels == -100).sum().item()
    total_count = labels.numel()
    print(f"    [PASS] Labels created. Masked tokens (padding): {masked_count}/{total_count}")

    print("\nDone! If you see [PASS] above, your data pipeline is ready for GPU training.")

if __name__ == "__main__":
    main()

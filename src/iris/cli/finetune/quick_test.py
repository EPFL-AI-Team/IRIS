import json

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

checkpoint = "/scratch/iris/checkpoints/qwen3b_finebio_run8_wo_vis"
data_file = "/scratch/iris/finebio_processed/splits/train_without_vis_analysis/finebio_test.jsonl"

USE_BASE_MODEL = False

# To use the base model directly, comment the above and uncomment below:

if USE_BASE_MODEL:
    print("[DEBUG] Loading BASE model")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
    )
else:
    print(f"[DEBUG] Loading model from: {checkpoint}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )

print("[DEBUG] Model loaded.")

print(f"[DEBUG] Loading processor from: {checkpoint}")
processor = AutoProcessor.from_pretrained(checkpoint)
print("[DEBUG] Processor loaded.")
model.eval()

print(f"[DEBUG] Reading one sample from: {data_file}")
with open(data_file) as f:
    sample = json.loads(f.readline())
print("[DEBUG] Sample loaded.")

messages = sample["messages"]
print(f"[DEBUG] messages: {messages}")

print("[DEBUG] Processing vision info...")
image_inputs, _ = process_vision_info(messages)
print(f"[DEBUG] image_inputs type: {type(image_inputs)}")

user_msgs = [m for m in messages if m["role"] != "assistant"]
print(f"[DEBUG] user_msgs: {user_msgs}")

print("[DEBUG] Applying chat template...")
text = processor.apply_chat_template(
    user_msgs, tokenize=False, add_generation_prompt=True
)
print(f"[DEBUG] text: {text}")

print("[DEBUG] Preparing model inputs...")
inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to(
    model.device
)
print(f"[DEBUG] inputs keys: {list(inputs.keys())}")

print("[DEBUG] Running model.generate()...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150)
print(f"[DEBUG] outputs shape: {outputs.shape}")

print("[DEBUG] Decoding response...")
response = processor.decode(
    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
)
print("[DEBUG] Decoded response.")

print("=" * 80)
print("GENERATED OUTPUT:")
print("=" * 80)
print(response)

print(f"Generated length: {len(response)}")
print("=" * 80)

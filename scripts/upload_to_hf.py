from huggingface_hub import HfApi
import os

# Configuration
HF_USERNAME = "animarcus"
REPO_NAME = f"{HF_USERNAME}/iris-qwen2.5-vl-3b-finebio"
ADAPTER_PATH = "models/qwen3b_finebio_finetune_iris"

# Files to upload (only adapter files, not checkpoint subfolders)
FILES_TO_UPLOAD = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def main():
    api = HfApi()

    # Create repo
    print(f"Creating repo: {REPO_NAME}")
    api.create_repo(repo_id=REPO_NAME, repo_type="model", exist_ok=True)

    # Upload files
    for filename in FILES_TO_UPLOAD:
        filepath = os.path.join(ADAPTER_PATH, filename)

        if not os.path.exists(filepath):
            print(f"⚠️  Skipping {filename} (not found)")
            continue

        print(f"📤 Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=REPO_NAME,
            repo_type="model",
        )
        print(f"✅ {filename}")

    print("\n🎉 Upload complete!")
    print(f"🔗 Model page: https://huggingface.co/{REPO_NAME}")


if __name__ == "__main__":
    main()

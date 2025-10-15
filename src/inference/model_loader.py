import torch
from transformers import AutoProcessor, PreTrainedModel, ProcessorMixin

from config.models import MODEL_CONFIGS


def load_model_and_processor(key: str) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load model and processor from config key."""
    model_cfg = MODEL_CONFIGS[key]
    print(f"Loading model: {model_cfg.id}...")
    model = model_cfg.loader.from_pretrained(
        model_cfg.id,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    processor = AutoProcessor.from_pretrained(model_cfg.id)
    print("Model loaded!")
    return model, processor

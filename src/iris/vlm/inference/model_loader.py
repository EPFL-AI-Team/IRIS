import torch
from transformers import AutoProcessor, PreTrainedModel, ProcessorMixin

from iris.config.models import MODEL_CONFIGS
from iris.utils.logging import setup_logger

logger = setup_logger(__name__)


def load_model_and_processor(key: str) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load model and processor from config key."""
    model_cfg = MODEL_CONFIGS[key]
    print(f"Loading model: {model_cfg.id}...")
    model = model_cfg.loader.from_pretrained(
        model_cfg.id,
        device_map="mps",
        dtype=torch.float16,  # dtype="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_cfg.id)

    logger.info("Model loaded!")
    logger.debug(f"Device: {model.device}")
    logger.debug(f"Dtype: {model.dtype}")
    if hasattr(model, "hf_device_map"):
        logger.debug(f"Device map: {model.hf_device_map}")

    return model, processor

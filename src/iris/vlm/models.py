"""
Model loading utilities + MODEL_CONFIGS registry. Shared by train + inference.
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from iris.utils.logging import setup_logger

logger = setup_logger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a vision-language model."""

    id: Annotated[str, Field(description="HuggingFace model ID")]
    loader: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "qwen2.5-7b": ModelConfig(
        id="Qwen/Qwen2.5-VL-7B-Instruct",
        loader=Qwen2_5_VLForConditionalGeneration,
    ),
    "qwen3-2b": ModelConfig(
        id="Qwen/Qwen3-VL-2B-Instruct",
        loader=Qwen3VLForConditionalGeneration,
    ),
    "qwen2.5-3b": ModelConfig(
        id="Qwen/Qwen2.5-VL-3B-Instruct",
        loader=Qwen2_5_VLForConditionalGeneration,
    ),
    "smolvlm": ModelConfig(
        id="HuggingFaceTB/SmolVLM-500M-Instruct",
        loader=AutoModelForImageTextToText,  # type: ignore[arg-type]
    ),
    "smolvlm2": ModelConfig(
        id="HuggingFaceTB/SmolVLM2-2.2B-Instruct", loader=AutoModelForImageTextToText
    ),
}


def load_model_and_processor(key: str) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load model and processor from config key."""
    model_cfg = MODEL_CONFIGS[key]
    print(f"Loading model: {model_cfg.id}...")
    model = model_cfg.loader.from_pretrained(
        model_cfg.id,
        device_map="mps",
        dtype="auto",
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

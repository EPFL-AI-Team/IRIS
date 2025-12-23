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
    # Qwen3VLForConditionalGeneration,
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
    # "qwen3-2b": ModelConfig(
    #     id="Qwen/Qwen3-VL-2B-Instruct",
    #     loader=Qwen3VLForConditionalGeneration,
    # ),
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


def load_model_and_processor(
    model_id: str,
    hardware: str | None = None,
    model_dtype: str | None = None,
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load model and processor.

    Args:
        model_id: HuggingFace model ID or key from MODEL_CONFIGS (e.g., "qwen2.5-7b")
        hardware: Optional hardware profile (e.g., "v100", "mac") from configs/vlm/hardware/
        model_dtype: Optional dtype override ("float16", "bfloat16", "float32", "auto")

    Returns:
        Tuple of (model, processor)
    """
    # Resolve model_id if it's a key in MODEL_CONFIGS
    resolved_model_id = model_id
    if model_id in MODEL_CONFIGS:
        logger.info(f"Resolving model key '{model_id}' from MODEL_CONFIGS")
        resolved_model_id = MODEL_CONFIGS[model_id].id

    # Load hardware profile if specified
    hw_config: dict[str, Any] = {}
    if hardware:
        from iris.vlm.config import load_hardware_profile

        hw_config = load_hardware_profile(hardware)
        logger.info(f"Using hardware profile: {hardware}")

    device = hw_config.get("device", "auto")
    dtype_str = model_dtype or hw_config.get("model", {}).get("dtype", "auto")
    torch_dtype = _parse_dtype(dtype_str)
    attn_implementation = hw_config.get("model", {}).get("attn_implementation", "sdpa")
    low_cpu_mem_usage = hw_config.get("model", {}).get("low_cpu_mem_usage", True)

    # Build model loading kwargs (NOTE: transformers uses torch_dtype, not dtype)
    model_kwargs: dict[str, Any] = {
        "device_map": device,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype

    # Add quantization config if specified
    quantization_config = _build_quantization_config(hw_config.get("quantization", {}))
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    logger.info(f"Loading model: {resolved_model_id}")
    model = AutoModelForImageTextToText.from_pretrained(resolved_model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(resolved_model_id)

    _log_model_info(model)
    return model, processor


def _parse_dtype(dtype_str: str) -> Any:
    """Parse dtype string to torch dtype or 'auto'."""
    import torch

    dtype_map: dict[str, Any] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_str in dtype_map:
        return dtype_map[dtype_str]

    logger.warning(f"Unknown dtype: {dtype_str}, using 'auto'")
    return "auto"


def _build_quantization_config(quant_cfg: dict) -> Any | None:
    """Build BitsAndBytesConfig from quantization config dict."""
    if not quant_cfg:
        return None

    load_in_4bit = quant_cfg.get("load_in_4bit", False)
    load_in_8bit = quant_cfg.get("load_in_8bit", False)

    if not (load_in_4bit or load_in_8bit):
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.error("BitsAndBytes not available. Install with: pip install bitsandbytes")
        return None

    if load_in_4bit:
        import torch

        compute_dtype_str = quant_cfg.get("bnb_4bit_compute_dtype", "float16")
        compute_dtype = _parse_dtype(compute_dtype_str)
        if compute_dtype == "auto":
            compute_dtype = torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
    elif load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)

    return None


def _log_model_info(model: PreTrainedModel) -> None:
    """Log model information after loading."""
    logger.info("Model loaded!")
    logger.debug(f"Device: {model.device}")
    logger.debug(f"Dtype: {model.dtype}")
    if hasattr(model, "hf_device_map"):
        logger.debug(f"Device map: {model.hf_device_map}")

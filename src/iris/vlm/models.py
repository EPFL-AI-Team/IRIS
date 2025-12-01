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


def load_model_and_processor(
    vlm_config_name: str = "serve",
    hardware: str | None = None,
    model_key: str | None = None,
) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load model and processor using VLM config system.

    Args:
        vlm_config_name: Config name (e.g., "serve", "train") from configs/vlm/
        hardware: Optional hardware override (e.g., "mac_m3", "v100", "a100")
        model_key: Deprecated. Legacy support for MODEL_CONFIGS keys.

    Returns:
        Tuple of (model, processor)
    """
    # Legacy support: if model_key is provided, use old system
    if model_key is not None:
        logger.warning(
            f"model_key parameter is deprecated. Use vlm_config_name and hardware instead."
        )
        model_cfg = MODEL_CONFIGS[model_key]
        logger.info(f"Loading model (legacy): {model_cfg.id}")
        model = model_cfg.loader.from_pretrained(
            model_cfg.id,
            device_map="mps",
            torch_dtype="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_cfg.id)
        _log_model_info(model)
        return model, processor

    # New config-based system
    from iris.vlm.config import load_config

    cfg = load_config(vlm_config_name, hardware=hardware)

    # Extract model configuration
    model_id = cfg["model"]["model_id"]
    device = cfg.get("device", "auto")
    torch_dtype = _parse_dtype(cfg.get("model", {}).get("torch_dtype", "auto"))
    attn_implementation = cfg.get("model", {}).get("attn_implementation", "sdpa")
    low_cpu_mem_usage = cfg.get("model", {}).get("low_cpu_mem_usage", True)

    # Build model loading kwargs
    model_kwargs = {
        "device_map": device,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }

    # Add quantization config if specified
    quantization_config = _build_quantization_config(cfg.get("quantization", {}))
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    # Load model and processor
    logger.info(f"Loading model: {model_id}")
    if hardware:
        logger.info(f"Hardware profile: {hardware}")

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    _log_model_info(model)
    return model, processor


def _parse_dtype(dtype_str: str) -> str | None:
    """Parse dtype string to torch dtype."""
    import torch

    dtype_map = {
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

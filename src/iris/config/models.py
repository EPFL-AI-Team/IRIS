from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration


class ModelConfig(BaseModel):
    """Configuration for a vision-language model."""

    id: Annotated[str, Field(description="HuggingFace model ID")]
    loader: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "qwen3b": ModelConfig(
        id="Qwen/Qwen2.5-VL-3B-Instruct",
        loader=Qwen2_5_VLForConditionalGeneration,
    ),
    "smolvlm": ModelConfig(
        id="HuggingFaceTB/SmolVLM-500M-Instruct",
        loader=AutoModelForImageTextToText,  # type: ignore[arg-type]
    ),
}

"""VLM Trainer with config-driven training using Unsloth."""

from pathlib import Path
from typing import Any
import torch
from unsloth import FastVisionModel

from iris.utils.logging import setup_logger
from iris.vlm.config import load_config

logger = setup_logger(__name__)


class VLMTrainer:
    """Vision-Language Model trainer using Unsloth."""

    def __init__(
        self,
        config_name: str = "train",
        hardware: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Initialize trainer with configuration.
        
        Args:
            config_name: Base config name (e.g., "train")
            hardware: Hardware profile (e.g., "v100", "mac")
            config_overrides: Runtime overrides for config values
        """
        # Load merged config
        self.cfg = load_config(config_name, hardware=hardware)
        
        # Apply runtime overrides
        if config_overrides:
            self._merge_overrides(config_overrides)
        
        # Resolve paths relative to project root
        self.project_root = self._get_project_root()
        self._resolve_paths()
        
        logger.info(f"Initialized trainer with config: {config_name}, hardware: {hardware}")
        
    def _get_project_root(self) -> Path:
        """Get project root directory."""
        current = Path(__file__).resolve()
        return current.parent.parent.parent.parent  # src/iris/vlm/trainer.py -> root
        
    def _resolve_paths(self) -> None:
        """Resolve paths. Relative paths are relative to project root. Absolute paths are kept as is."""
        if "data" in self.cfg:
            for key in ["train_path", "val_path", "cache_dir"]:
                if key in self.cfg["data"] and self.cfg["data"][key]:
                    path = Path(self.cfg["data"][key])
                    if not path.is_absolute():
                        self.cfg["data"][key] = str(self.project_root / path)
                    # If absolute, leave it alone
        
        if "training" in self.cfg and "output_dir" in self.cfg["training"]:
            path = Path(self.cfg["training"]["output_dir"])
            if not path.is_absolute():
                self.cfg["training"]["output_dir"] = str(self.project_root / path)
                
    def _merge_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge runtime overrides into config."""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in self.cfg:
                self.cfg[key].update(value)
            else:
                self.cfg[key] = value
        
    def run(self) -> None:
        """Run the training loop using Unsloth.
        
        Implement training following Unsloth guide:
        - Load model with FastVisionModel
        - Prepare dataset
        - Set up training arguments
        - Train with SFTTrainer
        """
        logger.info("Starting Unsloth training...")
        logger.info(f"Config: {self.cfg}")
        
        # TODO: Implement Unsloth training pipeline
        # 1. Load model: FastVisionModel.from_pretrained(...)
        model, tokenizer = FastVisionModel.from_pretrained(
            "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
            load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
        )
        
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,  # False if not finetuning vision layers
            finetune_language_layers=True,  # False if not finetuning language layers
            finetune_attention_modules=True,  # False if not finetuning attention layers
            finetune_mlp_modules=True,  # False if not finetuning MLP layers
            r=16,  # The larger, the higher the accuracy, but might overfit
            lora_alpha=16,  # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )
        
        # 2. Load dataset from self.cfg["data"]["train_path"]
        
        # 3. Setup LoRA with FastVisionModel.get_peft_model(...)
        # 4. Create SFTTrainer
        # 5. trainer.train()
        # 6. Save model to self.cfg["training"]["output_dir"]
        
        raise NotImplementedError("Implement Unsloth training pipeline")

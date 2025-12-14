"""VLM Trainer with config-driven training using Unsloth."""

from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastVisionModel, is_bf16_supported

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

        logger.info(
            f"Initialized trainer with config: {config_name}, hardware: {hardware}"
        )

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
        # 1. Load model
        logger.info(f"Loading model: {self.cfg['model']['name']}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=self.cfg["model"]["name"],
            load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
        )

        # 2. Apply LoRA adapters
        logger.info("Applying LoRA adapters...")
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=self.cfg["lora"].get(
                "finetune_vision_layers", False
            ),
            finetune_language_layers=self.cfg["lora"].get(
                "finetune_language_layers", True
            ),
            finetune_attention_modules=self.cfg["lora"].get(
                "finetune_attention_modules", True
            ),
            finetune_mlp_modules=self.cfg["lora"].get("finetune_mlp_modules", True),
            r=self.cfg["lora"]["r"],
            lora_alpha=self.cfg["lora"]["alpha"],
            lora_dropout=self.cfg["lora"]["dropout"],
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # 3. Load Datasets
        logger.info(f"Loading train data: {self.cfg['data']['train_path']}")
        train_dataset = load_dataset(
            "json", data_files=self.cfg["data"]["train_path"], split="train"
        )

        eval_dataset = None
        if self.cfg["data"].get("val_path"):
            logger.info(f"Loading val data: {self.cfg['data']['val_path']}")
            eval_dataset = load_dataset(
                "json", data_files=self.cfg["data"]["val_path"], split="train"
            )

        # 4. Training Arguments
        logger.info("Configuring training arguments...")
        args = TrainingArguments(
            per_device_train_batch_size=self.cfg["training"]["batch_size"],
            gradient_accumulation_steps=self.cfg["training"][
                "gradient_accumulation_steps"
            ],
            warmup_steps=self.cfg["training"]["warmup_steps"],
            max_steps=self.cfg["training"]["max_steps"],
            learning_rate=float(self.cfg["training"]["learning_rate"]),
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=self.cfg["training"].get("logging_steps", 10),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=self.cfg["training"]["output_dir"],
            report_to="none",  # Change to "wandb" if configured
            save_strategy="steps",
            save_steps=self.cfg["training"].get("save_steps", 100),
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            remove_unused_columns=False,
        )

        # 5. Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=FastVisionModel.get_data_collator(model, tokenizer),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            max_seq_length=self.cfg["model"].get("max_seq_length", 2048),
            dataset_text_field="messages",
            dataset_kwargs={"skip_prepare_dataset": True},
        )

        # 6. Train
        logger.info("Starting training...")
        trainer_stats = trainer.train()

        # 7. Save Final Model
        output_dir = Path(self.cfg["training"]["output_dir"]) / "final_model"
        logger.info(f"Saving final model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Training complete. Stats: {trainer_stats}")

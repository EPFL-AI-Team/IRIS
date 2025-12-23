"""VLM Trainer with config-driven training using PEFT + HuggingFace Trainer."""

from pathlib import Path
from typing import Any, cast

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,  # type: ignore[reportPrivateImportUsage]
    TrainingArguments,  # type: ignore[reportPrivateImportUsage]
)

from iris.utils.logging import setup_logger
from iris.vlm.config import load_config
from iris.vlm.data import QwenDataCollator

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
        """Execute LoRA training pipeline."""
        logger.info("=" * 80)
        logger.info("Starting LoRA Fine-tuning Pipeline")
        logger.info("=" * 80)

        try:
            # Setup quantization (not needed for A100 training)
            # logger.info("Setting up 4-bit quantization")
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            # )

            # 2. Load model in fp16
            model_name = self.cfg["model"]["name"]
            max_frames = self.cfg["data"].get("max_frames")
            max_pixels = self.cfg["data"].get("max_pixels")

            if max_pixels:
                logger.info(f"Resolution limit: {max_pixels} pixels")

            logger.info(f"Loading model: {model_name}")
            
            dtype_str = self.cfg["model"].get("dtype", "float16")
            torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                # quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(
                model_name, min_pixels=256 * 28 * 28, max_pixels=max_pixels
            )
            logger.info(
                f"Model loaded. Memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
            )
            
            # Enable model gradient checkpointing
            model.gradient_checkpointing_enable()

            # Prepare QLoRA (not used anymore)
            # logger.info("Preparing model for k-bit training")
            # model = prepare_model_for_kbit_training(model)

            # 4. Setup LoRA
            lora_cfg = self.cfg["lora"]
            logger.info(
                f"Setting up LoRA with r={lora_cfg['r']}, alpha={lora_cfg['alpha']}"
            )

            # Build target modules list
            target_modules = []
            if lora_cfg.get("finetune_attention_modules", True):
                target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
            if lora_cfg.get("finetune_mlp_modules", True):
                target_modules.extend(["gate_proj", "up_proj", "down_proj"])

            peft_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                target_modules=target_modules if target_modules else "all-linear",
                lora_dropout=lora_cfg["dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

            # Print trainable params
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
            )

            # Load data
            data_cfg = self.cfg["data"]
            logger.info(f"Loading datasets from {data_cfg['train_path']}")
            train_dataset = cast(
                Dataset,
                load_dataset("json", data_files=data_cfg["train_path"], split="train"),
            )

            val_dataset = None
            if data_cfg.get("val_path"):
                val_dataset = cast(
                    Dataset,
                    load_dataset(
                        "json", data_files=data_cfg["val_path"], split="train"
                    ),
                )

            logger.info(f"Train samples: {len(train_dataset)}")
            if val_dataset:
                logger.info(f"Val samples: {len(val_dataset)}")

            # Setup logging directory (Tensorboard and WandB)
            train_cfg = self.cfg["training"]
            logging_dir = str(Path(train_cfg["output_dir"]) / "logs")
            Path(logging_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"TensorBoard logs will be saved to: {logging_dir}")
            logger.info(f"View with: uv run tensorboard --logdir={logging_dir}")

            # Data collator
            logger.info("Setting up Qwen-specific data collator")
            data_collator = QwenDataCollator(
                processor=processor,
                max_frames=max_frames,
                max_pixels=max_pixels,
            )

            # Training args
            logger.info("Configuring training arguments")
            use_fp16 = self.cfg["training"].get("fp16", False)
            use_bf16 = self.cfg["training"].get("bf16", False)

            training_args = TrainingArguments(
                output_dir=train_cfg["output_dir"],
                max_steps=train_cfg["max_steps"],
                per_device_train_batch_size=train_cfg["batch_size"],
                gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
                learning_rate=train_cfg["learning_rate"],
                warmup_steps=train_cfg["warmup_steps"],
                lr_scheduler_type="cosine",
                optim=train_cfg.get("optim", "adamw_torch"),  # Regular AdamW for LoRA
                weight_decay=train_cfg.get("weight_decay", 0.01),
                fp16=use_fp16,  # Should be True for V100
                bf16=use_bf16,  # Should be False for V100
                gradient_checkpointing=False,
                logging_dir=logging_dir,
                logging_steps=train_cfg["logging_steps"],
                save_steps=train_cfg["save_steps"],
                save_strategy="steps",
                save_total_limit=3,
                eval_strategy="steps" if val_dataset else "no",
                eval_steps=train_cfg.get("eval_steps", 10), # Default to 10 if missing
                per_device_eval_batch_size=train_cfg.get("batch_size", 1),
                remove_unused_columns=False,
                dataloader_pin_memory=True,
                seed=42,
                report_to=["tensorboard", "wandb"],
            )

            # Train
            logger.info("Initializing HuggingFace Trainer")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                processing_class=processor,
            )

            logger.info("Starting training...")
            trainer.train()

            # Save
            logger.info(
                "Training complete! Saving model to %s", training_args.output_dir
            )
            trainer.save_model(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)

            logger.info("=" * 80)
            logger.info("Training Complete!")
            logger.info(f"Model saved to: {training_args.output_dir}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

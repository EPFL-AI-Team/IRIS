"""CLI entrypoint for VLM training."""

import argparse

from iris.utils.logging import setup_logger
from iris.vlm.trainer import VLMTrainer

logger = setup_logger(__name__)


def main() -> None:
    """Train VLM model with config-driven setup."""
    parser = argparse.ArgumentParser(description="Train VLM model using Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="train",
        help="Config name from configs/vlm/ (default: train)",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help="Hardware profile: v100, mac, etc. (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override training data path",
    )
    
    args = parser.parse_args()
    
    # Build runtime overrides from CLI args
    overrides = {}
    if args.output_dir:
        overrides["training"] = {"output_dir": args.output_dir}
    if args.data_path:
        overrides["data"] = {"train_path": args.data_path}
    
    # Create and run trainer
    trainer = VLMTrainer(
        config_name=args.config,
        hardware=args.hardware,
        config_overrides=overrides if overrides else None,
    )
    
    trainer.run()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

"""CLI entrypoint for VLM training."""

import argparse
import os

import wandb

from iris.utils.logging import setup_logger
from iris.vlm.trainer import VLMTrainer

logger = setup_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train", help="Config name")
    parser.add_argument("--hardware", default=None, help="Hardware profile")
    parser.add_argument(
        "--wandb-project", default="iris-qwen-training", help="WandB project"
    )
    parser.add_argument("--wandb-run-name", default=None, help="WandB run name")
    args = parser.parse_args()

    trainer = VLMTrainer(
        config_name=args.config,
        hardware=args.hardware,
    )

    # Initialize WandB
    if os.getenv("WANDB_API_KEY"):
        try:
            os.environ["WANDB_LOG_MODEL"] = "checkpoint"
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name
                or f"run-{args.config}-{args.hardware or 'default'}",
                config={
                    "config_name": args.config,
                    "hardware": args.hardware,
                    **trainer.cfg,
                },
            )
            logger.info(f"WandB tracking enabled: {wandb.run.url}")
        except Exception as e:
            logger.warning(f"WandB failed: {e}. Continuing without WandB.")
    else:
        logger.warning("WANDB_API_KEY not set, skipping WandB logging")

    trainer.run()

    # Finish wandb
    if os.getenv("WANDB_API_KEY"):
        wandb.finish()


if __name__ == "__main__":
    main()

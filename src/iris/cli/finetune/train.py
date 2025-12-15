"""CLI entrypoint for VLM training."""

import argparse

from iris.utils.logging import setup_logger
from iris.vlm.trainer import VLMTrainer

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train", help="Config name")
    parser.add_argument("--hardware", default=None, help="Hardware profile")
    args = parser.parse_args()

    trainer = VLMTrainer(
        config_name=args.config,
        hardware=args.hardware,
    )
    trainer.run()


if __name__ == "__main__":
    main()

import argparse
import logging
import os
import sys


def run():
    """Run the server with CLI argument support and proper signal handling."""
    from iris.config import load_yaml_config
    from iris.server.config import ServerConfig

    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="IRIS Video Understanding Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  IRIS_CONFIG_FILE    Path to config file (default: config.yaml)
                      Examples: config.rcp.yaml, /path/to/config.yaml

Configuration Priority:
  CLI arguments > Config file > Defaults

Queue Parameters:
  For single-user scenarios, set both to same value (e.g., 1-2).
  The system will drop frames when server can't keep up with incoming rate.

  --max-queue-size         Hard limit on queue capacity (prevents OOM)
  --live-queue-threshold   When to drop frames in live mode

  Constraint: live_queue_threshold <= max_queue_size

Examples:
  # Use alternate config file
  IRIS_CONFIG_FILE=config.rcp.yaml iris-server

  # Override model checkpoint
  iris-server --model-id /scratch/iris/checkpoints/qwen3b_finebio_finetune_iris/

  # Override port
  iris-server --port 9000

  # Combined: config file + CLI overrides
  IRIS_CONFIG_FILE=config.rcp.yaml iris-server --port 9000 --num-workers 2
        """
    )

    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HuggingFace model ID or path to local checkpoint"
    )
    parser.add_argument(
        "--vlm-hardware",
        type=str,
        choices=["v100", "mac", "a100"],
        default=None,
        help="Hardware profile for optimizations (v100, mac, a100)"
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["float16", "float32", "bfloat16", "auto"],
        default=None,
        help="Model dtype override (float16, float32, bfloat16, auto)"
    )

    # Server networking
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port number (default: 8001)"
    )

    # Queue and worker configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel inference workers (default: 1)"
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=None,
        help="Hard limit on queue capacity (default: 10)"
    )
    parser.add_argument(
        "--live-queue-threshold",
        type=int,
        default=None,
        help="When to drop frames in live mode (default: 1)"
    )

    # Batch inference configuration
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch-inference-enabled",
        action="store_true",
        default=None,
        help="Force enable batch inference, overriding config file"
    )
    batch_group.add_argument(
        "--batch-inference-disabled",
        action="store_true",
        default=None,
        help="Force disable batch inference, overriding config file"
    )
    parser.add_argument(
        "--batch-inference-size",
        type=int,
        default=None,
        help="Set batch size (implies --batch-inference-enabled)"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (can also use IRIS_CONFIG_FILE env var)"
    )

    args = parser.parse_args()

    # Load YAML configuration (respecting --config or IRIS_CONFIG_FILE)
    try:
        if args.config is not None:
            # CLI --config takes precedence over environment variable
            yaml_config = load_yaml_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            # Use IRIS_CONFIG_FILE env var or default
            config_path = os.getenv("IRIS_CONFIG_FILE", "config.yaml")
            yaml_config = load_yaml_config()
            logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create ServerConfig with CLI overrides
    try:
        new_config = ServerConfig.from_cli_args(args, yaml_config)
        print(f"Server configuration loaded successfully")
        print(f"  Model: {new_config.model_id}")
        print(f"  Host: {new_config.host}:{new_config.port}")
        print(f"  Workers: {new_config.num_workers}")
        print(f"  Queue limits: live_threshold={new_config.live_queue_threshold}, max_size={new_config.max_queue_size}")
    except ValueError as e:
        print(f"Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Patch the global config in iris.server.main before starting
    import iris.server.main
    iris.server.main.config = new_config

    # Start server with proper error handling
    try:
        from iris.server.main import main
        main()
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        print("\n\nServer startup interrupted. Exiting gracefully...", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    run()

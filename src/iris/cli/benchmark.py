"""CLI for benchmarking."""

from pathlib import Path
import sys


def run():
    """Run benchmark CLI."""
    # Import here to avoid loading transformers on every CLI call
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from iris.vlm.inference.benchmark_inference import main

    main()


if __name__ == "__main__":
    run()

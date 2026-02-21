"""CLI for benchmarking."""

from pathlib import Path
import sys


def run():
    """Run benchmark CLI."""
    # benchmark_inference.py lives in sft-vlm-finetune/vlm/inference/ after the extraction
    sft_path = str(Path(__file__).parents[3] / "sft-vlm-finetune")
    sys.path.insert(0, sft_path)
    from vlm.inference.benchmark_inference import main

    main()


if __name__ == "__main__":
    run()

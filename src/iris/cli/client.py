import argparse
import subprocess
import sys
from pathlib import Path

from iris.client.web.app import main


def build_frontend() -> None:
    frontend_dir = Path(__file__).parent.parent / "client" / "web" / "frontend"
    if not frontend_dir.exists():
        print(f"Frontend directory not found: {frontend_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Building frontend in {frontend_dir}...")
    result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir)
    if result.returncode != 0:
        print("Frontend build failed", file=sys.stderr)
        sys.exit(result.returncode)


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="IRIS client CLI")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the frontend before starting the server",
    )
    args = parser.parse_args()

    if args.build:
        build_frontend()
    main()


if __name__ == "__main__":
    main_cli()

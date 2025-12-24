import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run both React dev server and FastAPI backend concurrently."""
    frontend_dir = Path(__file__).parent / "frontend"

    # Start npm dev server in background
    npm_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
    )

    # Start FastAPI (this blocks)
    try:
        subprocess.run([sys.executable, "-m", "iris.client.web.app"])
    finally:
        npm_process.terminate()
        npm_process.wait()


if __name__ == "__main__":
    main()

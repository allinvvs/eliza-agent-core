from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

CUDA_INDEX_MAP: dict[str, str] = {
    "12.4": "https://github.io",
    "12.3": "https://github.io",
    "12.2": "https://github.io",
    "12.1": "https://github.io",
}
DEFAULT_CUDA_INDEX = "https://github.io"


def is_in_virtualenv() -> bool:
    """Checks whether the script is running inside a virtual environment."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def detect_cuda_version() -> str | None:
    """Detects NVIDIA GPU and extracts the supported CUDA version via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        output = subprocess.check_output(
            [nvidia_smi],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", output)
        return match.group(1) if match else "12.4"
    except (subprocess.SubprocessError, OSError):
        return None


def run_pip(args: list[str]) -> None:
    """Executes pip using the current Python interpreter."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def main() -> None:
    print("=== ELIZA v5 Automated Environment Setup ===")
    base_dir = Path(__file__).resolve().parent

    # 0. Guard against polluting global system Python
    if not is_in_virtualenv():
        print("\n[WARNING] You are NOT inside a virtual environment (.venv)!")
        print("Recommended: python -m venv .venv && .\\.venv\\Scripts\\Activate.ps1")
        choice = input("Continue installing into global Python anyway? [y/N]: ").strip().lower()
        if choice != "y":
            print("Setup aborted safely.")
            sys.exit(1)

    # 1. Upgrade pip and install project dependencies from pyproject.toml
    print("\n[1/3] Upgrading pip and installing base project dependencies...")
    run_pip(["--upgrade", "pip"])
    if (base_dir / "pyproject.toml").exists():
        run_pip(["-e", str(base_dir)])
    else:
        run_pip(["requests", "beautifulsoup4", "ddgs", "psutil", "pynvml"])

    # 2. Detect GPU and install optimal llama-cpp-python wheel
    cuda_ver = detect_cuda_version()
    if cuda_ver:
        index_url = CUDA_INDEX_MAP.get(cuda_ver, DEFAULT_CUDA_INDEX)
        print(f"\n[2/3] NVIDIA GPU detected (Driver CUDA {cuda_ver}). Installing CUDA wheel...")
        try:
            run_pip([
                "llama-cpp-python",
                "--upgrade",
                "--force-reinstall",
                "--extra-index-url",
                index_url,
            ])
            print(">>> CUDA acceleration configured successfully!")
        except subprocess.CalledProcessError as exc:
            print(f">>> CUDA wheel installation failed (exit code {exc.returncode}).")
            print(">>> Falling back to standard CPU build...")
            run_pip(["llama-cpp-python"])
    else:
        print("\n[2/3] No NVIDIA GPU detected. Installing standard CPU llama-cpp-python...")
        run_pip(["llama-cpp-python"])

    # 3. Prepare directories and .env config
    print("\n[3/3] Preparing directories and default configuration...")
    (base_dir / "models").mkdir(exist_ok=True)
    (base_dir / "plugins").mkdir(exist_ok=True)

    env_file = base_dir / ".env"
    env_example = base_dir / ".env.example"
    if not env_file.exists() and env_example.exists():
        shutil.copyfile(env_example, env_file)
        print(">>> Created .env from .env.example")

    print("\n=== Setup Complete! ===")
    print("Next step: Place your .gguf model into 'models/model.gguf' and run 'python main.py' or 'python server.py'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(130)


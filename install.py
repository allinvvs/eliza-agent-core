import os
import shutil
import subprocess
import sys


def has_nvidia_gpu() -> bool:
    """Checks if an NVIDIA GPU is present on the host system."""
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=3)
        return True
    except Exception:
        return False


def run_pip(args: list) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def main() -> None:
    print("=== ELIZA v5 Automated Environment Setup ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Install base dependencies
    print("\n[1/3] Installing base dependencies...")
    run_pip(["--upgrade", "pip"])
    run_pip(["requests", "beautifulsoup4", "ddgs", "psutil"])

    # 2. Detect GPU and install optimal llama-cpp-python build
    if has_nvidia_gpu():
        print("\n[2/3] NVIDIA GPU detected! Installing CUDA-accelerated llama-cpp-python...")
        try:
            cuda_index = "https://github.io"
            run_pip(["llama-cpp-python", "--force-reinstall", "--no-deps", "--extra-index-url", cuda_index])
            run_pip(["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"])
            print(">>> CUDA acceleration configured successfully!")
        except Exception as e:
            print(f">>> CUDA wheel failed ({e}), falling back to standard CPU build...")
            run_pip(["llama-cpp-python"])
    else:
        print("\n[2/3] No NVIDIA GPU detected. Installing standard CPU llama-cpp-python...")
        run_pip(["llama-cpp-python"])

    # 3. Prepare directories and .env config
    print("\n[3/3] Preparing directories and default configuration...")
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plugins"), exist_ok=True)

    env_file = os.path.join(base_dir, ".env")
    env_example = os.path.join(base_dir, ".env.example")
    if not os.path.exists(env_file) and os.path.exists(env_example):
        shutil.copyfile(env_example, env_file)
        print(">>> Created .env from .env.example")

    print("\n=== Setup Complete! ===")
    print("Next step: Place your .gguf model into 'models/model.gguf' and run 'python main.py' or 'python server.py'.")


if __name__ == "__main__":
    main()

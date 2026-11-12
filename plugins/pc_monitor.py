import platform
import shutil
import subprocess
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

PLUGIN_NAME = "Hardware & VRAM Telemetry"
TRIGGERS = ["система", "ресурсы", "нагрузка", "статус пк", "cpu", "ram", "видеокарта", "vram", "железо"]


def _get_gpu_info() -> str:
    """Queries NVIDIA GPU utilization and VRAM via nvidia-smi without extra libraries."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
        if out:
            name, mem_used, mem_total, util = [x.strip() for x in out.split(",")]
            return f"GPU: {name} (VRAM {mem_used}/{mem_total} МБ, нагрузка {util}%)"
    except Exception:
        pass
    return ""


def run(query: str = "") -> str:
    """Returns comprehensive host hardware, disk, and GPU telemetry."""
    os_info = f"{platform.system()} {platform.release()}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    disk = shutil.disk_usage(".")
    disk_free_gb = round(disk.free / (1024 ** 3), 1)

    parts = [f"ОС: {os_info}", f"Время: {now}", f"Диск свободно: {disk_free_gb} ГБ"]

    if psutil:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        parts.append(f"CPU: {cpu}%")
        parts.append(f"RAM: {round(ram.used / (1024 ** 3), 1)}/{round(ram.total / (1024 ** 3), 1)} ГБ ({ram.percent}%)")

    gpu_str = _get_gpu_info()
    if gpu_str:
        parts.append(gpu_str)

    return " | ".join(parts)


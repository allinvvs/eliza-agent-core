import atexit
import platform
import shutil
import subprocess
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    atexit.register(pynvml.nvmlShutdown)
    _NVML_AVAILABLE = True
except Exception:
    pynvml = None
    _NVML_AVAILABLE = False

PLUGIN_NAME = "Hardware & VRAM Telemetry"
TRIGGERS = ["система", "ресурсы", "нагрузка", "статус пк", "cpu", "ram", "видеокарта", "vram", "железо"]


def _get_gpu_nvml() -> str:
    """Reads NVIDIA GPU metrics directly via C-API (sub-millisecond latency)."""
    if not _NVML_AVAILABLE or pynvml is None:
        return ""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        mem_used_mb = int(mem.used // (1024 ** 2))
        mem_total_mb = int(mem.total // (1024 ** 2))
        return f"GPU: {name} (VRAM {mem_used_mb}/{mem_total_mb} МБ, нагрузка {util.gpu}%, {temp}°C)"
    except Exception:
        return ""

#Блок 2. Фоллбэк _get_gpu_smi() и основная точка входа run()

def _get_gpu_smi() -> str:
    """Fallback GPU telemetry via nvidia-smi subprocess if pynvml is unavailable."""
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


def _get_gpu_info() -> str:
    """Queries NVIDIA GPU telemetry preferring native NVML over subprocess."""
    return _get_gpu_nvml() or _get_gpu_smi()


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

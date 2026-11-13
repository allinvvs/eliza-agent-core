from __future__ import annotations

import atexit
import ctypes
import os
import platform
import shutil
import subprocess
from datetime import datetime
from functools import lru_cache

try:
    import psutil
    psutil.cpu_percent(interval=None)
except ImportError:
    psutil = None  # type: ignore[assignment]

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore[assignment]

PLUGIN_NAME = "Hardware & VRAM Telemetry"
TRIGGERS = [
    "system", "resources", "load", "pc status", "cpu", "ram", "gpu", "vram", "hardware", "time",
    "система", "ресурсы", "нагрузка", "статус пк", "видеокарта", "железо", "время",
]


@lru_cache(maxsize=1)
def _init_nvml_once() -> bool:
    """Thread-safe, lazy NVML initialization cached for the process lifetime."""
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        atexit.register(pynvml.nvmlShutdown)
        return True
    except Exception:
        return False


def _get_gpu_nvml() -> str:
    """Reads NVIDIA GPU metrics directly via C-API (< 1 ms) with per-sensor fault isolation."""
    if not _init_nvml_once() or pynvml is None:
        return ""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        raw_name = pynvml.nvmlDeviceGetName(handle)
        name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)

        metrics: list[str] = []
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.append(f"VRAM {int(mem.used // 1048576)}/{int(mem.total // 1048576)} MB")
        except Exception:
            pass

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics.append(f"load {util.gpu}%")
        except Exception:
            pass

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics.append(f"{temp}°C")
        except Exception:
            pass

        suffix = f" ({', '.join(metrics)})" if metrics else ""
        return f"GPU: {name}{suffix}"
    except Exception:
        return ""


def _get_gpu_smi() -> str:
    """Fallback GPU telemetry via nvidia-smi subprocess if pynvml is unavailable."""
    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        return ""
    try:
        out = subprocess.check_output(
            [
                smi_path,
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
        if out:
            first_line = out.splitlines()
            name, mem_used, mem_total, util, temp = [x.strip() for x in first_line.split(",")[:5]]
            return f"GPU: {name} (VRAM {mem_used}/{mem_total} MB, load {util}%, {temp}°C)"
    except Exception:
        pass
    return ""


def _get_gpu_info() -> str:
    """Queries NVIDIA GPU telemetry preferring native NVML over subprocess."""
    return _get_gpu_nvml() or _get_gpu_smi()


def _get_system_drive_root() -> str:
    """Returns the OS system drive root (critical for Swap/Pagefile space monitoring)."""
    if platform.system() == "Windows":
        sys_drive = os.environ.get("SystemDrive", "C:")
        return sys_drive if sys_drive.endswith("\\") else f"{sys_drive}\\"
    return "/"


def _get_windows_ram_fallback() -> str:
    """Zero-dependency Windows RAM & Pagefile fallback via kernel32 ctypes."""
    if platform.system() != "Windows":
        return ""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            total_gb = stat.ullTotalPhys / (1024 ** 3)
            used_gb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 ** 3)
            return f"RAM: {used_gb:.1f}/{total_gb:.1f} GB ({stat.dwMemoryLoad}%)"
    except Exception:
        pass
    return ""


def run(query: str = "") -> str:
    """Returns compact, token-efficient host hardware, system disk, time, and GPU telemetry."""
    _ = query
    os_info = f"{platform.system()} {platform.release()}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    disk_free_gb = round(shutil.disk_usage(_get_system_drive_root()).free / (1024 ** 3), 1)

    parts = [f"OS: {os_info}", f"Time: {now}", f"SysDisk free: {disk_free_gb} GB"]

    if psutil is not None:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        swap = psutil.swap_memory()
        parts.append(f"CPU: {cpu}%")
        parts.append(f"RAM: {ram.used / (1024 ** 3):.1f}/{ram.total / (1024 ** 3):.1f} GB ({ram.percent}%)")
        if swap.total > 0 and swap.percent > 10.0:
            parts.append(f"Swap: {swap.used / (1024 ** 3):.1f}/{swap.total / (1024 ** 3):.1f} GB")
    else:
        win_ram = _get_windows_ram_fallback()
        if win_ram:
            parts.append(win_ram)

    gpu_str = _get_gpu_info()
    if gpu_str:
        parts.append(gpu_str)

    return " | ".join(parts)


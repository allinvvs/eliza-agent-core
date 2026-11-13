"""
Eliza Zero-Trust Micro-Sandbox
Единый модуль безопасного исполнения (Windows / Linux / future WSL2)

Философия:
- Default Deny
- Математически доказуемый path containment
- Минимальная поверхность атаки
- Структурированный результат для LLM
"""

from __future__ import annotations

import ast
import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

@dataclass(frozen=True)
class SandboxConfig:
    max_exec_timeout: float = 5.0
    max_output_bytes: int = 10_000
    max_memory_mb: int = 128

DEFAULT_CONFIG = SandboxConfig()

from pathlib import Path
from typing import Final

logger = logging.getLogger("eliza.sandbox")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SandboxConfig:
    workspace_name: str = "sandbox_workspace"
    max_exec_timeout: float = 2.0
    max_output_chars: int = 1500
    max_file_bytes: int = 32 * 1024
    max_code_chars: int = 1000
    allowed_text_exts: frozenset[str] = frozenset({".txt", ".md", ".json", ".csv", ".log"})

CONFIG = SandboxConfig()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_ROOT = (PROJECT_ROOT / CONFIG.workspace_name).resolve()

IS_WINDOWS = platform.system() == "Windows"

# ──────────────────────────────────────────────
# Result type (единый контракт)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SandboxOutcome:
    allowed: bool
    category: str          # "ok" | "security_block" | "runtime_error"
    message: str

    def format_for_llm(self) -> str:
        prefix = {
            "ok": "[Sandbox]",
            "security_block": "[Sandbox Security]",
            "runtime_error": "[Sandbox Error]",
        }.get(self.category, "[Sandbox]")
        return f"{prefix}: {self.message}"

def _ok(msg: str) -> SandboxOutcome:
    return SandboxOutcome(True, "ok", msg)

def _block(msg: str) -> SandboxOutcome:
    return SandboxOutcome(False, "security_block", msg)

def _error(msg: str) -> SandboxOutcome:
    return SandboxOutcome(False, "runtime_error", msg)

# ──────────────────────────────────────────────
# Path Guard (математически строгий)
# ──────────────────────────────────────────────

# Запрещённые имена устройств Windows
_WINDOWS_DEVICES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})

def _ensure_workspace() -> Path:
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    return SANDBOX_ROOT

def resolve_safe_path(user_path: str) -> Path:
    """
    Гарантирует, что итоговый путь находится строго внутри SANDBOX_ROOT.
    Любая попытка выйти = PermissionError.
    """
    clean = (user_path or "").strip().strip("\"'")

    if not clean or "\x00" in clean:
        raise PermissionError("Empty or null-byte path is forbidden")

    # Абсолютные пути, UNC, ведущие слэши
    if Path(clean).is_absolute() or clean.startswith(("/", "\\")):
        raise PermissionError(f"Absolute/UNC paths are forbidden: '{user_path}'")

    # NTFS Alternate Data Streams и drive letters
    if ":" in clean:
        raise PermissionError("NTFS streams and drive letters are forbidden")

    # Windows device names
    pure_name = Path(clean).name.upper().split(".")[0]
    if pure_name in _WINDOWS_DEVICES:
        raise PermissionError(f"Windows device name is forbidden: '{pure_name}'")

    root = _ensure_workspace()
    candidate = (root / clean).resolve()

    # Финальная математическая проверка
    if not candidate.is_relative_to(root):
        raise PermissionError(f"Path traversal blocked, path escapes sandbox: '{user_path}'")
    return candidate

# ──────────────────────────────────────────────
# Isolated Python Executor
# ──────────────────────────────────────────────

_BLOCKED_AST_NODES = (
    ast.Import, ast.ImportFrom,
    ast.Global, ast.Nonlocal,
    ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
    ast.ClassDef,          # на всякий случай
)

_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool, "chr": chr,
    "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset, "hex": hex,
    "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
    "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}

def _validate_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")

    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_AST_NODES):
            raise PermissionError(f"Blocked AST node: {type(node).__name__}")

        # Запрет dunder-атрибутов
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PermissionError("Dunder attribute access is forbidden")

        # Запрет огромных степеней (DoS)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                if node.right.value > 10_000:
                    raise PermissionError("Excessive exponentiation blocked")

def run_isolated_python(code_snippet: str) -> SandboxOutcome:
    code = (code_snippet or "").strip()
    if not code:
        return _ok("Empty Python snippet")

    if len(code) > CONFIG.max_code_chars:
        return _block(f"Code exceeds {CONFIG.max_code_chars} characters limit")

    try:
        _validate_ast(code)
    except PermissionError as e:
        return _block(str(e))
    except SyntaxError as e:
        return _error(f"Syntax error: {e}")

    # Минимальное окружение
    env = {}
    if IS_WINDOWS:
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")

    # Безопасный способ передать код без проблем с f-string
    runner_code = """
import ast, math, json, re, datetime, sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(2048)

code = sys.argv[1]
tree = ast.parse(code, mode="exec")

safe_builtins = {
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool, "chr": chr,
    "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
    "float": float, "format": format, "frozenset": frozenset, "hex": hex,
    "int": int, "len": len, "list": list, "map": map, "max": max, "min": min,
    "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
    "reversed": reversed, "round": round, "set": set, "slice": slice,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}

safe_globals = {
    "__builtins__": safe_builtins,
    "math": math,
    "json": json,
    "re": re,
    "datetime": datetime,
}
exec(compile(tree, "<sandbox>", "exec"), safe_globals, {})
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", runner_code, code],
            cwd=str(_ensure_workspace()),
            env=env,
            capture_output=True,
            text=True,
            timeout=CONFIG.max_exec_timeout,
            shell=False,
        )

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            err = stderr.splitlines()[-1] if stderr else "Execution failed"
            return _error(err[:300])

        output = stdout if stdout else "(no stdout)"
        if len(output) > CONFIG.max_output_chars:
            output = output[:CONFIG.max_output_chars] + "\n...[TRUNCATED]"

        return _ok(f"Python output:\n{output}")

    except subprocess.TimeoutExpired:
        return _block(f"Execution killed after {CONFIG.max_exec_timeout}s timeout")
    except Exception as e:
        logger.exception("Python sandbox failure")
        return _error(f"{type(e).__name__}: {e}")

# ──────────────────────────────────────────────
# File operations
# ──────────────────────────────────────────────

def sandbox_write(rel_path: str, content: str) -> SandboxOutcome:
    try:
        target = resolve_safe_path(rel_path)

        if target.suffix.lower() not in CONFIG.allowed_text_exts:
            return _block(f"Extension '{target.suffix}' not allowed. Allowed: {sorted(CONFIG.allowed_text_exts)}")

        data = content.encode("utf-8")
        if len(data) > CONFIG.max_file_bytes:
            return _block(f"File exceeds {CONFIG.max_file_bytes // 1024} KB limit")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        rel = target.relative_to(SANDBOX_ROOT)
        return _ok(f"Saved {len(data)} bytes → sandbox_workspace/{rel}")

    except PermissionError as e:
        return _block(str(e))
    except OSError as e:
        return _error(f"Write failed: {e.strerror}")

def sandbox_read(rel_path: str) -> SandboxOutcome:
    try:
        target = resolve_safe_path(rel_path)

        if target.suffix.lower() not in CONFIG.allowed_text_exts:
            return _block(f"Reading '{target.suffix}' is blocked")

        if not target.is_file():
            return _ok("File does not exist")

        if target.stat().st_size > CONFIG.max_file_bytes:
            return _block(f"File exceeds {CONFIG.max_file_bytes // 1024} KB limit")

        text = target.read_text(encoding="utf-8", errors="replace")[:CONFIG.max_output_chars]
        return _ok(f"File {target.name}:\n{text}")

    except PermissionError as e:
        return _block(str(e))
    except OSError as e:
        return _error(f"Read failed: {e.strerror}")

def sandbox_mkdir(rel_path: str) -> SandboxOutcome:
    try:
        target = resolve_safe_path(rel_path)
        target.mkdir(parents=True, exist_ok=True)
        rel = target.relative_to(SANDBOX_ROOT)
        return _ok(f"Created directory sandbox_workspace/{rel}")
    except PermissionError as e:
        return _block(str(e))
    except OSError as e:
        return _error(f"mkdir failed: {e.strerror}")

def sandbox_ls() -> SandboxOutcome:
    root = _ensure_workspace()
    items = sorted(
        p.name + ("/" if p.is_dir() else "")
        for p in root.iterdir()
    )
    return _ok(", ".join(items) if items else "(empty)")

# ──────────────────────────────────────────────
# Allowlisted GUI apps
# ──────────────────────────────────────────────

ALLOWED_GUI = {
    "calc": ["calc.exe"] if IS_WINDOWS else ["gnome-calculator"],
    "calc.exe": ["calc.exe"] if IS_WINDOWS else ["gnome-calculator"],
    "notepad": ["notepad.exe"] if IS_WINDOWS else ["gedit"],
    "notepad.exe": ["notepad.exe"] if IS_WINDOWS else ["gedit"],
}

def launch_gui(app: str) -> SandboxOutcome:
    cmd = ALLOWED_GUI.get(app.lower())
    if not cmd:
        return _block(f"Application '{app}' is not in the allowlist")

    try:
        subprocess.Popen(
            cmd,
            cwd=str(_ensure_workspace()),
            env={},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
        return _ok(f"Launched {app}")
    except Exception as e:
        return _error(f"Failed to launch {app}: {type(e).__name__}")

# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────

PLUGIN_NAME = "Zero-Trust OS & Python Micro-Sandbox"

def run(query: str, config: SandboxConfig = DEFAULT_CONFIG) -> str:
    """
    Unified entry point.
    Always returns an LLM-ready formatted string.
    """
    """
    Единая точка входа.
    Всегда возвращает строку, готовую для передачи в LLM.
    """
    raw = (query or "").strip()
    if not raw:
        return _ok("Empty command").format_for_llm()

    # 1. Python
    m = re.match(r"^(?:py|python|eval)\s*:\s*(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if m:
        return run_isolated_python(m.group(1)).format_for_llm()

    # 2. write
    m = re.match(r"^write\s+(\S+)\s+(.+)$", raw, re.IGNORECASE | re.DOTALL)
    if m:
        return sandbox_write(m.group(1), m.group(2)).format_for_llm()

    # 3. Парсим как команду
    try:
        tokens = shlex.split(raw, posix=not IS_WINDOWS)
    except ValueError as e:
        return _block(f"Malformed command: {e}").format_for_llm()

    if not tokens:
        return _ok("No command").format_for_llm()

    cmd = tokens[0].lower()
    args = [a.strip("\"'") for a in tokens[1:]]

    if cmd in ("ls", "dir"):
        if args:
            return _block("ls does not accept arguments").format_for_llm()
        return sandbox_ls().format_for_llm()

    if cmd == "read":
        if len(args) != 1:
            return _block("read requires exactly one path").format_for_llm()
        return sandbox_read(args[0]).format_for_llm()

    if cmd == "mkdir":
        if len(args) != 1:
            return _block("mkdir requires exactly one path").format_for_llm()
        return sandbox_mkdir(args[0]).format_for_llm()

    if cmd in ALLOWED_GUI:
        if args:
            return _block(f"Arguments are forbidden for {cmd}").format_for_llm()
        return launch_gui(cmd).format_for_llm()

    return _block(
        f"Command '{cmd}' is not in the allowlist. "
        "Supported: py: <code>, write <file> <text>, read <file>, mkdir <dir>, ls, calc, notepad"
    ).format_for_llm()

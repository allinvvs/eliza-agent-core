from __future__ import annotations

import logging
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class SandboxOutcome:
    allowed: bool
    category: str  # "ok" | "security_block" | "runtime_error"
    message: str

    def format_for_llm(self) -> str:
        prefix = {
            "ok": "[Sandbox]",
            "security_block": "[Sandbox Security]",
            "runtime_error": "[Sandbox Error]",
        }[self.category]
        return f"{prefix}: {self.message}"

logger = logging.getLogger(__name__)

PLUGIN_NAME = "Zero-Trust OS & Python Micro-Sandbox"
TRIGGERS = [
    "sandbox", "run", "execute", "calc", "notepad", "mkdir", "folder", "eval", "calculate",
    "песочница", "запусти", "выполни", "калькулятор", "блокнот", "папку", "посчитай",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_ROOT = (PROJECT_ROOT / "sandbox_workspace").resolve()

MAX_EXEC_TIMEOUT = 2.0      # Hard execution wall-clock timeout (seconds)
MAX_OUTPUT_CHARS = 1500     # Strict output budget for n_ctx=3072
MAX_FILE_BYTES = 32 * 1024  # 32 KB max file size inside sandbox workspace
ALLOWED_TEXT_EXTS = frozenset({".txt", ".md", ".json", ".csv", ".log"})

ALLOWED_GUI_APPS: dict[str, list[str]] = (
    {
        "calc": ["calc.exe"],
        "calc.exe": ["calc.exe"],
        "notepad": ["notepad.exe"],
        "notepad.exe": ["notepad.exe"],
    }
    if platform.system() == "Windows"
    else {
        "calc": ["gnome-calculator"],
        "notepad": ["gedit"],
    }
)

# Isolated AST-verified Python evaluator (blocks imports, dunders, huge exponents & memory bombs)
_ISOLATED_PY_RUNNER = """
import ast, datetime, json, math, re, sys

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(2048)

code = sys.argv[1]
tree = ast.parse(code, mode="exec")

for node in ast.walk(tree):
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
        raise PermissionError(f"Blocked AST construct: {type(node).__name__}")
    if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
        raise PermissionError("Blocked dunder attribute access")
    # Block CPU/RAM exhaustion via huge exponentiation (e.g. 9**9**9)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) and node.right.value > 10000:
            raise PermissionError("Blocked excessive exponentiation")

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
""".strip()


def _ensure_sandbox_dir() -> Path:
    """Ensures the isolated sandbox directory exists."""
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    return SANDBOX_ROOT


def _resolve_safe_path(user_path: str) -> Path:
    """Resolves a path and mathematically guarantees it stays inside SANDBOX_ROOT (anti-traversal)."""
    clean = user_path.strip().strip("\"'")
    if not clean or "\x00" in clean:
        raise PermissionError("Path traversal blocked: empty or null-byte path.")

    # Red-Team Audit: блокируем UNC-пути (\\evil\share), диски (C:) и ведущие слэши до вызова Path.resolve()
    if Path(clean).is_absolute() or clean.startswith(("/", "\\")):
        raise PermissionError(f"Path traversal blocked: '{user_path}' must be relative to sandbox.")

    # Блокируем попытки создать скрытые потоки NTFS (file.txt:evil.exe) и device-имена Windows (CON, NUL, PRN)
    if ":" in clean:
        raise PermissionError("Path traversal blocked: NTFS stream syntax is forbidden.")

    root = _ensure_sandbox_dir()
    candidate = (root / clean).resolve()

    if not candidate.is_relative_to(root):
        raise PermissionError(f"Path traversal blocked: '{user_path}' escapes sandbox workspace.")
    return candidate


def _run_isolated_python(code_snippet: str) -> str:
    """Executes pure computational Python code in an isolated (-I -B -S), env-stripped subprocess."""
    snippet = code_snippet.strip()
    if not snippet:
        return "[Sandbox]: Empty Python snippet."
    if len(snippet) > 1000:
        return "[Sandbox Security]: Code snippet exceeds 1000 characters limit."

    minimal_env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows")} if platform.system() == "Windows" else {}

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-B", "-S", "-c", _ISOLATED_PY_RUNNER, snippet],
            cwd=str(_ensure_sandbox_dir()),
            env=minimal_env,
            capture_output=True,
            text=True,
            timeout=MAX_EXEC_TIMEOUT,
            shell=False,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        if proc.returncode != 0:
            err_line = stderr.splitlines()[-1] if stderr else "Execution failed"
            return f"[Sandbox Python Error]: {err_line[:300]}"

        output = stdout if stdout else "(Executed cleanly with no stdout)"
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n...[TRUNCATED]"
        return f"[Sandbox Python Output]:\n{output}"
    except subprocess.TimeoutExpired:
        return f"[Sandbox Security]: Execution killed after {MAX_EXEC_TIMEOUT}s timeout."
    except Exception as exc:
        logger.warning("Sandbox python execution error: %s", exc)
        return f"[Sandbox Error]: {type(exc).__name__}"


def run(query: str) -> str:
    """Main entry point for zero-trust OS actions, workspace files, and isolated Python evaluation."""
    raw = (query or "").strip()
    if not raw:
        return "[Sandbox]: Empty command."

    # 1. Isolated Python computational runtime ("py: <code>" or "eval: <code>")
    py_match = re.match(r"^(?:py|python|eval)\s*:\s*(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if py_match:
        return _run_isolated_python(py_match.group(1))

    # 2. Safe workspace text file writer ("write <file.txt> <content>")
    write_match = re.match(r"^write\s+([^\s]+)\s+(.+)$", raw, flags=re.IGNORECASE | re.DOTALL)
    if write_match:
        rel_name, body = write_match.group(1), write_match.group(2)
        try:
            target = _resolve_safe_path(rel_name)
            if target.suffix.lower() not in ALLOWED_TEXT_EXTS:
                return f"[Sandbox Security]: Extension '{target.suffix}' is not allowed. Use {sorted(ALLOWED_TEXT_EXTS)}."
            encoded = body.encode("utf-8")
            if len(encoded) > MAX_FILE_BYTES:
                return f"[Sandbox Security]: File size exceeds {MAX_FILE_BYTES // 1024} KB limit."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded)
            return f"[Sandbox]: Saved {len(encoded)} bytes to 'sandbox_workspace/{target.relative_to(SANDBOX_ROOT)}'."
        except PermissionError as sec_err:
            return f"[Sandbox Security]: {sec_err}"
        except OSError as os_err:
            return f"[Sandbox Error]: Write failed ({os_err.strerror})."

    try:
        tokens = shlex.split(raw, posix=(platform.system() != "Windows"))
    except ValueError as exc:
        return f"[Sandbox Security]: Malformed command syntax ({exc})."

    if not tokens:
        return "[Sandbox]: No executable token provided."

    cmd = tokens[0].lower()
    args = [arg.strip("\"'") for arg in tokens[1:]]

    # 3. List isolated workspace contents ("ls" or "dir")
    if cmd in ("ls", "dir"):
        if args:
            return "[Sandbox Security]: 'ls' does not accept path arguments."
        root = _ensure_sandbox_dir()
        items = sorted(p.name + ("/" if p.is_dir() else "") for p in root.iterdir())
        return f"[Sandbox Workspace]: {', '.join(items) if items else '(empty)'}"

    # 4. Safe text file reader ("read <file.txt>")
    if cmd == "read":
        if len(args) != 1:
            return "[Sandbox Security]: 'read' requires exactly one relative file path."
        try:
            target = _resolve_safe_path(args[0])
            if target.suffix.lower() not in ALLOWED_TEXT_EXTS:
                return f"[Sandbox Security]: Reading '{target.suffix}' files is blocked."
            if not target.is_file():
                return "[Sandbox]: File does not exist in workspace."
            if target.stat().st_size > MAX_FILE_BYTES:
                return f"[Sandbox Security]: File exceeds {MAX_FILE_BYTES // 1024} KB limit."
            text = target.read_text(encoding="utf-8", errors="replace")[:MAX_OUTPUT_CHARS]
            return f"[Sandbox File: {target.name}]:\n{text}"
        except PermissionError as sec_err:
            return f"[Sandbox Security]: {sec_err}"
        except OSError as os_err:
            return f"[Sandbox Error]: Read failed ({os_err.strerror})."

    # 5. Safe relative directory creation ("mkdir <rel_path>")
    if cmd == "mkdir":
        if len(args) != 1:
            return "[Sandbox Security]: 'mkdir' accepts exactly one relative directory name."
        try:
            target_dir = _resolve_safe_path(args[0])
            target_dir.mkdir(parents=True, exist_ok=True)
            rel_path = target_dir.relative_to(SANDBOX_ROOT)
            return f"[Sandbox]: Created isolated directory 'sandbox_workspace/{rel_path}'."
        except PermissionError as sec_err:
            return f"[Sandbox Security]: {sec_err}"
        except OSError as os_err:
            return f"[Sandbox Error]: Unable to create directory ({os_err.strerror})."

    # 6. Allowlisted GUI utilities (zero arguments allowed)
    if cmd in ALLOWED_GUI_APPS:
        if args:
            return f"[Sandbox Security]: Arguments are forbidden for '{cmd}'."
        try:
            subprocess.Popen(
                ALLOWED_GUI_APPS[cmd],
                cwd=str(_ensure_sandbox_dir()),
                env={},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            return f"[Sandbox]: Launched allowlisted application '{cmd}'."
        except Exception as exc:
            return f"[Sandbox Error]: Failed to launch '{cmd}' ({type(exc).__name__})."
    return (
        f"[Sandbox Security]: Command '{cmd}' is not in the allowlist. "
        "Allowed: 'py: [code]', 'mkdir [dir]', 'ls', 'write [file] [text]', 'read [file]', 'calc', 'notepad'."
    )

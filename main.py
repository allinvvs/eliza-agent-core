from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable

from core.engine import AgentEngine, EngineConfig, PROJECT_ROOT

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

EXIT_COMMANDS: frozenset[str] = frozenset({"exit", "quit", "/exit"})

CORE_MODULES: tuple[str, ...] = (
    "core/engine.py",
    "core/memory.py",
    "core/breathing_mod.py",
    "server.py",
    "main.py",
)

SANDBOX_VERBS: tuple[str, ...] = (
    "py:",
    "python:",
    "eval:",
    "write ",
    "read ",
    "mkdir ",
    "ls",
    "calc",
    "notepad",
)


@dataclass(frozen=True)
class CommandSpec:
    handler: Callable[[str], str | None]
    usage: str
    requires_arg: bool = False


def _count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _print_self_code() -> None:
    print("\n--- ELIZA v5 Architecture ---")
    total_lines = 0
    for rel in CORE_MODULES:
        target = PROJECT_ROOT / rel
        if target.is_file():
            count = _count_lines(target)
            total_lines += count
            print(f"  * {rel:<24} {count:>5} lines")
    print(f"  {'Total:':<26} {total_lines:>5} lines")
    print("-----------------------------\n")


def _invoke_plugin(engine: AgentEngine, name: str, query: str) -> str:
    plugin = engine.plugins.get(name)
    if plugin is None:
        return f"[Error]: Plugin '{name}' is not loaded."

    if isinstance(plugin, dict):
        runner = plugin.get("fn") or plugin.get("run") or plugin.get("handler") or plugin.get("func")
        if callable(runner):
            return str(runner(query))
        raise TypeError(f"Unsupported plugin dict schema for '{name}': keys={list(plugin.keys())}")

    if callable(plugin):
        return str(plugin(query))

    run_method = getattr(plugin, "run", None)
    if callable(run_method):
        return str(run_method(query))

    raise TypeError(f"Plugin '{name}' is not executable: {type(plugin).__name__}")

def _build_command_registry(engine: AgentEngine, metrics_ref: dict) -> dict[str, CommandSpec]:
    def _run_sandbox(arg: str) -> str:
        normalized = arg if arg.lower().startswith(SANDBOX_VERBS) else f"py: {arg}"
        return _invoke_plugin(engine, "sandbox", normalized)

    def _show_plugins(_: str) -> str:
        return f"Loaded plugins: {', '.join(engine.plugins.keys())}"

    def _show_metrics(_: str) -> str:
        return f"[Last turn metrics]: {metrics_ref or 'no data yet'}"

    return {
        "/pc": CommandSpec(
            handler=lambda _: _invoke_plugin(engine, "pc_monitor", ""),
            usage="/pc",
        ),
        "/web": CommandSpec(
            handler=lambda arg: _invoke_plugin(engine, "web_search", arg),
            usage="/web <request>",
            requires_arg=True,
        ),
        "/py": CommandSpec(
            handler=_run_sandbox,
            usage="/py <code_or_cmd>",
            requires_arg=True,
        ),
        "/plugins": CommandSpec(
            handler=_show_plugins,
            usage="/plugins",
        ),
        "/code": CommandSpec(
            handler=lambda _: _print_self_code(),
            usage="/code",
        ),
        "/metrics": CommandSpec(
            handler=_show_metrics,
            usage="/metrics",
        ),
    }

def _try_dispatch_command(user_input: str, commands: dict[str, CommandSpec]) -> bool:
    if not user_input.startswith("/"):
        return False

    raw_cmd, _, raw_arg = user_input.partition(" ")
    spec = commands.get(raw_cmd.lower())
    if spec is None:
        return False

    arg = raw_arg.strip()
    if spec.requires_arg and not arg:
        print(f"Usage: {spec.usage}\n")
        return True

    try:
        if output := spec.handler(arg):
            print(f"{output}\n")
    except Exception as exc:
        print(f"[Command error]: {type(exc).__name__}: {exc}\n")

    return True


def _render_turn(engine: AgentEngine, user_input: str, metrics_ref: dict) -> None:
    print("Eliza: ", end="", flush=True)
    thinking_shown = False

    for event in engine.stream_turn(user_input):
        match event:
            case {"type": "thinking"} if not thinking_shown:
                print("[thinking...] ", end="", flush=True)
                thinking_shown = True
            case {"type": "tool", "tool": tool, "query": query}:
                print(f"\n[Plugin {tool} -> {query}]\nEliza: ", end="", flush=True)
            case {"type": "token", "token": token}:
                print(token, end="", flush=True)
            case {"type": "done", "result": res} if isinstance(res, dict):
                metrics_ref.clear()
                metrics_ref.update(res.get("metrics") or {})
                if actions := res.get("system_actions"):
                    print(f"\n[System actions]: {', '.join(actions)}", end="")

    print("\n")

def main() -> None:
    config = EngineConfig.from_env()
    print("Loading core ELIZA v5 (Single-Source AgentEngine)...")
    engine = AgentEngine(config=config)

    last_metrics: dict = {}
    commands = _build_command_registry(engine, last_metrics)
    banner_items = ["'exit'", *(f"'{spec.usage}'" for spec in commands.values())]

    print(f"Done! Active plugins: {list(engine.plugins.keys())}")
    print(f"Commands: {', '.join(banner_items)}\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in EXIT_COMMANDS:
                break
            if _try_dispatch_command(user_input, commands):
                continue

            _render_turn(engine, user_input, last_metrics)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        engine.shutdown()
        print("Eliza: Goodbye! Memory saved.")


if __name__ == "__main__":
    main()


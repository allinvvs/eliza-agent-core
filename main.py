import logging
from pathlib import Path
import sys

from core.engine import AgentEngine, EngineConfig, PROJECT_ROOT

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_self_code() -> None:
    files = ["core/engine.py", "core/memory.py", "core/breathing_mod.py", "server.py", "main.py"]
    print("\n--- Architecture ELIZA v5 ---")
    for rel in files:
        p = PROJECT_ROOT / rel
        if p.exists():
            lines = len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
            print(f"  * {rel}: {lines} строк")
    print("----------------------------\n")


def main() -> None:
    config = EngineConfig.from_env()
    print("Loading core ELIZA v5 (Single-Source AgentEngine)...")
    engine = AgentEngine(config=config)
    print(f"Done! Active plugins: {list(engine.plugins.keys())}")
    print("Comands: 'exit', '/pc', '/web <request>', '/plugins', '/code', '/metrics'\n")

    last_metrics: dict = {}

#Блок 2. Интерактивный потоковый цикл диалога

    EXIT_COMMANDS = frozenset({"exit", "quit", "/exit"})

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in EXIT_COMMANDS:
                break
            if cmd == "/code":
                _print_self_code()
                continue
            if cmd == "/metrics":
                print(f"[Last turn metrics]: {last_metrics or 'no data yet'}\n")
                continue

            print("Eliza: ", end="", flush=True)
            thinking_shown = False

            for event in engine.stream_turn(user_input):
                etype = event.get("type")
                if etype == "thinking" and not thinking_shown:
                    print("[thinking...] ", end="", flush=True)
                    thinking_shown = True
                elif etype == "tool":
                    print(f"\n[Plugin {event.get('tool')} -> {event.get('query')}]\nEliza: ", end="", flush=True)
                elif etype == "token":
                    print(event.get("token", ""), end="", flush=True)
                elif etype == "done":
                    res = event.get("result") or {}
                    last_metrics = res.get("metrics", {})
                    actions = res.get("system_actions") or []
                    if actions:
                        print(f"\n[System actions]: {', '.join(actions)}", end="")
            print("\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        engine.shutdown()
        print("Eliza: Goodbye! Memory saved.")


if __name__ == "__main__":
    main()

import logging
from pathlib import Path
import sys

from core.engine import AgentEngine, EngineConfig, PROJECT_ROOT

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")


def _print_self_code() -> None:
    files = ["core/engine.py", "core/memory.py", "core/breathing_mod.py", "server.py", "main.py"]
    print("\n--- Архитектура ELIZA v5 ---")
    for rel in files:
        p = PROJECT_ROOT / rel
        if p.exists():
            lines = len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
            print(f"  * {rel}: {lines} строк")
    print("----------------------------\n")


def main() -> None:
    config = EngineConfig.from_env()
    print("Загрузка ядра ELIZA v5 (Single-Source AgentEngine)...")
    engine = AgentEngine(config=config)
    print(f"Готово! Активные плагины: {list(engine.plugins.keys())}")
    print("Команды: 'выход', '/pc', '/web <запрос>', '/plugins', '/code', '/metrics'\n")

    last_metrics: dict = {}

#Блок 2. Интерактивный потоковый цикл диалога

    try:
        while True:
            try:
                user_input = input("Вы: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ("выход", "exit", "quit", "/exit"):
                engine.shutdown()
                print("Eliza: До встречи! Память сохранена.")
                break

            if cmd_lower in ("код", "/code"):
                _print_self_code()
                continue

            if cmd_lower == "/metrics":
                print(f"[Метрики последнего хода]: {last_metrics or 'ещё нет данных'}\n")
                continue

            print("Eliza: ", end="", flush=True)
            thinking_shown = False
            for event in engine.stream_turn(user_input):
                etype = event["type"]
                if etype == "thinking" and not thinking_shown:
                    print("[размышляет...] ", end="", flush=True)
                    thinking_shown = True
                elif etype == "tool":
                    print(f"\n[Плагин {event['tool']} -> {event['query']}]\nEliza: ", end="", flush=True)
                elif etype == "token":
                    print(event["token"], end="", flush=True)
                elif etype == "done":
                    res = event["result"]
                    last_metrics = res.get("metrics", {})
                    actions = res.get("system_actions") or []
                    if actions:
                        print(f"\n[Системные действия]: {', '.join(actions)}", end="")
            print("\n")
    except KeyboardInterrupt:
        print("\nСохранение памяти и остановка...")
        engine.shutdown()


if __name__ == "__main__":
    main()

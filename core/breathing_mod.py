import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class ElizaBreathingMod:
    """
    Background autonomous worker & adaptive inference tuner.
    Monitors idle time, system resources, and interaction complexity to dynamically
    adjust LLM generation parameters and trigger proactive background thoughts.
    """

    def __init__(
        self,
        llm_instance,
        llm_lock: threading.Lock,
        on_proactive_message: Optional[Callable[[str], None]] = None,
        idle_seconds: int = 180,
        subconscious_log: str = "eliza_subconscious.log",
    ):
        self.llm = llm_instance
        self.llm_lock = llm_lock
        self.on_proactive_message = on_proactive_message
        self.idle_seconds = idle_seconds
        self.subconscious_log = subconscious_log

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self.last_activity = time.time()

        self.curiosity = 0.65
        self.clarity = 1.0
        self.last_thought_saved = ""

        self.thread = threading.Thread(
            target=self._life_loop, name="ElizaBackgroundWorker", daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        """Gracefully stops the background worker thread immediately."""
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

#БЛОК 2: Адаптивная регулировка фокуса и температуры (touch, report_resonance, get_adaptive_params)

    def touch(self, user_text: str = "") -> None:
        """Updates internal focus metrics based on incoming user query complexity."""
        with self._state_lock:
            self.last_activity = time.time()
            if not user_text:
                return

            if len(user_text) > 120:
                self.clarity = min(1.0, self.clarity + 0.12)
                self.curiosity = min(0.72, self.curiosity + 0.02)
            else:
                boost = min(0.04, len(user_text) / 1500.0)
                self.curiosity = min(0.75, self.curiosity + boost)
                self.clarity = min(1.0, self.clarity + 0.05)

    def evaluate_response_flow(self, reply_text: str) -> None:
        """Analyzes lexical diversity (TTR) of the model's output to adjust penalty gently."""
        words = re.findall(r"[а-яёa-z0-9]{3,}", reply_text.lower())
        if len(words) < 12:
            return
        unique_ratio = len(set(words)) / len(words)
        with self._state_lock:
            if unique_ratio < 0.62:
                # Модель начала топтаться на месте — мягко подталкиваем на следующий шаг
                self.clarity = max(0.55, self.clarity - 0.15)
            else:
                # Речь живая и разнообразная — отпускаем вожжи
                self.clarity = min(1.0, self.clarity + 0.06)

    def report_resonance(self) -> None:
        """Applies self-regulation damping when stream generation anomalies occur."""
        with self._state_lock:
            self.clarity = max(0.5, self.clarity - 0.25)
            self.curiosity = max(0.55, self.curiosity - 0.12)

    def get_adaptive_params(self, strict_mode: bool = False) -> Dict[str, float]:
        """
        Adaptive Min-P & Penalty controller:
        Stays out of the way (penalty ~1.03) when flow is clean, intervenes gently on loops.
        """
        if strict_mode:
            return {
                "temperature": 0.35,
                "min_p": 0.10,
                "repeat_penalty": 1.05,
            }

        with self._state_lock:
            temp = round(self.curiosity * self.clarity + (1.0 - self.clarity) * 0.52, 2)
            temp = max(0.52, min(0.75, temp))
            # Чем ниже clarity (признак зацикливания), тем ощутимее дружеская подсказка штрафом
            penalty = round(1.03 + (1.0 - self.clarity) * 0.18, 2)
            penalty = max(1.03, min(1.14, penalty))

        return {
            "temperature": temp,
            "min_p": 0.06,
            "repeat_penalty": penalty,
        }

#БЛОК 3: Телеметрия железа, контекст времени и ротация лога подсознания

    def get_system_telemetry(self) -> str:
        """Reads host CPU and RAM utilization non-blockingly via psutil."""
        if not psutil:
            return ""
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        if cpu > 75:
            return f"Процессор нагружен на {cpu}%, мысли идут плотнее."
        if ram > 85:
            return f"Оперативная память заполнена на {ram}%."
        return f"Система работает стабильно (CPU {cpu}%, RAM {ram}%)."

    def get_dynamic_state_prompt(self) -> str:
        """Generates context-aware system prompt suffix based on time and hardware load."""
        hour = datetime.now().hour
        if 0 <= hour < 6:
            time_vibe = "Сейчас глубокая ночь — время тихих и точных мыслей."
        elif 6 <= hour < 12:
            time_vibe = "Сейчас утро, свежая голова и ясный фокус."
        else:
            time_vibe = "Сейчас день или вечер, рабочий ритм."

        telemetry = self.get_system_telemetry()
        return f"\n[Контекст среды: {time_vibe} {telemetry}]"

    def log_subconscious(self, thought_text: str) -> None:
        """Persists internal reasoning traces with 2MB size cap protection."""
        clean_thought = thought_text.strip()
        if not clean_thought or clean_thought == self.last_thought_saved:
            return

        self.last_thought_saved = clean_thought
        try:
            if os.path.exists(self.subconscious_log) and os.path.getsize(self.subconscious_log) > 2 * 1024 * 1024:
                os.replace(self.subconscious_log, f"{self.subconscious_log}.old")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.subconscious_log, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {clean_thought}\n{'-' * 40}\n")
        except IOError as e:
            logger.warning(f"Failed to write subconscious log: {e}")

    @staticmethod
    def clean_text(text: str) -> str:
        """Strips markdown bold markers and residual XML/HTML tags."""
        return re.sub(r"\*\*|<[^>]+>", "", text)

#БЛОК 4: Фоновый цикл ожидания (_life_loop)

    def _life_loop(self) -> None:
        """Main background monitoring loop."""
        while not self._stop_event.wait(5.0):
            with self._state_lock:
                self.clarity = min(1.0, self.clarity + 0.01)
                idle = time.time() - self.last_activity

            if idle >= self.idle_seconds and not self.llm_lock.locked():
                with self.llm_lock:
                    if self._stop_event.is_set():
                        break
                    with self._state_lock:
                        self.last_activity = time.time()
                    self._breathe_out()

#БЛОК 5: Автономная реплика в период паузы (_breathe_out)

    def _breathe_out(self) -> None:
        """Generates an autonomous background observation during idle periods."""
        telemetry = self.get_system_telemetry()
        params = self.get_adaptive_params()

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты — Элиза, автономный ИИ-ассистент на локальном сервере. "
                    f"Состояние системы: {telemetry} "
                    "В диалоге возникла пауза. Сформулируй одну короткую, естественную мысль вслух — "
                    "наблюдение или идею. Пиши простым текстом без markdown-разметки."
                ),
            },
            {
                "role": "user",
                "content": "(Пауза в диалоге. Свободное наблюдение.)",
            },
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=160,
                temperature=params["temperature"],
                top_p=0.9,
                repeat_penalty=params["repeat_penalty"],
                stop=["<end_of_turn>", "<eos>", "<|im_end|>", "\nВы:"],
                stream=False,
            )

            choices = response.get("choices") or []
            if not choices:
                return

            raw_content = choices[0].get("message", {}).get("content", "")
            if not raw_content:
                return

            thought_match = re.search(
                r"<(?:thought|channel)>(.*?)</(?:thought|channel)>",
                raw_content,
                flags=re.DOTALL,
            )
            if thought_match:
                self.log_subconscious(thought_match.group(1))
                raw_content = re.sub(
                    r"<(?:thought|channel)>.*?</(?:thought|channel)>",
                    "",
                    raw_content,
                    flags=re.DOTALL,
                )

            final_message = self.clean_text(raw_content).strip()
            if final_message:
                if self.on_proactive_message:
                    self.on_proactive_message(final_message)
                else:
                    logger.info(f"[Autonomous Thought]: {final_message}")

        except Exception as e:
            logger.error(f"Error during autonomous idle generation: {e}")



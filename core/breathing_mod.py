from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime
from typing import Any, Callable

try:
    import psutil
    psutil.cpu_percent(interval=None)
except ImportError:
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger("eliza.breathing")


class ElizaBreathingMod:
    """
    Non-blocking background autonomous worker & adaptive inference tuner.
    Uses monotonic clocks, zero-contention lock acquisition, and lexical diversity (TTR)
    feedback to dynamically tune sampling parameters without ever delaying user turns.
    """

    def __init__(
        self,
        llm_instance: Any,
        llm_lock: threading.Lock,
        on_proactive_message: Callable[[str], None] | None = None,
        idle_seconds: int = 180,
        subconscious_log: str = "eliza_subconscious.log",
    ) -> None:
        self.llm = llm_instance
        self.llm_lock = llm_lock
        self.on_proactive_message = on_proactive_message
        self.idle_seconds = max(10, int(idle_seconds))
        self.subconscious_log = subconscious_log

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_activity_mono = time.monotonic()

        self.curiosity = 0.65
        self.clarity = 1.0
        self.last_thought_saved = ""

        self.thread = threading.Thread(
            target=self._life_loop,
            name="ElizaBackgroundWorker",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        """Gracefully stops the background worker thread immediately."""
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def touch(self, user_text: str = "") -> None:
        """Updates monotonic activity timestamp and focus metrics on incoming user turn."""
        with self._state_lock:
            self._last_activity_mono = time.monotonic()
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
        """Analyzes lexical diversity (TTR) of model output to detect repetition loops."""
        words = re.findall(r"[а-яёa-z0-9]{3,}", reply_text.lower())
        if len(words) < 12:
            return
        unique_ratio = len(set(words)) / len(words)
        with self._state_lock:
            if unique_ratio < 0.62:
                self.clarity = max(0.55, self.clarity - 0.15)
            else:
                self.clarity = min(1.0, self.clarity + 0.06)

    def report_resonance(self) -> None:
        """Applies self-regulation damping when stream generation anomalies occur."""
        with self._state_lock:
            self.clarity = max(0.50, self.clarity - 0.25)
            self.curiosity = max(0.55, self.curiosity - 0.12)

    def get_adaptive_params(self, strict_mode: bool = False) -> dict[str, float]:
        """
        Adaptive Min-P & Repeat-Penalty controller:
        Stays unobtrusive (penalty ~1.03) during healthy flow; intervenes gently on loops.
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
            penalty = round(1.03 + (1.0 - self.clarity) * 0.18, 2)
            penalty = max(1.03, min(1.14, penalty))

        return {
            "temperature": temp,
            "min_p": 0.06,
            "repeat_penalty": penalty,
        }

#Блок 2 из 2: Дискретная телеметрия (KV-Cache friendly), неблокирующий _life_loop и _breathe_out

    def get_system_telemetry(self) -> str:
        """Reads host CPU/RAM state into discrete buckets to preserve LLM KV-cache stability."""
        if psutil is None:
            return "Host state: nominal."
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        if cpu > 80.0:
            return "Host state: high CPU load (keep reasoning concise)."
        if ram > 88.0:
            return "Host state: high RAM pressure (keep responses compact)."
        return "Host state: nominal."

    def get_dynamic_state_prompt(self) -> str:
        """Generates KV-cache-friendly context suffix based on time of day and discrete load."""
        hour = datetime.now().hour
        if 0 <= hour < 6:
            time_vibe = "Late night (quiet, focused tone)."
        elif 6 <= hour < 12:
            time_vibe = "Morning (clear, energetic focus)."
        elif 12 <= hour < 18:
            time_vibe = "Afternoon (steady working rhythm)."
        else:
            time_vibe = "Evening (calm, analytical tone)."

        telemetry = self.get_system_telemetry()
        return f"[Environment: {time_vibe} {telemetry}]"

    def log_subconscious(self, thought_text: str) -> None:
        """Persists internal reasoning traces with 2 MB log rotation protection."""
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
        except OSError as exc:
            logger.warning("Failed to write subconscious log: %s", exc)

    @staticmethod
    def clean_text(text: str) -> str:
        """Strips residual XML/HTML tags while preserving clean formatting."""
        return re.sub(r"<[^>]+>", "", text)

    def _life_loop(self) -> None:
        """Background idle monitor with zero-contention lock acquisition."""
        while not self._stop_event.wait(5.0):
            if self.llm is None:
                continue

            with self._state_lock:
                self.clarity = min(1.0, self.clarity + 0.01)
                idle_elapsed = time.monotonic() - self._last_activity_mono

            if idle_elapsed < self.idle_seconds:
                continue

            # Strictly non-blocking acquire: never queue behind or delay an active user turn
            if not self.llm_lock.acquire(blocking=False):
                continue

            try:
                if self._stop_event.is_set():
                    break
                with self._state_lock:
                    # Re-check idle inside lock in case a user turn just arrived
                    if (time.monotonic() - self._last_activity_mono) < self.idle_seconds:
                        continue
                    self._last_activity_mono = time.monotonic()
                self._breathe_out()
            finally:
                self.llm_lock.release()

    def _breathe_out(self) -> None:
        """Generates a brief autonomous observation during true idle periods."""
        if self.llm is None:
            return

        telemetry = self.get_system_telemetry()
        params = self.get_adaptive_params(strict_mode=False)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are ELIZA, an autonomous local AI core running on the user's workstation. "
                    f"{telemetry} "
                    "There is a brief pause in conversation. Share one concise, insightful observation "
                    "or technical thought in the user's primary language without markdown clutter."
                ),
            },
            {
                "role": "user",
                "content": "(Conversation idle pause. Emit one brief autonomous thought.)",
            },
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=140,
                temperature=params["temperature"],
                min_p=params["min_p"],
                repeat_penalty=params["repeat_penalty"],
                stop=["<end_of_turn>", "<eos>", "<|im_end|>", "\nYou:", "\nUser:", "\nВы:"],
                stream=False,
            )

            choices = response.get("choices") or []
            if not choices:
                return

            first_choice = next(iter(choices), {})
            raw_content = first_choice.get("message", {}).get("content", "")
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
                    logger.info("[Autonomous Thought]: %s", final_message)

        except Exception as exc:
            logger.error("Error during autonomous idle generation: %s", exc)

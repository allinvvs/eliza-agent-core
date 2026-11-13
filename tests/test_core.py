import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core import grammar_router
from core.breathing_mod import ElizaBreathingMod
from core.engine import AgentEngine, EngineConfig, execute_os_command
from core.memory import MemoryManager
from plugins import pc_monitor


class TestElizaCore(unittest.TestCase):
    def test_memory_idf_and_ring_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_path = os.path.join(tmpdir, "history.json")
            diary_path = os.path.join(tmpdir, "diary.txt")
            mem = MemoryManager(hist_path, diary_path, max_window=4)

            mem.write_to_diary("Настройка CUDA ускорения для видеокарты NVIDIA GTX 1650 прошла успешно.")
            mem.write_to_diary("Адаптивный сэмплер регулирует min_p и штраф за повторы.")

            res = mem.retrieve_crystals("Как работает видеокарта GTX 1650?", top_n=1)
            self.assertIn("GTX 1650", res)

    def test_pc_monitor_handles_headless_no_gpu_environment(self):
        with (
            patch.object(pc_monitor, "_NVML_AVAILABLE", False),
            patch("subprocess.check_output", side_effect=FileNotFoundError("nvidia-smi not found")),
        ):
            output = pc_monitor.run("железо")
        self.assertIn("ОС:", output)
        self.assertIn("Диск свободно:", output)
        self.assertNotIn("GPU:", output)

    def test_grammar_router_routes_valid_cot_intent(self):
        fake_plugins = {
            "pc_monitor": {"name": "Hardware Telemetry", "run": lambda q: "OK"}
        }
        mock_llm = MagicMock(
            return_value={
                "choices": [{"text": '{"reason": "запрос температуры видеокарты", "tool": "pc_monitor", "query": "температура gpu"}'}]
            }
        )
        del mock_llm.create_chat_completion
        with patch.object(grammar_router, "LlamaGrammar") as mock_grammar_cls:
            mock_grammar_cls.from_string.return_value = object()
            tool, query = grammar_router.route_intent_with_grammar(
                mock_llm, threading.Lock(), "Кулеры шумят, глянь чип", fake_plugins
            )
        self.assertEqual(tool, "pc_monitor")
        self.assertEqual(query, "температура gpu")

    def test_grammar_router_blocks_hallucinated_tools(self):
        fake_plugins = {
            "pc_monitor": {"name": "Hardware Telemetry", "run": lambda q: "OK"}
        }
        with patch.object(grammar_router, "LlamaGrammar") as mock_grammar_cls:
            mock_grammar_cls.from_string.return_value = object()

            hallucinated_llm = MagicMock(
                return_value={"choices": [{"text": '{"reason": "опасная команда", "tool": "rm_rf_root", "query": "x"}'}]}
            )
            del hallucinated_llm.create_chat_completion
            tool, _ = grammar_router.route_intent_with_grammar(
                hallucinated_llm, threading.Lock(), "сделай магию", fake_plugins
            )
            self.assertIsNone(tool)

    def test_agent_engine_slash_command_and_sandbox(self):
        self.assertIn("заблокирован", execute_os_command("mkdir C:/Windows/test_dir").lower())
        self.assertIn("отклонена", execute_os_command("powershell -Command whoami").lower())

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EngineConfig(
                history_file=os.path.join(tmpdir, "h.json"),
                diary_file=os.path.join(tmpdir, "d.txt"),
                idle_seconds=9999,
            )
            mock_llm = MagicMock()
            mock_llm.create_chat_completion.return_value = iter([
                {"choices": [{"delta": {"content": "Температура чипа в норме."}}]}
            ])
            with patch("threading.Timer"):
                eng = AgentEngine(config=cfg, llm_instance=mock_llm)
                res = eng.run_turn("/pc")
            self.assertEqual(res["plugin_used"], "pc_monitor")
            self.assertIn("Температура чипа", res["reply"])
            self.assertIn("rag_ms", res["metrics"])

    def test_breathing_mod_adaptive_ttr_and_strict_mode(self):
        with patch("threading.Timer"):
            mod = ElizaBreathingMod(llm_instance=None, llm_lock=threading.Lock(), idle_seconds=9999)
        strict = mod.get_adaptive_params(strict_mode=True)
        self.assertEqual(strict["temperature"], 0.35)
        self.assertEqual(strict["min_p"], 0.1)

        baseline = mod.get_adaptive_params(strict_mode=False)
        mod.evaluate_response_flow("повтор " * 15)
        after_repetition = mod.get_adaptive_params(strict_mode=False)
        self.assertGreater(after_repetition["repeat_penalty"], baseline["repeat_penalty"])


if __name__ == "__main__":
    unittest.main()


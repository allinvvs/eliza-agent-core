from __future__ import annotations

import concurrent.futures
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import grammar_router
from core.breathing_mod import ElizaBreathingMod
from core.engine import AgentEngine, EngineConfig, execute_os_command
from core.memory import MemoryManager
from plugins import pc_monitor, sandbox, web_search


class TestElizaCore(unittest.TestCase):
    """Hermetic, deterministic, zero-network test suite with concurrency & latency SLAs."""

    # -------------------------------------------------------------------------
    # 1. Memory Engine: Retrieval, Deduplication, Concurrency & Latency SLA
    # -------------------------------------------------------------------------
    def test_memory_idf_retrieval_concurrency_and_latency_sla(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_path = os.path.join(tmpdir, "history.json")
            diary_path = os.path.join(tmpdir, "diary.txt")
            mem = MemoryManager(hist_path, diary_path, max_window=4)

            # Seed 150 records + test deduplication
            for i in range(150):
                mem.write_to_diary(f"Telemetry log entry #{i}: system nominal under standard load.")
            mem.write_to_diary("CUDA hardware acceleration configured for NVIDIA GTX 1650 GPU.")
            mem.write_to_diary("CUDA hardware acceleration configured for NVIDIA GTX 1650 GPU.")

            # Concurrent read/write stress test (simulates multi-threaded HTTP + background worker)
            def worker(idx: int) -> str:
                mem.write_to_diary(f"Concurrent worker thread {idx} checkpoint.")
                return mem.retrieve_crystals("GTX 1650 GPU acceleration", top_n=1)

            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(worker, range(16)))
            elapsed_ms = (time.perf_counter() - t0) * 1000

            for res in results:
                self.assertIn("GTX 1650", res)
                self.assertEqual(res.count("GTX 1650"), 1)

            # Strict latency SLA: 16 concurrent writes + RAG lookups must finish well under 500ms
            self.assertLess(elapsed_ms, 500.0, f"Memory concurrency regression: {elapsed_ms:.1f}ms")

    # -------------------------------------------------------------------------
    # 2. Hardware & VRAM Telemetry (plugins/pc_monitor.py)
    # -------------------------------------------------------------------------
    def test_pc_monitor_headless_and_nvml_boundaries(self) -> None:
        pc_monitor._init_nvml_once.cache_clear()
        try:
            with (
                patch.object(pc_monitor, "pynvml", None),
                patch("shutil.which", return_value=None),
                patch("subprocess.check_output", side_effect=FileNotFoundError("nvidia-smi missing")),
            ):
                headless_out = pc_monitor.run("hardware")

            self.assertIn("OS:", headless_out)
            self.assertIn("SysDisk free:", headless_out)
            self.assertNotIn("GPU:", headless_out)
        finally:
            pc_monitor._init_nvml_once.cache_clear()

        fake_nvml = MagicMock()
        fake_nvml.NVML_TEMPERATURE_GPU = 0
        fake_nvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce GTX 1650"
        fake_nvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(used=2048 * 1048576, total=4096 * 1048576)
        fake_nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=42)
        fake_nvml.nvmlDeviceGetTemperature.return_value = 58

        try:
            with patch.object(pc_monitor, "pynvml", fake_nvml):
                gpu_out = pc_monitor.run("gpu")

            self.assertIn("NVIDIA GeForce GTX 1650", gpu_out)
            self.assertIn("VRAM 2048/4096 MB", gpu_out)
            self.assertIn("load 42%", gpu_out)
            self.assertIn("58°C", gpu_out)
        finally:
            pc_monitor._init_nvml_once.cache_clear()

    # -------------------------------------------------------------------------
    # 3. GBNF Structured-CoT Intent Router (core/grammar_router.py)
    # -------------------------------------------------------------------------
    def test_grammar_router_valid_and_adversarial_outputs(self) -> None:
        fake_plugins = {"pc_monitor": {"name": "Hardware Telemetry", "run": lambda q: "OK"}}
        with patch.object(grammar_router, "LlamaGrammar") as mock_grammar_cls:
            mock_grammar_cls.from_string.return_value = object()

            # Valid tool route
            valid_llm = MagicMock(
                return_value={"choices": [{"text": '{"reason": "check temp", "tool": "pc_monitor", "query": "temp"}'}]}
            )
            del valid_llm.create_chat_completion
            tool, query = grammar_router.route_intent_with_grammar(
                valid_llm, threading.Lock(), "Check chip temp", fake_plugins
            )
            self.assertEqual((tool, query), ("pc_monitor", "temp"))

            # Hallucinated tool & corrupted JSON must safely degrade to (None, ...)
            for bad_payload in ('{"reason": "x", "tool": "rm_rf", "query": "y"}', "{broken_json"):
                bad_llm = MagicMock(return_value={"choices": [{"text": bad_payload}]})
                del bad_llm.create_chat_completion
                bad_tool, _ = grammar_router.route_intent_with_grammar(
                    bad_llm, threading.Lock(), "test", fake_plugins
                )
                self.assertIsNone(bad_tool)

    # -------------------------------------------------------------------------
    # 4. Zero-Trust OS & Python Micro-Sandbox (Red-Team Security Tests)
    # -------------------------------------------------------------------------
    def test_sandbox_red_team_security_and_resource_caps(self) -> None:
        # Engine-level OS guard (strict protocol prefix check)
        forbidden_dir = "C:/Windows/test_dir" if os.name == "nt" else "/etc/eliza_test_dir"
        self.assertTrue(execute_os_command(f"mkdir {forbidden_dir}").startswith("[Sandbox Security]:"))
        self.assertTrue(execute_os_command("powershell -Command whoami").startswith("[Sandbox Security]:"))

        # Path traversal & NTFS stream attacks
        self.assertIn("traversal", sandbox.run("mkdir ../../Windows/System32").lower())
        self.assertIn("security", sandbox.run("mkdir folder:evil_stream").lower())
        self.assertIn("not in the allowlist", sandbox.run("cmd.exe /c whoami"))

        # AST sandbox escapes
        self.assertIn("Blocked AST construct", sandbox.run("py: import os; print(os.getcwd())"))
        self.assertIn("Blocked dunder", sandbox.run("py: print(().__class__.__bases__)"))

        # Output flood protection (must truncate strictly within context budget)
        flood_res = sandbox.run("py: print('X' * 5000)")
        self.assertIn("[TRUNCATED]", flood_res)
        self.assertLess(len(flood_res), 1700)

        # Infinite loop timeout enforcement
        with patch.object(sandbox, "MAX_EXEC_TIMEOUT", 0.3):
            timeout_res = sandbox.run("py: while True: pass")
            self.assertIn("timeout", timeout_res.lower())

    # -------------------------------------------------------------------------
    # 5. Hermetic SSRF Guard & Media Magic Bytes (Zero Real Network)
    # -------------------------------------------------------------------------
    def test_web_search_hermetic_ssrf_and_magic_bytes(self) -> None:
        # Direct private/loopback/blacklisted URLs
        for unsafe in (
            "http://127.0.0",
            "http://localhost:8000/chat",
            "http://192.168.0",
            "http://169.254.169",
            "ftp://://example.com",
            "https://openai.com",
        ):
            self.assertFalse(web_search._is_safe_url(unsafe), f"Failed to block unsafe URL: {unsafe}")

        # Simulate DNS Rebinding attack (public domain resolving to 127.0.0.1) without touching real DNS
        rebind_addr = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        with patch("socket.getaddrinfo", return_value=rebind_addr):
            self.assertFalse(web_search._is_safe_url("http://example.com"))

        # Simulate clean public IP resolution (e.g., 93.184.216.34)
        public_addr = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with patch("socket.getaddrinfo", return_value=public_addr):
            self.assertTrue(web_search._is_safe_url("https://example.com"))

        # Binary file signature verification
        self.assertEqual(web_search._detect_image_extension(b"\x89PNG\r\n\x1a\npayload"), ".png")
        self.assertEqual(web_search._detect_image_extension(b"\xff\xd8\xffpayload"), ".jpg")
        self.assertIsNone(web_search._detect_image_extension(b"<html><body>Spoofed MIME</body></html>"))

    # -------------------------------------------------------------------------
    # 6. Core AgentEngine & Adaptive Inference Controller
    # -------------------------------------------------------------------------
    def test_agent_engine_and_adaptive_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EngineConfig(
                history_file=os.path.join(tmpdir, "h.json"),
                diary_file=os.path.join(tmpdir, "d.txt"),
                idle_seconds=9999,
            )
            mock_llm = MagicMock()
            mock_llm.create_chat_completion.return_value = iter([
                {"choices": [{"delta": {"content": "Chip temperature is optimal."}}]}
            ])
            with patch("threading.Timer"):
                eng = AgentEngine(config=cfg, llm_instance=mock_llm)
                res = eng.run_turn("/pc")

            self.assertEqual(res["plugin_used"], "pc_monitor")
            self.assertIn("Chip temperature", res["reply"])
            self.assertIn("rag_ms", res["metrics"])

        with patch("threading.Timer"):
            mod = ElizaBreathingMod(llm_instance=None, llm_lock=threading.Lock(), idle_seconds=9999)

        strict = mod.get_adaptive_params(strict_mode=True)
        self.assertEqual((strict["temperature"], strict["min_p"]), (0.35, 0.1))

        baseline = mod.get_adaptive_params(strict_mode=False)
        mod.evaluate_response_flow("repeat_token " * 15)
        self.assertGreater(mod.get_adaptive_params(strict_mode=False)["repeat_penalty"], baseline["repeat_penalty"])


if __name__ == "__main__":
    unittest.main()



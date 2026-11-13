from __future__ import annotations

import http.client
import json
import sys
import threading
import unittest
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import server


class _FakeStreamEngine:
    """Deterministic zero-GPU mock engine for HTTP & SSE protocol verification."""

    def __init__(self, fail_mid_stream: bool = False) -> None:
        self.fail_mid_stream = fail_mid_stream
        self.shutdown_called = False

    def get_status(self) -> dict[str, Any]:
        return {"status": "online", "plugins": ["pc_monitor", "sandbox", "web_search"]}

    def drain_notifications(self) -> list[str]:
        return ["Autonomous check: systems nominal."]

    def shutdown(self) -> None:
        self.shutdown_called = True

    def stream_turn(self, user_input: str) -> Iterator[dict[str, Any]]:
        yield {"type": "thinking"}
        yield {"type": "token", "token": "Hello "}
        if self.fail_mid_stream:
            raise RuntimeError("Simulated CUDA out-of-memory mid-stream")
        yield {"type": "token", "token": f"from ELIZA ({user_input})"}
        yield {
            "type": "done",
            "result": {
                "status": "success",
                "reply": f"Hello from ELIZA ({user_input})",
                "plugin_used": None,
                "system_actions": [],
                "metrics": {"prompt_tokens": 24, "tokens_generated": 6, "total_ms": 12.5},
            },
        }


class TestElizaHTTPServerSecurity(unittest.TestCase):
    """Red-Team, Concurrency & Protocol test suite for ElizaAPIHandler."""

    def setUp(self) -> None:
        self.fake_engine = _FakeStreamEngine()
        self.httpd = server.ElizaThreadingServer(("127.0.0.1", 0), server.ElizaAPIHandler)
        self.port = self.httpd.server_address[1]
        self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.server_thread.join(timeout=2.0)

    def _raw_request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            req_headers = {"Host": f"127.0.0.1:{self.port}"}
            if headers:
                req_headers.update(headers)
            conn.request(method, path, body=body, headers=req_headers)
            resp = conn.getresponse()
            raw_body = resp.read()
            resp_headers = {k.lower(): v for k, v in resp.getheaders()}
            return resp.status, resp_headers, raw_body
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # 1. Trap: DNS Rebinding via Spoofed Host Header (even with valid API key)
    # -------------------------------------------------------------------------
    def test_dns_rebinding_host_header_trap(self) -> None:
        with (
            patch.object(server, "HOST", "127.0.0.1"),
            patch.object(server, "API_KEY", "super-secret-token"),
            patch.object(server, "get_engine", return_value=self.fake_engine),
        ):
            status, headers, raw = self._raw_request(
                "GET",
                "/status",
                headers={
                    "Host": "evil-rebinding-attacker.com:8000",
                    "Authorization": "Bearer super-secret-token",
                },
            )

        self.assertEqual(status, 403)
        self.assertIn("x-request-id", headers)
        self.assertIn("Invalid Host", raw.decode("utf-8"))

    # -------------------------------------------------------------------------
    # 2. Bearer Token Authentication & Constant-Time Guard
    # -------------------------------------------------------------------------
    def test_bearer_token_enforcement(self) -> None:
        with (
            patch.object(server, "API_KEY", "top-secret-key-2026"),
            patch.object(server, "get_engine", return_value=self.fake_engine),
        ):
            # Missing token -> 401
            st_no_auth, _, _ = self._raw_request("GET", "/health")
            self.assertEqual(st_no_auth, 401)

            # Wrong token -> 401
            st_bad_auth, _, _ = self._raw_request(
                "GET", "/health", headers={"Authorization": "Bearer wrong-key"}
            )
            self.assertEqual(st_bad_auth, 401)

            # Valid token -> 200 + X-Request-ID + nosniff security header
            st_ok, resp_headers, raw = self._raw_request(
                "GET", "/health", headers={"Authorization": "Bearer top-secret-key-2026"}
            )
            self.assertEqual(st_ok, 200)
            self.assertEqual(resp_headers.get("x-content-type-options"), "nosniff")
            self.assertTrue(resp_headers.get("x-request-id", "").startswith("req-"))
            self.assertEqual(json.loads(raw)["status"], "ok")

    # -------------------------------------------------------------------------
    # 3. DoS Memory Shield (Oversized Content-Length Rejection)
    # -------------------------------------------------------------------------
    def test_oversized_payload_rejected_before_reading(self) -> None:
        with (
            patch.object(server, "API_KEY", ""),
            patch.object(server, "get_engine", return_value=self.fake_engine),
        ):
            status, _, raw = self._raw_request(
                "POST",
                "/chat",
                body=b'{"message": "ping"}',
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(10 * 1024 * 1024),  # Claims 10 MB payload
                },
            )

        self.assertEqual(status, 400)
        self.assertIn("Invalid payload size", raw.decode("utf-8"))

    # -------------------------------------------------------------------------
    # 4. VRAM Backpressure Guard (429 Too Many Requests when slots full)
    # -------------------------------------------------------------------------
    def test_vram_concurrency_backpressure_429(self) -> None:
        # Exhaust all inference semaphore slots to simulate saturated GPU
        exhausted_sem = threading.BoundedSemaphore(1)
        exhausted_sem.acquire()

        with (
            patch.object(server, "API_KEY", ""),
            patch.object(server, "_INFERENCE_SLOTS", exhausted_sem),
            patch.object(server, "get_engine", return_value=self.fake_engine),
        ):
            payload = json.dumps({"message": "hello"}).encode("utf-8")
            status, _, raw = self._raw_request(
                "POST",
                "/chat",
                body=payload,
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(status, 429)
        self.assertIn("queue full", raw.decode("utf-8"))

    # -------------------------------------------------------------------------
    # 5. OpenAI-Compatible API (Sync JSON & SSE Streaming)
    # -------------------------------------------------------------------------
    def test_openai_completions_sync_and_sse_stream(self) -> None:
        with (
            patch.object(server, "API_KEY", ""),
            patch.object(server, "get_engine", return_value=self.fake_engine),
        ):
            # Sync OpenAI request
            sync_body = json.dumps({
                "model": "eliza-v5-local",
                "messages": [{"role": "user", "content": "status check"}],
                "stream": False,
            }).encode("utf-8")
            st_sync, _, raw_sync = self._raw_request(
                "POST", "/v1/chat/completions", body=sync_body, headers={"Content-Type": "application/json"}
            )
            self.assertEqual(st_sync, 200)
            data_sync = json.loads(raw_sync)
            self.assertEqual(data_sync["choices"][0]["message"]["content"], "Hello from ELIZA (status check)")
            self.assertEqual(data_sync["usage"]["completion_tokens"], 6)

            # Streaming SSE OpenAI request
            sse_body = json.dumps({
                "model": "eliza-v5-local",
                "messages": [{"role": "user", "content": "stream test"}],
                "stream": True,
            }).encode("utf-8")
            st_sse, sse_headers, raw_sse = self._raw_request(
                "POST", "/v1/chat/completions", body=sse_body, headers={"Content-Type": "application/json"}
            )
            self.assertEqual(st_sse, 200)
            self.assertIn("text/event-stream", sse_headers.get("content-type", ""))
            sse_text = raw_sse.decode("utf-8")
            self.assertIn("chat.completion.chunk", sse_text)
            self.assertIn("data: [DONE]", sse_text)

    # -------------------------------------------------------------------------
    # 6. Mid-Stream Fault Containment (Graceful SSE Error Termination)
    # -------------------------------------------------------------------------
    def test_native_sse_mid_stream_fault_containment(self) -> None:
        crashing_engine = _FakeStreamEngine(fail_mid_stream=True)
        with (
            patch.object(server, "API_KEY", ""),
            patch.object(server, "get_engine", return_value=crashing_engine),
        ):
            body = json.dumps({"message": "trigger crash", "stream": True}).encode("utf-8")
            status, _, raw = self._raw_request(
                "POST", "/chat", body=body, headers={"Content-Type": "application/json"}
            )

        self.assertEqual(status, 200)
        stream_out = raw.decode("utf-8")
        self.assertIn("Stream inference error", stream_out)
        self.assertIn('"done": true', stream_out)

    # -------------------------------------------------------------------------
    # 7. Protocol Fuzzing (Malformed JSON, Wrong Root Type, Sensitive Paths)
    # -------------------------------------------------------------------------
    
    def test_protocol_fuzzing_and_sensitive_path_probes(self) -> None:
        with (
            patch.object(server, "API_KEY", ""),
            patch.object(server, "get_engine", return_value=self.fake_engine),
        ):
            # Probe .env / admin routes -> 404
            for probe in ("/.env", "/admin", "/../secrets.json"):
                st, _, _ = self._raw_request("GET", probe)
                self.assertEqual(st, 404)

            # Malformed JSON -> 400
            st_bad_json, _, _ = self._raw_request(
                "POST", "/chat", body=b"{broken_json", headers={"Content-Type": "application/json"}
            )
            self.assertEqual(st_bad_json, 400)

            # JSON array instead of dict -> 400
            st_list_json, _, _ = self._raw_request(
                "POST", "/chat", body=b'["not", "an", "object"]', headers={"Content-Type": "application/json"}
            )
            self.assertEqual(st_list_json, 400)


if __name__ == "__main__":
    unittest.main()

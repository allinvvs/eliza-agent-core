from __future__ import annotations

import hmac
import json
import logging
import os
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from core.engine import AgentEngine, EngineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ElizaHTTPServer")

HOST = os.getenv("ELIZA_HOST", "127.0.0.1")
PORT = int(os.getenv("ELIZA_PORT", "8000"))
API_KEY = os.getenv("ELIZA_API_KEY", "").strip()
MAX_PAYLOAD_BYTES = 131_072  # 128 KB max JSON body
MAX_QUEUED_REQUESTS = 2      # Hard concurrency cap for 4 GB VRAM

_ALLOWED_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
_INFERENCE_SLOTS = threading.BoundedSemaphore(MAX_QUEUED_REQUESTS)
_engine_instance: AgentEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> AgentEngine:
    """Lazily initializes AgentEngine so importing server.py in tests doesn't load GGUF weights."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                logger.info("Initializing Eliza v5.1 HTTP/OpenAI Core...")
                _engine_instance = AgentEngine(config=EngineConfig.from_env())
    return _engine_instance


def _collect_turn_from_stream(eng: AgentEngine, user_msg: str) -> dict[str, Any]:
    """Single-path execution: drives stream_turn to completion so sync & SSE never diverge."""
    tokens: list[str] = []
    final_result: dict[str, Any] = {}

    for event in eng.stream_turn(user_msg):
        etype = event.get("type")
        if etype == "token":
            tokens.append(event.get("token", ""))
        elif etype == "done":
            final_result = event.get("result") or {}

    if "reply" not in final_result:
        final_result["reply"] = "".join(tokens).strip()
    return final_result


class ElizaThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ElizaAPIHandler(BaseHTTPRequestHandler):
    """REST JSON, Native SSE, and OpenAI-compatible HTTP API handler."""

    def setup(self) -> None:
        super().setup()
        self.request.settimeout(45.0)
        self.req_id = f"req-{uuid.uuid4().hex[:8]}"

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", self.req_id)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Request-ID", self.req_id)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("[%s] %s - %s", self.req_id, self.client_address[0], format % args)

    def _check_security_guards(self) -> bool:
        """Enforces DNS-Rebinding Host validation and constant-time Bearer auth with anti-bruteforce delay."""
        host_header = self.headers.get("Host", "")
        hostname = host_header.split(":", 1)[0].strip().lower()
        allowed_hosts = {"127.0.0.1", "localhost", "::1", HOST.lower()}
        if hostname not in allowed_hosts:
            self._send_json(403, {"status": "error", "request_id": self.req_id, "message": "Invalid Host header"})
            return False

        if not API_KEY:
            return True

        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            time.sleep(0.15)
            self._send_json(401, {"status": "error", "request_id": self.req_id, "message": "Unauthorized"})
            return False

        provided = auth_header[7:].strip()
        if not hmac.compare_digest(provided.encode("utf-8"), API_KEY.encode("utf-8")):
            time.sleep(0.15)
            self._send_json(401, {"status": "error", "request_id": self.req_id, "message": "Unauthorized"})
            return False

        return True

    def _stream_native_sse(self, eng: AgentEngine, user_msg: str) -> None:
        self._start_sse()
        try:
            for event in eng.stream_turn(user_msg):
                etype = event.get("type")
                if etype == "token":
                    payload: dict[str, Any] = {"token": event.get("token", ""), "done": False}
                elif etype == "thinking":
                    payload = {"token": "", "status": "thinking", "done": False}
                elif etype == "tool":
                    payload = {"token": "", "status": "tool", "tool": event.get("tool"), "done": False}
                elif etype == "done":
                    payload = {**(event.get("result") or {}), "request_id": self.req_id, "token": "", "done": True}
                else:
                    continue
                self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.info("[%s] Native SSE client disconnected mid-stream.", self.req_id)
        except Exception:
            logger.exception("[%s] Error during native SSE stream", self.req_id)
            err_payload = {"status": "error", "request_id": self.req_id, "message": "Stream inference error", "done": True}
            try:
                self.wfile.write(f"data: {json.dumps(err_payload)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass

    def _handle_openai_completions(self, eng: AgentEngine, data: dict[str, Any]) -> None:
        raw_msgs = data.get("messages") or []
        user_msg = ""
        for m in reversed(raw_msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                user_msg = str(m.get("content", "")).strip()
                break

        if not user_msg:
            self._send_json(400, {"error": {"message": "No user message found in 'messages'", "request_id": self.req_id}})
            return

        completion_id = f"chatcmpl-{self.req_id}"
        created_ts = int(time.time())

        if bool(data.get("stream", False)):
            self._start_sse()
            try:
                for event in eng.stream_turn(user_msg):
                    etype = event.get("type")
                    if etype == "token":
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": "eliza-v5-local",
                            "choices": [{"index": 0, "delta": {"content": event.get("token", "")}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    elif etype == "done":
                        stop_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": "eliza-v5-local",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        self.wfile.write(f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.info("[%s] OpenAI SSE client disconnected mid-stream.", self.req_id)
            except Exception:
                logger.exception("[%s] Error during OpenAI SSE stream", self.req_id)
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
            return

        result = _collect_turn_from_stream(eng, user_msg)
        metrics = result.get("metrics") or {}
        self._send_json(200, {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": "eliza-v5-local",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.get("reply", "")},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": metrics.get("prompt_tokens", 0),
                "completion_tokens": metrics.get("tokens_generated", 0),
            },
            "eliza_metrics": metrics,
        })

    def do_GET(self) -> None:
        if not self._check_security_guards():
            return

        route = urlparse(self.path).path
        if route == "/health":
            self._send_json(200, {"status": "ok", "request_id": self.req_id, "model": "eliza-v5-local"})
            return

        eng = get_engine()
        if route == "/status":
            self._send_json(200, {**eng.get_status(), "request_id": self.req_id})
        elif route == "/notifications":
            self._send_json(200, {"status": "ok", "request_id": self.req_id, "notifications": eng.drain_notifications()})
        elif route == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": "eliza-v5-local", "object": "model", "owned_by": "local"}],
            })
        else:
            self._send_json(404, {"status": "error", "request_id": self.req_id, "message": "Endpoint not found"})

    def do_POST(self) -> None:
        if not self._check_security_guards():
            return

        route = urlparse(self.path).path
        if route not in ("/chat", "/v1/chat/completions"):
            self._send_json(404, {"status": "error", "request_id": self.req_id, "message": "Endpoint not found"})
            return

        if not _INFERENCE_SLOTS.acquire(blocking=False):
            self._send_json(429, {"status": "error", "request_id": self.req_id, "message": "Server busy: queue full"})
            return

        try:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"status": "error", "request_id": self.req_id, "message": "Invalid Content-Length"})
                return

            if content_length <= 0 or content_length > MAX_PAYLOAD_BYTES:
                self._send_json(400, {"status": "error", "request_id": self.req_id, "message": "Invalid payload size"})
                return

            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(data, dict):
                self._send_json(400, {"status": "error", "request_id": self.req_id, "message": "JSON object required"})
                return

            eng = get_engine()
            if route == "/v1/chat/completions":
                self._handle_openai_completions(eng, data)
                return

            user_msg = str(data.get("message", "")).strip()
            if not user_msg:
                self._send_json(400, {"status": "error", "request_id": self.req_id, "message": "Field 'message' required"})
                return

            if bool(data.get("stream", False)):
                self._stream_native_sse(eng, user_msg)
            else:
                result = _collect_turn_from_stream(eng, user_msg)
                self._send_json(200, {**result, "request_id": self.req_id})
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "request_id": self.req_id, "message": "Malformed JSON"})
        except Exception:
            logger.exception("[%s] Unhandled HTTP POST error", self.req_id)
            self._send_json(500, {"status": "error", "request_id": self.req_id, "message": "Internal server error"})
        finally:
            _INFERENCE_SLOTS.release()


def main() -> None:
    eng = get_engine()
    server = ElizaThreadingServer((HOST, PORT), ElizaAPIHandler)
    shutdown_event = threading.Event()

    def _signal_handler(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s, initiating graceful shutdown...", signum)
        shutdown_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    auth_mode = "Bearer Token Enabled" if API_KEY else "Open Local Mode"
    logger.info("ELIZA v5.1 HTTP/OpenAI API listening on http://%s:%s [%s]", HOST, PORT, auth_mode)
    try:
        server.serve_forever()
    finally:
        logger.info("Flushing memory and closing server socket...")
        eng.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

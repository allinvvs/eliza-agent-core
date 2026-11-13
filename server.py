from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import time
import uuid

from core.engine import AgentEngine, EngineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ElizaHTTPServer")

config = EngineConfig.from_env()
HOST = os.getenv("ELIZA_HOST", "127.0.0.1")
PORT = int(os.getenv("ELIZA_PORT", "8000"))

logger.info("Initializing Eliza v5 HTTP/OpenAI Core...")
engine = AgentEngine(config=config)


class ElizaThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ElizaAPIHandler(BaseHTTPRequestHandler):
    """REST JSON, Native SSE, and OpenAI-compatible HTTP API handler."""

    def setup(self) -> None:
        super().setup()
        self.request.settimeout(30.0)

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        logger.info(f"{self.client_address} - {format % args}")

#Блок 2 для server.py (Эндпоинты /chat, /status, /notifications и OpenAI /v1/chat/completions)

    def _stream_native_sse(self, user_msg: str) -> None:
        self._start_sse()
        try:
            for event in engine.stream_turn(user_msg):
                etype = event["type"]
                if etype == "token":
                    payload = {"token": event["token"], "done": False}
                elif etype == "thinking":
                    payload = {"token": "", "status": "thinking", "done": False}
                elif etype == "tool":
                    payload = {"token": "", "status": "tool", "tool": event["tool"], "done": False}
                elif etype == "done":
                    payload = {**event["result"], "token": "", "done": True}
                else:
                    continue
                self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.info("Native SSE client disconnected mid-stream.")

    def _handle_openai_completions(self, data: dict) -> None:
        raw_msgs = data.get("messages") or []
        user_msg = ""
        for m in reversed(raw_msgs):
            if m.get("role") == "user":
                user_msg = str(m.get("content", "")).strip()
                break
        if not user_msg:
            self._send_json(400, {"error": {"message": "No user message found in 'messages'"}})
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if bool(data.get("stream", False)):
            self._start_sse()
            try:
                for event in engine.stream_turn(user_msg):
                    if event["type"] == "token":
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": "eliza-v5-local",
                            "choices": [{"index": 0, "delta": {"content": event["token"]}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    elif event["type"] == "done":
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
                logger.info("OpenAI SSE client disconnected mid-stream.")
            return

        result = engine.run_turn(user_msg)
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
                "prompt_tokens": result.get("metrics", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("metrics", {}).get("tokens_generated", 0),
            },
            "eliza_metrics": result.get("metrics", {}),
        })

    def do_GET(self) -> None:
        if self.path == "/status":
            self._send_json(200, engine.get_status())
        elif self.path == "/notifications":
            self._send_json(200, {"status": "ok", "notifications": engine.drain_notifications()})
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": "eliza-v5-local", "object": "model", "owned_by": "local"}],
            })
        else:
            self._send_json(404, {"status": "error", "message": "Endpoint not found"})

    def do_POST(self) -> None:
        if self.path not in ("/chat", "/v1/chat/completions"):
            self._send_json(404, {"status": "error", "message": "Endpoint not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 131072:
                self._send_json(400, {"status": "error", "message": "Invalid payload size"})
                return

            data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if self.path == "/v1/chat/completions":
                self._handle_openai_completions(data)
                return

            user_msg = str(data.get("message", "")).strip()
            if not user_msg:
                self._send_json(400, {"status": "error", "message": "Field 'message' is required"})
                return

            if bool(data.get("stream", False)):
                self._stream_native_sse(user_msg)
            else:
                self._send_json(200, engine.run_turn(user_msg))
        except json.JSONDecodeError:
            self._send_json(400, {"status": "error", "message": "Malformed JSON"})
        except Exception as e:
            logger.error(f"HTTP POST error: {e}")
            self._send_json(500, {"status": "error", "message": str(e)})


def main() -> None:
    server = ElizaThreadingServer((HOST, PORT), ElizaAPIHandler)
    logger.info(f"ELIZA v5 HTTP/OpenAI API listening on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down HTTP server...")
        engine.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

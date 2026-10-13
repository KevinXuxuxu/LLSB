"""Minimal OpenAI-compatible mock server for testing the orchestrator's
lifecycle (launch/health/stages/teardown) without touching real models.

Endpoints: GET /v1/models, POST /v1/chat/completions (stream and non-stream).
Responses are canned; scores are meaningless — lifecycle is the test.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": "mock-1", "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        stream = req.get("stream", False)
        if not stream:
            self._send(200, {
                "id": "mock", "object": "chat.completion", "model": "mock-1",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "mock reply", "tool_calls": []}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
                          "completion_tokens_details": {"reasoning_tokens": 0}},
            })
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ("mock ", "reply"):
            chunk = {"id": "mock", "object": "chat.completion.chunk", "model": "mock-1",
                     "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        final = {"id": "mock", "object": "chat.completion.chunk", "model": "mock-1",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
                           "completion_tokens_details": {"reasoning_tokens": 0}}}
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9911
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()

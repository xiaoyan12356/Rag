"""A stand-in for the embedding provider.

The end-to-end tests need an embedding provider that answers without a
network and without credentials. This server answers every request with the
same vector, which is enough to let the whole pipeline run.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIMENSION = 8


class StubEmbeddingHandler(BaseHTTPRequestHandler):
    """Answers every embedding request with the same vector."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({"embedding": [0.25] * DIMENSION}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def start_stub_server() -> tuple[str, ThreadingHTTPServer]:
    """Start the server and return its address."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubEmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


def stop_stub_server(server: ThreadingHTTPServer) -> None:
    """Stop the server."""
    server.shutdown()
    server.server_close()

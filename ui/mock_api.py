"""HTTP mock of the backend API: ``python -m ui.mock_api`` (``make mock-api``).

Serves :class:`ui.mock_backend.MockBackend` on ``http://127.0.0.1:8000/api/v1`` with the
standard library only, so the UI (and teammates' curl checks) can run with no backend.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from ui.mock_backend import MockBackend
from ui.settings import load_ui_config


def make_handler(backend: MockBackend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ui-mock-api/1.0"

        def _dispatch(self, method: str) -> None:
            url = urlsplit(self.path)
            body: Any = None
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    self._send(400, {"error": {"code": "validation", "message": "invalid JSON"}})
                    return
            resp = backend.handle(
                method,
                url.path,
                params=dict(parse_qsl(url.query)),
                body=body,
                headers=dict(self.headers.items()),
            )
            self._send(resp.status, resp.body, resp.content_type)

        def _send(self, status: int, body: Any, content_type: str = "application/json") -> None:
            data = (
                json.dumps(body).encode("utf-8")
                if content_type == "application/json"
                else str(body).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # never log request bodies (they may hold ticket text); method + path only
            print(f"mock-api {self.command} {urlsplit(self.path).path} -> {args[1]}")

    return Handler


def make_server(host: str, port: int, backend: MockBackend | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(backend or MockBackend()))


def main(argv: list[str] | None = None) -> None:
    cfg = load_ui_config()["mock_api"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=cfg["host"])
    parser.add_argument("--port", type=int, default=int(cfg["port"]))
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    print(f"Mock API (golden fixtures) on http://{args.host}:{args.port}/api/v1 - Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

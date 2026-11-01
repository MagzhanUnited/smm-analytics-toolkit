"""Read-only HTTP service used for the test deployment.

Routes:
    GET /            the markdown report rendered at start-up
    GET /healthz     liveness/readiness probe: ``{"status": "ok", "version": ...}``
    GET /api/metrics engagement statistics as JSON, ``?by=channel|format|weekday|hour``
"""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from smmkit import __version__
from smmkit.metrics import GROUPINGS, summarise


def make_handler(posts, report_md: str):
    """Build a request handler bound to an already loaded dataset and report."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"smmkit/{__version__}"

        def _send(self, status: int, body: str, content_type: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status: int, payload) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (name required by BaseHTTPRequestHandler)
            url = urlparse(self.path)
            if url.path == "/healthz":
                self._json(200, {"status": "ok", "version": __version__, "posts": len(posts)})
            elif url.path == "/api/metrics":
                by = parse_qs(url.query).get("by", ["channel"])[0]
                if by not in GROUPINGS:
                    self._json(400, {"error": f"unknown grouping {by!r}", "allowed": list(GROUPINGS)})
                    return
                self._json(200, [asdict(s) for s in summarise(posts, by=by)])
            elif url.path == "/":
                self._send(200, report_md, "text/markdown")
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, fmt: str, *args) -> None:  # one line per request on stdout
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def make_server(posts, report_md: str, host: str = "0.0.0.0", port: int = 8080) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(posts, report_md))

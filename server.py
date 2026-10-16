#!/usr/bin/env python3
"""HTTP interface, on the standard library only (no uvicorn/FastAPI needed).

    python server.py                 # listens on 127.0.0.1:8000
    curl -s localhost:8000/healthz
    curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "How do I reset my password?"}'

The knowledge base is loaded once at startup (LOAD_KB) and reused per request.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from support_answer.config import ANSWERER_MODES, DEFAULT_KB_PATH, Settings
from support_answer.kb import KbError
from support_answer.pipeline import SupportAnswerService

MAX_BODY_BYTES = 64 * 1024


class AskHandler(BaseHTTPRequestHandler):
    """Routes: ``POST /ask`` and ``GET /healthz``."""

    server_version = "support-answer/1.0"
    service: SupportAnswerService  # injected below

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") in ("/healthz", "/health"):
            self._send(
                200,
                {
                    "status": "ok",
                    "kb_path": self.service.kb_path,
                    "documents": len(self.service.documents),
                    "passages": len(self.service.passages),
                    "answerer": self.service.settings.answerer,
                },
            )
            return
        self._send(404, {"error": f"no such route: GET {self.path}", "routes": ["POST /ask", "GET /healthz"]})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.rstrip("/") != "/ask":
            self._send(404, {"error": f"no such route: POST {self.path}", "routes": ["POST /ask"]})
            return

        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            self._send(400, {"error": "invalid content-length header"})
            return
        if length <= 0:
            self._send(400, {"error": 'request body required: {"question": "..."}'})
            return
        if length > MAX_BODY_BYTES:
            self._send(413, {"error": "request body too large"})
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"invalid JSON body: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": 'body must be a JSON object: {"question": "..."}'})
            return

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            self._send(400, {"error": "'question' must be a non-empty string"})
            return

        try:
            response = self.service.ask(question)
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        self._send(200, response)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="server.py", description=__doc__)
    parser.add_argument("--kb", default=DEFAULT_KB_PATH, help="path to kb.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--answerer", default="rule", choices=list(ANSWERER_MODES))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        service = SupportAnswerService.from_path(args.kb, Settings(answerer=args.answerer))
    except KbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    handler = type("BoundAskHandler", (AskHandler,), {"service": service})
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"loaded {len(service.documents)} documents from {args.kb!r} "
        f"({len(service.passages)} passages); answerer={args.answerer}",
        file=sys.stderr,
    )
    print(f"listening on http://{args.host}:{args.port}  (POST /ask)", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

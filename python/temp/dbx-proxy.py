#!/usr/bin/env python3
"""Local Chat Completions proxy for Databricks Model Serving."""

from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_PORT = 19000


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def databricks_url() -> str:
    explicit_url = env("DATABRICKS_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    host = env("DBX_WORKSPACE_HOST")
    endpoint = env("DBX_ENDPOINT")
    if not host or not endpoint:
        raise RuntimeError(
            "Set DBX_WORKSPACE_HOST and DBX_ENDPOINT, or set DATABRICKS_URL."
        )

    host = host.removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{host}/serving-endpoints/{endpoint}/invocations"


def proxy_port() -> int:
    try:
        return int(env("DBX_PROXY_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
    except ValueError as exc:
        raise RuntimeError("DBX_PROXY_PORT must be an integer.") from exc


TOKEN = env("DBX_TOKEN")
ENDPOINT = env("DBX_ENDPOINT")
MODEL = env("DBX_MODEL", ENDPOINT)
TARGET_URL = databricks_url()
PORT = proxy_port()
TIMEOUT_SECONDS = int(env("DBX_PROXY_TIMEOUT_SECONDS", "300") or "300")


def build_opener() -> urllib.request.OpenerDirector:
    """Build an opener that bypasses system proxies for Databricks traffic."""
    insecure = env("DBX_INSECURE_SKIP_VERIFY", "0") in {"1", "true", "yes"}
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=context),
    )


OPENER = build_opener()


class DatabricksProxy(BaseHTTPRequestHandler):
    server_version = "dbx-proxy/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in {"/", "/healthz"}:
            self.send_json(200, {"status": "ok", "target": TARGET_URL})
            return
        self.send_json(404, {"error": f"Unsupported path: {self.path}"})

    def do_POST(self) -> None:
        if not TOKEN:
            self.send_json(500, {"error": "DBX_TOKEN is not set."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self.send_json(400, {"error": f"Invalid JSON request body: {exc}"})
            return

        if not isinstance(body, dict):
            self.send_json(400, {"error": "Request body must be a JSON object."})
            return

        body = self.normalize_request(body)
        print(
            f"REQUEST path={self.path} stream={body.get('stream', False)} "
            f"model={body.get('model')}",
            flush=True,
        )

        request = urllib.request.Request(
            TARGET_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "Accept": self.headers.get("Accept", "application/json"),
            },
            method="POST",
        )

        try:
            with OPENER.open(request, timeout=TIMEOUT_SECONDS) as response:
                result = response.read()
                content_type = response.headers.get("Content-Type", "application/json")
                self.send_response(response.status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)
        except urllib.error.HTTPError as exc:
            result = exc.read()
            content_type = exc.headers.get("Content-Type", "application/json")
            self.send_response(exc.code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)
        except Exception as exc:  # noqa: BLE001 - return proxy failures to caller
            print(f"ERROR unexpected: {type(exc).__name__}: {exc}", flush=True)
            self.send_json(502, {"error": str(exc), "type": type(exc).__name__})

    @staticmethod
    def normalize_request(body: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(body)

        if "temperature" in cleaned:
            cleaned.pop("top_p", None)

        cleaned.pop("stream_options", None)

        if MODEL:
            cleaned["model"] = MODEL

        return cleaned


def main() -> int:
    if not TOKEN:
        print("Error: DBX_TOKEN is not set.", file=sys.stderr)
        return 1
    if not MODEL:
        print("Error: set DBX_ENDPOINT or DBX_MODEL.", file=sys.stderr)
        return 1

    print(f"Proxy running on http://localhost:{PORT}", flush=True)
    print(f"Forwarding to {TARGET_URL}", flush=True)
    ThreadingHTTPServer(("localhost", PORT), DatabricksProxy).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared fixtures and helpers for opencode-sync tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Mock vLLM HTTP server
# ---------------------------------------------------------------------------

class _MockVLLMHandler(BaseHTTPRequestHandler):
    """Minimal handler that serves GET /v1/models."""

    models: List[str] = []
    fail_with: int = 0  # If non-zero, return this HTTP status instead

    def do_GET(self):
        if self.fail_with:
            self.send_response(self.fail_with)
            self.end_headers()
            return

        if self.path == "/v1/models":
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"id": m, "object": "model", "owned_by": "vllm"}
                        for m in self.models
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_):  # suppress request logging during tests
        pass


class MockVLLMServer:
    def __init__(self, models: List[str], fail_with: int = 0):
        # Per-server handler subclass: models/fail_with are class attributes, so
        # sharing _MockVLLMHandler would let a second server clobber the first's
        # model list.  Sync-all tests need two servers with different lists.
        handler = type(
            "_ScopedMockVLLMHandler",
            (_MockVLLMHandler,),
            {"models": list(models), "fail_with": fail_with},
        )
        self._server = HTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def shutdown(self):
        self._server.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


@pytest.fixture()
def mock_server():
    """Factory fixture: call with a model list to spin up a mock vLLM server."""
    servers = []

    def factory(models: List[str], fail_with: int = 0) -> MockVLLMServer:
        srv = MockVLLMServer(models, fail_with=fail_with)
        servers.append(srv)
        return srv

    yield factory

    for srv in servers:
        srv.shutdown()


# ---------------------------------------------------------------------------
# Sample config text
# ---------------------------------------------------------------------------

SAMPLE_JSONC = """\
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM (local)",
      "options": {
        // The endpoint URL
        "baseURL": "http://localhost:8080/v1"
      },
      "models": {
        // MUST match the --model argument
        "org/model-a": {
          "name": "Model A"
        },
        "org/model-b": {
          "name": "Model B"
        }
      }
    }
  },
  "model": "vllm/org/model-a",
  "small_model": "vllm/org/model-a"
}
"""

SAMPLE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "vllm": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "vLLM (local)",
            "options": {"baseURL": "http://localhost:8080/v1"},
            "models": {
                "org/model-a": {"name": "Model A"},
                "org/model-b": {"name": "Model B"},
            },
        }
    },
    "model": "vllm/org/model-a",
    "small_model": "vllm/org/model-a",
}

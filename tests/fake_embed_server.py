#!/usr/bin/env python3
"""fake_embed_server.py — stdlib-only stand-in for an embedding endpoint.

Serves both wire protocols `hermes_pgvector/embed.py` speaks:

- ``POST /v1/embeddings``  (OpenAI-compatible)  -> ``{"data": [{"embedding": [...]}, ...]}``
- ``POST /api/embed``      (Ollama-native)       -> ``{"embeddings": [[...], ...]}``

Vectors are deterministic: the same input text always produces the same
unit-length vector (SHA-256 of the text, expanded with a counter to reach
`dim` floats, then L2-normalised). Different text is *not* guaranteed to
produce a nearby vector -- this is a wire-protocol fake, not a semantic one.

Request `input` may be a JSON string or a list of strings (both protocols);
the response always has one embedding per input, in order.

No third-party dependencies: only the standard library, so it runs with any
Python >= 3.11 and needs nothing installed -- `scripts/test-env.sh` invokes it
with plain `python3`.

CLI:
    python -m tests.fake_embed_server [--host 127.0.0.1] [--port 11999]
        [--dim 768] [--delay SECONDS] [--fail-substring TEXT]
        [--require-bearer TOKEN]

Same knobs via environment variables (CLI flag wins when both are given):
    FAKE_EMBED_HOST, FAKE_EMBED_PORT, FAKE_EMBED_DIM, FAKE_EMBED_DELAY,
    FAKE_EMBED_FAIL_SUBSTRING, FAKE_EMBED_REQUIRE_BEARER

Also importable for tests -- see `FakeEmbedServer`, which starts the same
handler in-process on a background daemon thread (port 0 = pick a free
port), so unit tests need no subprocess and no script:

    srv = FakeEmbedServer(dim=32)
    srv.start()
    ...  # srv.base_url == "http://127.0.0.1:<port>"
    srv.stop()
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Union

InputType = Union[str, List[str]]


# ---------------------------------------------------------------------------
# Deterministic vectors
# ---------------------------------------------------------------------------

def deterministic_vector(text: str, dim: int) -> List[float]:
    """Unit-length vector of length `dim`, deterministic in `text`.

    Same text -> same vector (bit-for-bit); different text -> (almost
    certainly) a different vector. Built from SHA-256(f"{text}:{counter}"),
    walking `counter` up whenever more bytes are needed than one digest
    holds, so any `dim` is reachable.
    """
    floats: List[float] = []
    counter = 0
    while len(floats) < dim:
        digest = hashlib.sha256(f"{text}\0{counter}".encode("utf-8")).digest()
        for i in range(0, len(digest) - 3, 4):
            if len(floats) >= dim:
                break
            chunk = int.from_bytes(digest[i:i + 4], "big", signed=False)
            floats.append((chunk / 0xFFFFFFFF) * 2.0 - 1.0)
        counter += 1
    norm = math.sqrt(sum(x * x for x in floats)) or 1.0
    return [x / norm for x in floats]


def _as_list(value: InputType) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "FakeEmbedServer/1.0"

    # Quiet by default -- tests run many requests and stderr noise from the
    # default BaseHTTPRequestHandler logger obscures real failures.
    def log_message(self, fmt, *args):  # noqa: D401 -- stdlib override
        if os.environ.get("FAKE_EMBED_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _config(self):
        return self.server.fake_embed_config  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw or b"{}")
        except (json.JSONDecodeError, ValueError):
            return {}

    def do_POST(self):  # noqa: N802 -- stdlib naming
        cfg = self._config()

        require_bearer = cfg.get("require_bearer")
        if require_bearer:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {require_bearer}":
                self._send_json(401, {"error": "missing or invalid bearer token"})
                return

        delay = float(cfg.get("delay") or 0.0)
        if delay > 0:
            time.sleep(delay)

        body = self._read_body()
        raw_input = body.get("input")
        if raw_input is None:
            self._send_json(400, {"error": "missing 'input'"})
            return
        inputs = _as_list(raw_input)

        fail_substring = cfg.get("fail_substring")
        if fail_substring and any(fail_substring in text for text in inputs):
            self._send_json(400, {"error": f"input contains disallowed substring {fail_substring!r}"})
            return

        dim = int(cfg.get("dim") or 768)
        vectors = [deterministic_vector(text, dim) for text in inputs]

        if self.path.rstrip("/") == "/v1/embeddings":
            self._send_json(200, {
                "data": [
                    {"embedding": vec, "index": i, "object": "embedding"}
                    for i, vec in enumerate(vectors)
                ],
                "model": body.get("model") or "fake-embed",
                "object": "list",
            })
        elif self.path.rstrip("/") == "/api/embed":
            self._send_json(200, {
                "embeddings": vectors,
                "model": body.get("model") or "fake-embed",
            })
        else:
            self._send_json(404, {"error": f"unknown path {self.path}"})


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# In-process server, for tests/conftest.py
# ---------------------------------------------------------------------------

class FakeEmbedServer:
    """Runs `_Handler` on a background daemon thread. Safe to start/stop per test."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        dim: int = 768,
        delay: float = 0.0,
        fail_substring: Optional[str] = None,
        require_bearer: Optional[str] = None,
    ):
        self.host = host
        self._requested_port = port
        self._config = {
            "dim": dim,
            "delay": delay,
            "fail_substring": fail_substring,
            "require_bearer": require_bearer,
        }
        self._httpd: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("server not started")
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "FakeEmbedServer":
        if self._httpd is not None:
            return self
        httpd = _Server((self.host, self._requested_port), _Handler)
        httpd.fake_embed_config = self._config  # type: ignore[attr-defined]
        self._httpd = httpd
        thread = threading.Thread(target=httpd.serve_forever, name="fake-embed-server", daemon=True)
        thread.start()
        self._thread = thread
        return self

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None

    def __enter__(self) -> "FakeEmbedServer":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fake_embed_server",
        description="stdlib-only fake embedding endpoint for tests and the local harness",
    )
    p.add_argument("--host", default=os.environ.get("FAKE_EMBED_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=_env_int("FAKE_EMBED_PORT", 11999))
    p.add_argument("--dim", type=int, default=_env_int("FAKE_EMBED_DIM", 768))
    p.add_argument(
        "--delay", type=float, default=_env_float("FAKE_EMBED_DELAY", 0.0),
        help="seconds to sleep before answering each request (timeout tests)",
    )
    p.add_argument(
        "--fail-substring", default=os.environ.get("FAKE_EMBED_FAIL_SUBSTRING") or None,
        help="answer HTTP 400 when any input contains this substring",
    )
    p.add_argument(
        "--require-bearer", default=os.environ.get("FAKE_EMBED_REQUIRE_BEARER") or None,
        help="answer HTTP 401 unless Authorization: Bearer <this> is sent",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = FakeEmbedServer(
        host=args.host,
        port=args.port,
        dim=args.dim,
        delay=args.delay,
        fail_substring=args.fail_substring,
        require_bearer=args.require_bearer,
    )
    server.start()
    print(f"fake_embed_server listening on {server.base_url} (dim={args.dim})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

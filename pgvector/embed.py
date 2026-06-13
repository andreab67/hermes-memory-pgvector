"""embed.py — minimal embedding client for the pgvector memory plugin.

Posts to an OpenAI-compatible /v1/embeddings or Ollama native /api/embed
endpoint and returns a list of floats. No retries beyond a single attempt
— callers decide what to do with failures (we want fail-soft, not retry
storms — that was Honcho's mistake).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding endpoint fails to return a usable vector."""


def embed(
    text: str,
    *,
    base_url: str,
    model: str = "nomic-embed-text",
    timeout: float = 10.0,
    retries: int = 0,
    backoff: float = 0.1,
) -> List[float]:
    """Return a 768-dim embedding for `text`.

    Tries the OpenAI-compatible `/v1/embeddings` path first; falls back to
    Ollama's native `/api/embed`. Raises EmbeddingError on any failure.

    `retries` (default 0 — single attempt) bounds re-tries on EmbeddingError
    with exponential backoff (`backoff * 2**i`). v0.4.0: ONLY the background
    AsyncWriter drain path passes ``retries>0`` — prefetch, the recall tools,
    and sync_turn keep the default single attempt so the agent loop never
    waits on a slow embed endpoint (invariant #2: no retry storms in the hot
    path; durable recovery is the backfill sweep, not inline retries).
    """
    if not text or not text.strip():
        raise EmbeddingError("empty input")

    # Trim to avoid the n_ctx_train=2048 cliff on nomic — at ~4 chars/token
    # that's ~8000 chars. Keep a safety margin.
    if len(text) > 6000:
        text = text[:6000]
    base_url = base_url.rstrip("/")

    attempts = max(0, int(retries)) + 1
    last_exc: Optional[EmbeddingError] = None
    for i in range(attempts):
        try:
            return _embed_once(text, base_url=base_url, model=model, timeout=timeout)
        except EmbeddingError as exc:
            last_exc = exc
            if i + 1 < attempts:
                time.sleep(backoff * (2 ** i))
    raise last_exc if last_exc else EmbeddingError("embed failed")


def _embed_once(text: str, *, base_url: str, model: str, timeout: float) -> List[float]:
    """One embedding attempt: OpenAI-compatible path, then Ollama-native."""
    # Path A: OpenAI-compatible
    try:
        return _post(
            f"{base_url}/v1/embeddings",
            {"model": model, "input": text},
            timeout=timeout,
            extract=lambda d: d["data"][0]["embedding"],
        )
    except EmbeddingError as exc:
        logger.debug("OpenAI-compat embed failed (%s); trying native", exc)

    # Path B: Ollama native
    return _post(
        f"{base_url}/api/embed",
        {"model": model, "input": text},
        timeout=timeout,
        extract=lambda d: (d.get("embeddings") or [d.get("embedding")])[0],
    )


def _post(url: str, body: dict, *, timeout: float, extract) -> List[float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise EmbeddingError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(f"connection failed: {exc.reason}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise EmbeddingError(f"invalid JSON response: {exc}") from exc

    try:
        vec = extract(payload)
    except (KeyError, IndexError, TypeError) as exc:
        raise EmbeddingError(f"unexpected response shape: {exc}") from exc

    if not isinstance(vec, list) or not vec:
        raise EmbeddingError("response had no embedding array")
    if len(vec) != 768:
        raise EmbeddingError(f"expected 768 dims, got {len(vec)}")
    return vec


def to_pgvector_literal(vec: List[float]) -> str:
    """Render a Python list of floats as a pgvector input literal.

    psycopg can also handle this via type adapters, but the literal form
    keeps the plugin dependency-light.
    """
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"

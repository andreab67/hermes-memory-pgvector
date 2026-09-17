"""embed.py — minimal embedding client for the pgvector memory plugin.

Posts to an OpenAI-compatible /v1/embeddings or Ollama native /api/embed
endpoint and returns a list of floats. No retries beyond a single attempt
— callers decide what to do with failures (we want fail-soft, not retry
storms — that was Honcho's mistake).

v0.5.3: the vector dimension (`dim`, default 768), an optional bearer token
read from a NAMED environment variable (`api_key_env`), and the wire
protocol (`protocol`: auto | openai | ollama) are parameters instead of
hard-coded, so a hosted OpenAI-compatible endpoint such as OpenRouter works.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Dimension of the reference model (nomic-embed-text) and of the vector(768)
# columns created by migrations/001_schema.sql. The default everywhere, so a
# deployment that sets nothing behaves exactly as before v0.5.3.
DEFAULT_EMBED_DIM = 768

# auto   -- OpenAI-compatible /v1/embeddings, then Ollama-native /api/embed
# openai -- /v1/embeddings only; errors (401, unknown model) surface as-is
# ollama -- /api/embed only
EMBED_PROTOCOLS = ("auto", "openai", "ollama")

# One warning per distinct misconfiguration, not one per embed call.
_warned_protocols: set = set()
_warned_key_envs: set = set()


def normalize_protocol(protocol: Optional[str]) -> str:
    """Return a valid protocol name; unknown values fall back to 'auto'.

    Unset/empty means 'auto'. An unrecognised value is logged ONCE and treated
    as 'auto' rather than raised, because this is reached from fail-soft hook
    paths (invariant #4).
    """
    value = str(protocol).strip().lower() if protocol is not None else ""
    if not value:
        return "auto"
    if value in EMBED_PROTOCOLS:
        return value
    if value not in _warned_protocols:
        _warned_protocols.add(value)
        logger.warning(
            "embed_protocol %r is not one of %s; using 'auto'",
            protocol,
            "/".join(EMBED_PROTOCOLS),
        )
    return "auto"


def _auth_headers(api_key_env: Optional[str]) -> Dict[str, str]:
    """Authorization header from the env var NAMED by api_key_env, or {}.

    Read at call time (never cached), so rotating the variable in the service
    environment needs no code path to re-read config. The key VALUE must never
    reach a log line, an exception message, or a repr: only the variable
    NAME is ever mentioned.
    """
    name = (api_key_env or "").strip()
    if not name:
        return {}
    value = (os.environ.get(name) or "").strip()
    if not value:
        if name not in _warned_key_envs:
            _warned_key_envs.add(name)
            logger.warning(
                "embed_api_key_env names $%s, but it is unset or empty; "
                "sending no Authorization header",
                name,
            )
        return {}
    if any(ch in value for ch in "\r\n\x00"):
        # http.client would reject this with a ValueError that echoes the
        # header value -- i.e. the key -- verbatim. Refuse it here instead.
        raise EmbeddingError(
            f"the API key in ${name} contains a control character; refusing to send it"
        )
    return {"Authorization": f"Bearer {value}"}


class EmbeddingError(Exception):
    """Raised when the embedding endpoint fails to return a usable vector."""


class EmbeddingDimensionError(EmbeddingError):
    """Endpoint reachable but returned wrong-dimension vectors.

    Distinct from generic EmbeddingError so the protocol-fallback path can
    tell "endpoint down, try the other protocol" apart from "endpoint fine,
    the MODEL is misconfigured" — the latter must surface as-is (invariant
    #7), not be masked by a 404 from retrying the other protocol (v0.4.2).
    """


def embed(
    text: str,
    *,
    base_url: str,
    model: str = "nomic-embed-text",
    timeout: float = 10.0,
    retries: int = 0,
    backoff: float = 0.1,
    max_total: Optional[float] = None,
    dim: int = DEFAULT_EMBED_DIM,
    api_key_env: Optional[str] = None,
    protocol: Optional[str] = "auto",
) -> List[float]:
    """Return a `dim`-length embedding for `text` (default 768).

    `protocol` 'auto' (default) tries the OpenAI-compatible `/v1/embeddings`
    path first and falls back to Ollama's native `/api/embed`; 'openai' and
    'ollama' use only their own path. Raises EmbeddingError on any failure,
    and EmbeddingDimensionError when the endpoint answers with a vector whose
    length is not `dim`.

    `api_key_env` is the NAME of an environment variable. When it names a
    non-empty variable, its value is sent as `Authorization: Bearer <value>`;
    otherwise no Authorization header is sent.

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
    try:
        dim = int(dim)
    except (TypeError, ValueError):
        raise EmbeddingError(f"invalid embedding dimension: {dim!r}") from None
    if dim <= 0:
        raise EmbeddingError(f"invalid embedding dimension: {dim}")
    protocol = normalize_protocol(protocol)

    attempts = max(0, int(retries)) + 1
    last_exc: Optional[EmbeddingError] = None
    started = time.monotonic()
    for i in range(attempts):
        try:
            return _embed_once(
                text,
                base_url=base_url,
                model=model,
                timeout=timeout,
                dim=dim,
                api_key_env=api_key_env,
                protocol=protocol,
            )
        except EmbeddingError as exc:
            last_exc = exc
            if i + 1 >= attempts:
                break
            # Deadline, not just a per-attempt timeout. Each attempt can cost up
            # to 2 x `timeout` (the OpenAI-compat path then the Ollama-native
            # fallback), so retries multiply a SLOW endpoint into minutes of
            # blocking on the caller's thread -- for the writer drain that means
            # the bounded queue fills and starts dropping writes. Transient
            # failures, which is what retries are actually for, fail fast and
            # are unaffected by this.
            if max_total is not None and (time.monotonic() - started) >= max_total:
                logger.debug(
                    "embed: giving up after %.1fs (budget %.1fs), %d/%d attempts",
                    time.monotonic() - started, max_total, i + 1, attempts,
                )
                break
            time.sleep(backoff * (2 ** i))
    raise last_exc if last_exc else EmbeddingError("embed failed")


def _embed_once(
    text: str,
    *,
    base_url: str,
    model: str,
    timeout: float,
    dim: int = DEFAULT_EMBED_DIM,
    api_key_env: Optional[str] = None,
    protocol: Optional[str] = "auto",
) -> List[float]:
    """One embedding attempt over the selected protocol.

    'auto': OpenAI-compatible path, then Ollama-native. 'openai' / 'ollama':
    that path only -- no fallback, so an auth or unknown-model error from a
    hosted endpoint is reported as-is instead of being replaced by the 404 the
    other protocol's path would return.
    """
    protocol = normalize_protocol(protocol)

    # Path A: OpenAI-compatible
    if protocol in ("auto", "openai"):
        try:
            return _post(
                f"{base_url}/v1/embeddings",
                {"model": model, "input": text},
                timeout=timeout,
                extract=lambda d: d["data"][0]["embedding"],
                dim=dim,
                api_key_env=api_key_env,
            )
        except EmbeddingDimensionError:
            # The endpoint answered — the model is just wrong-dimensional.
            # Falling through to the native protocol would 404 on a pure
            # OpenAI-compatible server and bury the actionable message.
            raise
        except EmbeddingError as exc:
            if protocol == "openai":
                raise
            logger.debug("OpenAI-compat embed failed (%s); trying native", exc)

    # Path B: Ollama native
    return _post(
        f"{base_url}/api/embed",
        {"model": model, "input": text},
        timeout=timeout,
        extract=lambda d: (d.get("embeddings") or [d.get("embedding")])[0],
        dim=dim,
        api_key_env=api_key_env,
    )


def _post(
    url: str,
    body: dict,
    *,
    timeout: float,
    extract,
    dim: int = DEFAULT_EMBED_DIM,
    api_key_env: Optional[str] = None,
) -> List[float]:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(_auth_headers(api_key_env))
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    # A ValueError raised while SENDING (e.g. http.client rejecting a header)
    # can quote the header value -- the API key. Flag it here and raise outside
    # the except block, so the original exception is neither chained nor
    # attached as __context__.
    rejected_request = False
    raw = b""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise EmbeddingError(f"HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(f"connection failed: {exc.reason}") from exc
    except (OSError, http.client.HTTPException) as exc:
        # urllib only wraps errors raised while sending the request. A server
        # that accepts the connection but answers slower than `timeout` raises
        # a bare TimeoutError from getresponse(), which used to escape as a
        # non-EmbeddingError: past the hot path's `except EmbeddingError`, and
        # past the protocol fallback and the writer's retries.
        raise EmbeddingError(f"connection failed: {type(exc).__name__}: {exc}") from exc
    except ValueError:
        rejected_request = True
    if rejected_request:
        raise EmbeddingError(f"request rejected before sending to {url}")

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise EmbeddingError(f"invalid JSON response: {exc}") from exc

    try:
        vec = extract(payload)
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise EmbeddingError(f"unexpected response shape: {exc}") from exc

    if not isinstance(vec, list) or not vec:
        raise EmbeddingError("response had no embedding array")
    if len(vec) != dim:
        raise EmbeddingDimensionError(
            f"expected {dim} dims (embed_dim), got {len(vec)} -- embed_model, "
            "embed_dim and the vector(N) columns must all agree"
        )
    return vec


def to_pgvector_literal(vec: List[float]) -> str:
    """Render a Python list of floats as a pgvector input literal.

    psycopg can also handle this via type adapters, but the literal form
    keeps the plugin dependency-light.
    """
    return "[" + ",".join(f"{x:.6g}" for x in vec) + "]"

"""Unit tests for tests/fake_embed_server.py and the `fake_embed_server` fixture.

None of these need a database or a real embedding endpoint -- they run
everywhere, including in the `unit` CI job. They exist so the harness itself
(WP-0) has coverage, and so a later change to the fake server's wire shapes
fails loudly here instead of silently in a DB-gated test that might be
skipped in some environments.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_embed_server import FakeEmbedServer, deterministic_vector  # noqa: E402


def _post(url: str, payload: dict, headers=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# deterministic_vector()
# ---------------------------------------------------------------------------

def test_deterministic_vector_same_text_same_vector():
    a = deterministic_vector("hello world", 32)
    b = deterministic_vector("hello world", 32)
    assert a == b
    assert len(a) == 32


def test_deterministic_vector_different_text_different_vector():
    a = deterministic_vector("hello world", 32)
    b = deterministic_vector("goodbye world", 32)
    assert a != b


def test_deterministic_vector_is_unit_length():
    vec = deterministic_vector("normalise me", 64)
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_deterministic_vector_any_dim():
    # dim not a multiple of 8 (sha256 digest / 4-byte chunks) still works.
    assert len(deterministic_vector("odd dim", 5)) == 5
    assert len(deterministic_vector("odd dim", 769)) == 769


# ---------------------------------------------------------------------------
# HTTP surface, via the in-process fixture
# ---------------------------------------------------------------------------

def test_openai_endpoint_shape(fake_embed_server):
    url = fake_embed_server(dim=16)
    status, body = _post(f"{url}/v1/embeddings", {"model": "x", "input": "hello"})
    assert status == 200
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == 16


def test_ollama_endpoint_shape(fake_embed_server):
    url = fake_embed_server(dim=16)
    status, body = _post(f"{url}/api/embed", {"model": "x", "input": "hello"})
    assert status == 200
    assert len(body["embeddings"]) == 1
    assert len(body["embeddings"][0]) == 16


def test_list_input_returns_one_embedding_per_item(fake_embed_server):
    url = fake_embed_server(dim=8)
    status, body = _post(f"{url}/v1/embeddings", {"model": "x", "input": ["a", "b", "c"]})
    assert status == 200
    assert len(body["data"]) == 3
    # same text -> same vector even across a batched call
    status2, body2 = _post(f"{url}/api/embed", {"model": "x", "input": ["a", "a"]})
    assert body2["embeddings"][0] == body2["embeddings"][1]


def test_fail_substring_returns_400(fake_embed_server):
    url = fake_embed_server(fail_substring="denied")
    status, body = _post(f"{url}/v1/embeddings", {"model": "x", "input": "this is denied"})
    assert status == 400
    status_ok, _ = _post(f"{url}/v1/embeddings", {"model": "x", "input": "this is fine"})
    assert status_ok == 200


def test_require_bearer_401_without_200_with(fake_embed_server):
    url = fake_embed_server(require_bearer="secrettoken")
    status, _ = _post(f"{url}/v1/embeddings", {"model": "x", "input": "hi"})
    assert status == 401
    status_ok, _ = _post(
        f"{url}/v1/embeddings", {"model": "x", "input": "hi"},
        headers={"Authorization": "Bearer secrettoken"},
    )
    assert status_ok == 200


def test_delay_is_respected(fake_embed_server):
    url = fake_embed_server(delay=0.3)
    started = time.monotonic()
    status, _ = _post(f"{url}/api/embed", {"model": "x", "input": "slow"})
    elapsed = time.monotonic() - started
    assert status == 200
    assert elapsed >= 0.25


def test_two_calls_to_the_fixture_start_independent_servers(fake_embed_server):
    url_a = fake_embed_server(dim=8)
    url_b = fake_embed_server(dim=8)
    assert url_a != url_b


# ---------------------------------------------------------------------------
# Integration with hermes_pgvector.embed.embed() -- the real client code
# ---------------------------------------------------------------------------

def test_embed_module_against_fake_server_openai_protocol(fake_embed_server):
    from hermes_pgvector.embed import embed

    url = fake_embed_server(dim=24)
    vec = embed("hello world", base_url=url, dim=24, protocol="openai")
    assert len(vec) == 24


def test_embed_module_against_fake_server_ollama_protocol(fake_embed_server):
    from hermes_pgvector.embed import embed

    url = fake_embed_server(dim=24)
    vec = embed("hello world", base_url=url, dim=24, protocol="ollama")
    assert len(vec) == 24


def test_embed_module_against_fake_server_dimension_mismatch_raises(fake_embed_server):
    from hermes_pgvector.embed import embed, EmbeddingDimensionError

    url = fake_embed_server(dim=24)
    with pytest.raises(EmbeddingDimensionError):
        embed("hello world", base_url=url, dim=768, protocol="openai")


def test_fake_embed_server_context_manager_starts_and_stops():
    with FakeEmbedServer(dim=8) as srv:
        status, body = _post(f"{srv.base_url}/api/embed", {"model": "x", "input": "hi"})
        assert status == 200
        assert len(body["embeddings"][0]) == 8

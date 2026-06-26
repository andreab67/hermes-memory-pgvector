"""tests/test_embedder.py — comprehensive tests for hexus.embedder.

Two layers:
  1. Pure-Python / structural tests (no model load) — fast, run always.
  2. Real-model tests (load sentence-transformers MiniLM-L6-v2) — slow on
     first run (~1-2s model load + ~90MB download) but cached on subsequent
     runs. Skipped if SENTENCE_TRANSFORMERS_SKIP_REAL=1.

The real-model tests are integration tests against the actual library;
unit-level mocking the model would test that our mocks are correct, not
that the library works.
"""

from __future__ import annotations

import os
import threading
from typing import List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Structural / fast tests — no model load
# ---------------------------------------------------------------------------

def test_constants():
    """The public constants are pinned to the values the rest of the
    code (and the schema migration) assume."""
    from hexus.embedder import DEFAULT_MODEL, DEFAULT_DIM
    assert DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert DEFAULT_DIM == 384


def test_embedder_import_is_fast():
    """Importing the embedder module does NOT load the model. Verified
    by ensuring `is_loaded` is False on a fresh instance.

    This guards against an accidental eager load in __init__ that would
    slow down plugin import (and break hermes-agent's startup time).
    """
    from hexus.embedder import LocalBertEmbedder
    e = LocalBertEmbedder()
    assert e.is_loaded is False
    assert e.dim == 384  # constant, not from loaded model


def test_embedder_custom_model_constant_dim():
    """For a non-default model, dim returns 0 until loaded (we don't
    know the dim ahead of time)."""
    from hexus.embedder import LocalBertEmbedder
    e = LocalBertEmbedder(model_name="some/custom-model")
    assert e.dim == 0
    assert e.is_loaded is False


def test_embed_empty_list_returns_empty():
    """Passing an empty list returns an empty list (no model load)."""
    from hexus.embedder import LocalBertEmbedder
    e = LocalBertEmbedder()
    assert e.embed([]) == []
    # Still not loaded — empty input doesn't trigger load.
    assert e.is_loaded is False


def test_embed_filters_whitespace_only():
    """All-whitespace input filters to empty, returns empty. No model load."""
    from hexus.embedder import LocalBertEmbedder
    e = LocalBertEmbedder()
    assert e.embed(["", "   ", "\n\t  "]) == []
    assert e.is_loaded is False


def test_embed_disables_sentence_transformers_progress_bar():
    """Hermes owns progress reporting; the embedder must not print tqdm bars."""
    from hexus.embedder import LocalBertEmbedder

    class ArrayLike:
        def tolist(self):
            return [[1.0, 2.0, 3.0]]

    class FakeModel:
        def __init__(self):
            self.encode_kwargs = None

        def encode(self, texts, **kwargs):
            self.encode_kwargs = kwargs
            return ArrayLike()

    e = LocalBertEmbedder()
    fake_model = FakeModel()
    with patch.object(e, "_load_model", return_value=fake_model):
        assert e.embed(["hello"]) == [[1.0, 2.0, 3.0]]

    assert fake_model.encode_kwargs is not None
    assert fake_model.encode_kwargs["show_progress_bar"] is False


def test_singleton_returns_same_instance():
    """get_default_embedder is process-wide — same args → same instance."""
    from hexus.embedder import get_default_embedder, reset_default_embedder
    reset_default_embedder()
    e1 = get_default_embedder()
    e2 = get_default_embedder()
    assert e1 is e2
    reset_default_embedder()


def test_singleton_caches_by_model_name():
    """Different model_name → different singleton. (Mostly relevant for
    tests; production uses one model.)"""
    from hexus.embedder import get_default_embedder, reset_default_embedder
    reset_default_embedder()
    e_a = get_default_embedder(model_name="model-a")
    e_b = get_default_embedder(model_name="model-b")
    assert e_a is not e_b
    # Same name → same instance.
    e_a2 = get_default_embedder(model_name="model-a")
    assert e_a is e_a2
    reset_default_embedder()


def test_singleton_is_thread_safe():
    """Concurrent get_default_embedder() calls converge on a single instance."""
    from hexus.embedder import get_default_embedder, reset_default_embedder
    reset_default_embedder()
    instances: List = []
    barrier = threading.Barrier(8)

    def grab():
        barrier.wait()
        instances.append(get_default_embedder())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(i is instances[0] for i in instances), \
        "all threads must see the same singleton instance"
    reset_default_embedder()


def test_reset_drops_singleton():
    """reset_default_embedder() forces the next get_default_embedder()
    call to construct a fresh instance. Test-only helper, but the
    contract is documented for downstream callers."""
    from hexus.embedder import get_default_embedder, reset_default_embedder
    e1 = get_default_embedder()
    reset_default_embedder()
    e2 = get_default_embedder()
    assert e1 is not e2
    reset_default_embedder()


# ---------------------------------------------------------------------------
# Real-model tests — skipped by default if the dep is unavailable or
# the operator wants a fast CI run.
# ---------------------------------------------------------------------------

# Fixture: one shared embedder for all real-model tests in this module.
# Module-scoped so the model load (~1-2s) happens once.
@pytest.fixture(scope="module")
def embedder():
    if os.environ.get("SENTENCE_TRANSFORMERS_SKIP_REAL") == "1":
        pytest.skip("SENTENCE_TRANSFORMERS_SKIP_REAL=1")
    from hexus.embedder import LocalBertEmbedder
    e = LocalBertEmbedder()
    e.ensure_loaded()
    yield e


def test_ensure_loaded_works(embedder):
    """ensure_loaded() sets is_loaded=True and the dim property is the
    actual model dim."""
    assert embedder.is_loaded is True
    assert embedder.dim == 384


def test_embed_single_text(embedder):
    """Embedding one short text returns a 384-dim float vector."""
    vecs = embedder.embed(["hello world"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384
    assert all(isinstance(x, float) for x in vecs[0])
    # Values should be non-trivially populated (not all zero).
    assert any(abs(x) > 1e-6 for x in vecs[0])


def test_embed_batch(embedder):
    """Embedding a batch returns one vector per input, in order."""
    texts = [
        "the quick brown fox",
        "jumps over the lazy dog",
        "completely unrelated sentence about gardening",
    ]
    vecs = embedder.embed(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == 384


def test_semantic_similarity(embedder):
    """Related sentences have higher cosine similarity than unrelated ones.

    This is the whole point of the BERT swap — the embeddings should
    encode semantic meaning well enough that a near-duplicate scores
    higher than a random one. We don't assert hard thresholds (model
    quality can drift); we assert the relative ordering.
    """
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    vecs = embedder.embed([
        "How do I configure Postgres connection pooling?",
        "What's the right way to size a psycopg connection pool?",
        "The recipe calls for two cups of flour and one egg.",
    ])
    sim_related = cos(vecs[0], vecs[1])
    sim_unrelated = cos(vecs[0], vecs[2])
    assert sim_related > sim_unrelated, (
        f"related similarity {sim_related:.3f} should beat "
        f"unrelated {sim_unrelated:.3f}"
    )


def test_embed_filters_empty_in_batch(embedder):
    """A batch with mixed empty/non-empty inputs: the empty ones are
    silently dropped, only non-empty vectors are returned. The current
    caller surface doesn't care about per-input correlation; this just
    pins the contract so a regression is caught."""
    vecs = embedder.embed(["hello", "", "  ", "world"])
    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)


# ---------------------------------------------------------------------------
# embed.py dispatch tests (no model load for the local path — uses a stub)
# ---------------------------------------------------------------------------

def test_embed_no_base_url_uses_local(monkeypatch):
    """embed() with no base_url dispatches to the local embedder."""
    from hexus import embed as embed_fn

    class FakeEmbedder:
        def __init__(self):
            self.called_with = None
        def embed(self, texts):
            self.called_with = texts
            return [[0.1] * 384]

    fake = FakeEmbedder()
    monkeypatch.setattr("hexus.embedder.get_default_embedder", lambda **kw: fake)

    vec = embed_fn("hello", base_url=None, model="some/model")
    assert vec == [0.1] * 384
    assert fake.called_with == ["hello"]


def test_embed_no_base_url_default_model(monkeypatch):
    """embed() with no base_url and no model uses DEFAULT_MODEL."""
    from hexus import embed as embed_fn

    captured = {}
    class FakeEmbedder:
        def embed(self, texts):
            captured["texts"] = texts
            return [[0.0] * 384]

    def fake_getter(model_name=None, **kw):
        captured["model_name"] = model_name
        return FakeEmbedder()

    monkeypatch.setattr("hexus.embedder.get_default_embedder", fake_getter)
    embed_fn("hello")
    from hexus.embedder import DEFAULT_MODEL
    assert captured["model_name"] == DEFAULT_MODEL


def test_embed_base_url_dispatches_to_http(monkeypatch):
    """embed() with a base_url goes through the HTTP path, not local."""
    from hexus import embed as embed_fn

    # Spy: if the local embedder is called, the test fails.
    local_called = {"value": False}
    class SpyLocal:
        def embed(self, texts):
            local_called["value"] = True
            return [[0.0] * 384]

    monkeypatch.setattr("hexus.embedder.get_default_embedder", lambda **kw: SpyLocal())

    # Patch urllib.request.urlopen to return a fake 384-dim embedding.
    class FakeResp:
        def __init__(self, body):
            self.body = body.encode("utf-8")
        def read(self):
            return self.body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fake_response_body = '{"data": [{"embedding": ' + str([0.2] * 384).replace("'", '"') + '}]}'

    with patch("urllib.request.urlopen") as urlopen:
        urlopen.return_value = FakeResp(fake_response_body)
        vec = embed_fn("hello", base_url="http://fake:11434")

    assert vec == [0.2] * 384
    assert local_called["value"] is False, "local embedder should not be called when base_url is set"


def test_embed_truncates_long_text():
    """Text longer than MAX_INPUT_CHARS is silently truncated (the
    embedder would truncate anyway, but doing it here keeps log lines
    sane and avoids the model choking on a 100KB string)."""
    from hexus import embed as embed_fn
    from hexus.embed import MAX_INPUT_CHARS

    captured = {}
    class FakeEmbedder:
        def embed(self, texts):
            captured["lengths"] = [len(t) for t in texts]
            return [[0.0] * 384]

    with patch("hexus.embedder.get_default_embedder", lambda **kw: FakeEmbedder()):
        huge = "x" * (MAX_INPUT_CHARS + 500)
        embed_fn(huge)

    assert captured["lengths"] == [MAX_INPUT_CHARS]


def test_embed_http_404_raises_embedding_error():
    """The HTTP path raises EmbeddingError on a non-2xx response, not
    a urllib.error.HTTPError leaking out."""
    from hexus import embed as embed_fn
    from hexus.embed import EmbeddingError
    import urllib.error

    with patch("urllib.request.urlopen") as urlopen:
        urlopen.side_effect = urllib.error.HTTPError(
            url="http://fake/v1/embeddings",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with pytest.raises(EmbeddingError):
            embed_fn("hello", base_url="http://fake:11434")

"""Malformed endpoint responses and bounded writer lifecycle contracts."""

from __future__ import annotations

import http.client
import io
import json
import queue
import threading
import urllib.error
from importlib import import_module
from unittest.mock import Mock

import pytest

from hermes_pgvector.writer import AsyncWriter

embed_mod = import_module("hermes_pgvector.embed")


def _response(monkeypatch, payload, *, raw=False):
    contents = payload if raw else json.dumps(payload).encode("utf-8")
    request = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(contents))
    monkeypatch.setattr(embed_mod.urllib.request, "urlopen", request)
    return request


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_blank_text_is_rejected_before_network(monkeypatch, text):
    request = _response(monkeypatch, {})
    with pytest.raises(embed_mod.EmbeddingError, match="empty input"):
        embed_mod.embed(text, base_url="http://embed")
    request.assert_not_called()


@pytest.mark.parametrize("protocol", [None, "", "  "])
def test_unset_protocol_uses_auto(protocol):
    assert embed_mod.normalize_protocol(protocol) == "auto"


def test_large_input_is_truncated_and_base_url_normalized(monkeypatch):
    request = _response(monkeypatch, {"data": [{"embedding": [0.5, 0.5]}]})
    assert embed_mod.embed("x" * 9000, base_url="http://embed///", dim=2) == [0.5, 0.5]
    req = request.call_args.args[0]
    assert req.full_url == "http://embed/v1/embeddings"
    assert json.loads(req.data)["input"] == "x" * 6000
    assert req.get_header("Content-type") == "application/json"


@pytest.mark.parametrize("payload,expected", [
    (b"not json", "invalid JSON response"),
    (b'{"data": []}', "unexpected response shape"),
    (b'{"data": null}', "unexpected response shape"),
    (b'{}', "unexpected response shape"),
    (b'{"data": [{"embedding": null}]}', "response had no embedding array"),
    (b'{"data": [{"embedding": "bad"}]}', "response had no embedding array"),
    (b'{"data": [{"embedding": []}]}', "response had no embedding array"),
])
def test_bad_endpoint_payloads_are_embedding_errors(monkeypatch, payload, expected):
    _response(monkeypatch, payload, raw=True)
    with pytest.raises(embed_mod.EmbeddingError, match=expected):
        embed_mod.embed("hello", base_url="http://embed", dim=2, protocol="openai")


@pytest.mark.parametrize("payload", [None, [], "not-a-mapping"])
def test_native_protocol_rejects_non_mapping_payload(monkeypatch, payload):
    _response(monkeypatch, payload)
    with pytest.raises(embed_mod.EmbeddingError, match="unexpected response shape"):
        embed_mod.embed("hello", base_url="http://embed", protocol="ollama")


def test_native_protocol_accepts_legacy_single_embedding_shape(monkeypatch):
    _response(monkeypatch, {"embedding": [0.3, 0.4]})
    assert embed_mod.embed("hello", base_url="http://embed", dim=2, protocol="ollama") == [0.3, 0.4]


@pytest.mark.parametrize("value", [None, "0.3", True, float("nan"), float("inf"), 3.5e38, 10**400])
def test_invalid_vector_values_never_reach_storage(monkeypatch, value):
    _response(monkeypatch, {"data": [{"embedding": [value]}]})
    with pytest.raises(embed_mod.EmbeddingError, match="non-finite or non-numeric"):
        embed_mod.embed("hello", base_url="http://embed", dim=1, protocol="openai")


@pytest.mark.parametrize("error", [
    urllib.error.URLError("host unavailable"),
    http.client.IncompleteRead(b"partial", 20),
    ConnectionResetError("server closed"),
])
def test_transport_failures_are_wrapped_for_fail_soft_callers(monkeypatch, error):
    monkeypatch.setattr(embed_mod.urllib.request, "urlopen", Mock(side_effect=error))
    with pytest.raises(embed_mod.EmbeddingError, match="connection failed") as exc:
        embed_mod.embed("hello", base_url="http://embed", protocol="openai")
    assert exc.value.__cause__ is error


def test_retries_use_exponential_backoff_and_return_recovered_vector(monkeypatch):
    attempts = Mock(side_effect=[embed_mod.EmbeddingError("first"), embed_mod.EmbeddingError("second"), [0.2]])
    sleeps = Mock()
    monkeypatch.setattr(embed_mod, "_embed_once", attempts)
    monkeypatch.setattr(embed_mod.time, "sleep", sleeps)
    assert embed_mod.embed("hello", base_url="http://embed", retries=2, backoff=0.25) == [0.2]
    assert [call.args[0] for call in sleeps.call_args_list] == [0.25, 0.5]
    assert attempts.call_count == 3


def test_pgvector_literal_uses_bounded_precision_and_valid_array_delimiters():
    assert embed_mod.to_pgvector_literal([1.23456789, -0.25, 0, 1e-7]) == "[1.23457,-0.25,0,1e-07]"
    assert embed_mod.to_pgvector_literal([]) == "[]"


def _enqueue(writer, content):
    return writer.enqueue(action="add", agent_identity="marketing", target="memory", content=content)


def test_full_queue_drops_without_blocking_and_warns_once(caplog):
    entered = threading.Event()
    release = threading.Event()
    received = []

    def sink(item):
        entered.set()
        assert release.wait(5)
        received.append(item.content)

    writer = AsyncWriter(sink, maxsize=1)
    try:
        assert _enqueue(writer, "in flight")
        assert entered.wait(5)
        assert _enqueue(writer, "queued")
        assert not _enqueue(writer, "dropped one")
        assert not _enqueue(writer, "dropped two")
        assert writer.stats() == {"queue_size": 1, "queue_max": 1, "dropped_total": 2, "thread_alive": True}
        assert caplog.text.count("queue full") == 1
        # With a blocked worker and a full queue, shutdown cannot insert its
        # wake-up sentinel. It must return on the timeout and report loss risk.
        writer.shutdown(timeout=0)
        assert writer.draining
        assert "did not drain" in caplog.text
        assert "abandoned 1 queued write(s)" in caplog.text
    finally:
        release.set()
        writer.shutdown(timeout=5)
    assert received == ["in flight", "queued"]
    assert writer.stats()["thread_alive"] is False
    assert writer._queue.unfinished_tasks == 0


def test_worker_exception_is_logged_and_next_write_still_processed(caplog):
    received = []

    def sink(item):
        if item.content == "bad":
            raise RuntimeError("store unavailable")
        received.append(item)

    writer = AsyncWriter(sink)
    try:
        assert _enqueue(writer, "bad")
        assert writer.enqueue(action="replace", agent_identity="sales", target="user", content="good",
                              extra={"old_text": "previous"}, metadata={"session_id": "session-1"})
    finally:
        writer.shutdown(timeout=5)
    assert len(received) == 1
    assert received[0].extra == {"old_text": "previous"}
    assert received[0].metadata == {"session_id": "session-1"}
    assert "RuntimeError on (add/marketing/memory): store unavailable" in caplog.text
    assert writer._queue.unfinished_tasks == 0


def test_idle_writer_survives_queue_timeout_then_accepts_work(monkeypatch):
    idle = threading.Event()
    received = threading.Event()
    writer = AsyncWriter(lambda item: received.set())
    real_get = writer._queue.get

    def get(*, timeout):
        try:
            return real_get(timeout=0.01)
        except queue.Empty:
            idle.set()
            raise

    monkeypatch.setattr(writer._queue, "get", get)
    try:
        writer._ensure_thread()
        assert idle.wait(5)
        assert _enqueue(writer, "after idle timeout")
        assert received.wait(5)
    finally:
        writer.shutdown(timeout=5)
    assert writer._queue.unfinished_tasks == 0


def test_nondraining_sentinel_stops_worker_and_next_enqueue_restarts_it():
    received = []
    writer = AsyncWriter(lambda item: received.append(item.content))
    try:
        writer._queue.put_nowait(None)
        writer._ensure_thread()
        original_thread = writer._thread
        original_thread.join(timeout=5)
        assert not original_thread.is_alive()
        assert not writer.draining
        assert _enqueue(writer, "after restart")
        assert writer._thread is not original_thread
    finally:
        writer.shutdown(timeout=5)
    assert received == ["after restart"]
    assert writer._queue.unfinished_tasks == 0


def test_shutdown_finished_writer_is_idempotent_and_new_write_clears_draining():
    received = []
    writer = AsyncWriter(lambda item: received.append(item.content))
    try:
        assert _enqueue(writer, "before shutdown")
        writer.shutdown(timeout=5)
        assert writer.draining
        writer.shutdown(timeout=0)
        assert _enqueue(writer, "after shutdown")
        assert not writer.draining
    finally:
        writer.shutdown(timeout=5)
    assert received == ["before shutdown", "after shutdown"]

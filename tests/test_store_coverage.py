"""Storage boundary failures: honest degradation without requiring a server.

SQL behavior belongs in the live companion module; these tests inject only
connection/driver failures that a healthy disposable server cannot reproduce.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from hermes_pgvector import store as storage
from hermes_pgvector.store import MemoryStore


def _driver_store(*rows, error=None):
    """A DB-API boundary with sequential results or a connection failure."""
    store = MemoryStore("unused-test-dsn")
    pool = MagicMock()
    conn = pool.connection.return_value.__enter__.return_value
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.side_effect = rows
    if error is not None:
        pool.connection.side_effect = error
    store._pool = pool
    return store, pool


@pytest.mark.parametrize("encoding, expected", [
    (b"UTF8", storage._HAS_TEXT_SQL_UTF8),
    (bytearray(b"UTF8"), storage._HAS_TEXT_SQL_UTF8),
    ("LATIN1", storage._HAS_TEXT_SQL_ASCII),
])
def test_encoding_predicate_decodes_driver_bytes_and_caches(encoding, expected):
    store, pool = _driver_store((encoding,))
    assert store._has_text_sql() == expected
    pool.connection.side_effect = RuntimeError("server became unavailable")
    assert store._has_text_sql() == expected


def test_encoding_probe_failure_is_not_cached_and_retries():
    store, pool = _driver_store(("UTF8",), error=OSError("offline"))
    assert store._has_text_sql() == storage._HAS_TEXT_SQL_ASCII
    assert store._has_text_sql_cached is None
    pool.connection.side_effect = None
    assert store._has_text_sql() == storage._HAS_TEXT_SQL_UTF8


def test_pool_close_failure_discards_broken_pool_and_is_idempotent(caplog):
    store, pool = _driver_store()
    pool.close.side_effect = OSError("close interrupted")
    with caplog.at_level("DEBUG", logger="hermes_pgvector.store"):
        store.close()
        store.close()
    assert store._pool is None
    assert "close interrupted" in caplog.text


def test_pool_creation_is_serialized_when_callers_arrive_together(monkeypatch):
    store = MemoryStore("unused-test-dsn")
    constructing = threading.Event()
    second_arrived = threading.Event()
    underlying_lock = threading.Lock()
    attempts = 0
    constructed = []
    results = []
    failures = []

    def construct(**kwargs):
        constructing.set()
        if not second_arrived.wait(3):
            raise RuntimeError("second caller never reached lock")
        pool = object()
        constructed.append(pool)
        return pool

    def get():
        try:
            results.append(store._get_pool())
        except RuntimeError as exc:
            failures.append(exc)

    class ObservedLock:
        def __enter__(self):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                second_arrived.set()
            underlying_lock.acquire()

        def __exit__(self, *args):
            underlying_lock.release()

    store._lock = ObservedLock()
    monkeypatch.setattr(storage, "ConnectionPool", construct)
    first = threading.Thread(target=get)
    second = threading.Thread(target=get)
    first.start()
    assert constructing.wait(3)
    second.start()
    first.join(4)
    second.join(4)
    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert len(constructed) == 1
    assert results == [constructed[0], constructed[0]]
    assert store._get_pool() is constructed[0]


@pytest.mark.parametrize("rows", [((None, "edges"),), (("agents", None),), (("agents", "edges"), None)])
def test_attribution_probe_requires_both_tables_and_parent_column(rows):
    store, _ = _driver_store(*rows)
    assert store.ensure_migration_002_applied() is False


def test_attribution_probe_failure_degrades_to_disabled(caplog):
    store, _ = _driver_store(error=OSError("database unavailable"))
    with caplog.at_level("DEBUG", logger="hermes_pgvector.store"):
        assert store.ensure_migration_002_applied() is False
    assert "database unavailable" in caplog.text


@pytest.mark.parametrize("row", [None, (None,), (-1,), (0,), (14.75,)])
def test_estimate_unknown_or_unanalyzed_is_not_reported_as_zero(row):
    store, _ = _driver_store(row)
    assert store.estimate_count("memory_entries") == (14 if row == (14.75,) else None)


def test_estimate_database_failure_returns_unknown(caplog):
    store, _ = _driver_store(error=OSError("estimate unavailable"))
    with caplog.at_level("DEBUG", logger="hermes_pgvector.store"):
        assert store.estimate_count("conversations") is None
    assert "estimate unavailable" in caplog.text


def test_health_missing_table_reports_actionable_status():
    store, _ = _driver_store((False,))
    assert store.health() == {
        "ok": False, "error": "memory_entries table missing", "row_count_estimate": None,
    }


def test_health_connection_error_is_bounded():
    message = "unavailable " * 30
    store, _ = _driver_store(error=OSError(message))
    assert store.health() == {"ok": False, "error": message[:200], "row_count_estimate": None}


def test_maintenance_audit_failure_is_best_effort(caplog):
    store, _ = _driver_store(error=OSError("audit unavailable"))
    with caplog.at_level("DEBUG", logger="hermes_pgvector.store"):
        assert store.log_maintenance(operation="prune") is None
    assert "audit unavailable" in caplog.text


@pytest.mark.parametrize("result", [None, (), "not a vector", [0.1]])
def test_backfill_refuses_wrong_embedding_shape_before_any_sql(result):
    store = MemoryStore("unused-test-dsn")
    with pytest.raises(ValueError, match="refusing to backfill"):
        store.backfill_null_embeddings(embed_fn=lambda _: result)
    assert store._pool is None


def test_backfill_endpoint_outage_reports_unknown_remaining_for_every_table(caplog):
    store = MemoryStore("unused-test-dsn")

    def unavailable(_):
        raise TimeoutError("endpoint stalled")

    report = store.backfill_null_embeddings(embed_fn=unavailable)
    assert set(report) == {"memory_entries", "conversations"}
    for outcome in report.values():
        assert outcome == {
            "processed": 0, "succeeded": 0, "failed": 0, "remaining": None,
            "unembeddable": None, "skipped_changed": None, "note": "embed-unavailable",
        }
    assert store._pool is None
    assert "endpoint stalled" in caplog.text


@pytest.mark.parametrize("query", [None, "", " \n\t"])
def test_blank_hybrid_turn_query_returns_before_opening_pool(query):
    store = MemoryStore("unused-test-dsn")
    assert store.hybrid_search_turns(query_text=query) == []
    assert store._pool is None

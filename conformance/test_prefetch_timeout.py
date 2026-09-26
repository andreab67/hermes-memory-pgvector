"""WP-F conformance test 5 (docs/release/PLAN-1.0.md Sec 5 WP-F):

MemoryManager prefetch with the fake embed server at --delay 20 returns
without the external-prefetch timeout warning.

agent.memory_manager.MemoryManager._prefetch_provider runs an external
provider's prefetch() on a background thread and joins it with
`_EXTERNAL_PREFETCH_TIMEOUT_S` (8.0s at the pinned ref). If the thread is
still alive after that, it logs (at WARNING, on the "agent.memory_manager"
logger) that the provider "timed out ... skipping it until the stuck call
returns" and returns "" -- the thread itself keeps running, detached.

This is only avoidable if the provider enforces its OWN, shorter, overall
deadline on the embed call it makes inside prefetch() -- PLAN-1.0.md Sec 3
calls this `prefetch_budget`, default 5.0s (comfortably under upstream's
8.0s). WP-D2 is adding that to hermes_pgvector/__init__.py in PARALLEL with
this work package and may not have landed in this checkout yet: today,
prefetch() only has `embed_timeout` (default 10.0s, i.e. ABOVE upstream's
8.0s budget), so this test is EXPECTED TO FAIL until WP-D2 lands. Written
correctly regardless, per the WP-F brief: run it and report the failure
mode, don't weaken it to pass.
"""

from __future__ import annotations

import logging
import os
import time

import pytest

agent_memory_provider = pytest.importorskip("agent.memory_provider")
agent_memory_manager = pytest.importorskip("agent.memory_manager")

import hermes_pgvector  # noqa: E402

MemoryManager = agent_memory_manager.MemoryManager


@pytest.fixture
def dsn():
    d = os.environ.get("PG_TEST_DSN")
    if not d:
        pytest.skip("PG_TEST_DSN not set")
    return d


def test_prefetch_all_does_not_trip_the_external_prefetch_timeout(
    dsn, tmp_path, fake_embed_server, caplog
):
    embed_url = fake_embed_server(delay=20)
    identity = "pytest-wpf-prefetch-" + os.urandom(4).hex()
    provider = hermes_pgvector.PgvectorMemoryProvider(
        config={"dsn": dsn, "embed_url": embed_url, "bulk_sync_on_init": False}
    )
    manager = MemoryManager()
    manager.add_provider(provider)
    provider.initialize(
        session_id="conformance-prefetch", hermes_home=str(tmp_path), agent_identity=identity,
    )
    assert provider._healthy, "provider failed to initialize against PG_TEST_DSN"

    try:
        with caplog.at_level(logging.WARNING, logger="agent.memory_manager"):
            start = time.monotonic()
            result = manager.prefetch_all(
                "what do we know about the launch plan?", session_id="conformance-prefetch",
            )
            elapsed = time.monotonic() - start

        timeout_warnings = [
            r.getMessage() for r in caplog.records
            if r.name == "agent.memory_manager" and "timed out" in r.getMessage()
        ]
        assert not timeout_warnings, (
            "MemoryManager logged an external-prefetch timeout warning: "
            f"{timeout_warnings!r} -- the provider's prefetch() did not return "
            "within upstream's external_prefetch_timeout budget "
            f"({manager._external_prefetch_timeout}s). This is EXPECTED until "
            "prefetch_budget (WP-D2) lands and is set below that timeout."
        )
        assert result == "", f"expected no recall context from a timed-out embed, got: {result!r}"
        assert elapsed < manager._external_prefetch_timeout, (
            f"prefetch_all() took {elapsed:.1f}s, at or beyond upstream's "
            f"{manager._external_prefetch_timeout}s external-prefetch timeout -- "
            "the provider is not enforcing its own shorter deadline"
        )
    finally:
        provider.shutdown()
        import psycopg

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (identity,))
                conn.commit()

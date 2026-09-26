"""DB-gated tests for the v0.6.0 WP-A store-correctness fixes.

Skip without PG_TEST_DSN. Point it at a THROWAWAY database (see
tests/test_store_v04.py for why: several methods under test sweep whole
plugin-owned tables, not a single agent_identity).

Covers, one behaviour per finding:
  H3  -- backfill_null_embeddings: keyset pagination survives permanently
         failing rows; a consecutive-failure breaker aborts a flapping run.
  M2  -- backfill's persisting UPDATE is guarded against a concurrent
         replace()/remove(); a changed row is skipped_changed, not failed.
  M1  -- add() ON CONFLICT DO NOTHING (no target) still rejects an exact
         duplicate; bulk_upsert_md() continues past one entry that fails to
         insert and reports it in `failed`.
  H2  -- replace()/remove() exact_content: exact match, never LIKE; a blank
         exact_content never touches a row.
  M6  -- health() reports row_count_estimate (not row_count); estimate_count()
         is whitelist-guarded and never raises.
  L6  -- search()/search_turns() exclude NULL-embedding rows.
  L8  -- scan_pii()'s default pattern no longer matches inside a longer
         digit run, and honours `tables`.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector.store import MemoryStore  # noqa: E402


@pytest.fixture
def store():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    s = MemoryStore(dsn)
    s.ensure_schema()
    agent = "pytest-v060-" + os.urandom(4).hex()
    yield s, agent
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity LIKE %s", (agent + "%",))
            cur.execute("DELETE FROM conversations WHERE agent_identity LIKE %s", (agent + "%",))
            conn.commit()


# --- H3: keyset pagination + consecutive-failure breaker -------------------

def test_backfill_keyset_pagination_gets_past_failing_rows(store):
    """A block of permanently-failing rows at the low end of the id range
    must not stall every later row -- each row is visited exactly once."""
    s, agent = store
    for i in range(5):
        s.append_turn(session_id="s", agent_identity=agent, role="user",
                       content=f"POISON {agent} {i}")
    for i in range(5):
        s.append_turn(session_id="s", agent_identity=agent, role="user",
                       content=f"GOOD {agent} {i}")

    def embed_fn(text):
        if "POISON" in text:
            raise RuntimeError("endpoint rejects this input")
        return [0.1] * 768

    result = s.backfill_null_embeddings(
        embed_fn=embed_fn, tables=("conversations",), batch_size=3,
    )["conversations"]

    assert result["failed"] == 5, "each failing row counted exactly once"
    assert result["succeeded"] == 5, "good rows must not be stuck behind failures"
    assert result["processed"] == 10


def test_backfill_consecutive_failure_breaker_trips_and_notes(store):
    s, agent = store
    for i in range(6):
        s.append_turn(session_id="s", agent_identity=agent, role="user",
                       content=f"POISON {agent} {i}")

    def embed_fn(text):
        # The up-front dimension probe (text == "dimension probe") must
        # succeed, or backfill treats it as an unreachable endpoint and
        # aborts with note="embed-unavailable" before ever reaching a row --
        # a different code path than the one this test targets.
        if text == "dimension probe":
            return [0.1] * 768
        raise RuntimeError("endpoint down")

    result = s.backfill_null_embeddings(
        embed_fn=embed_fn, tables=("conversations",), batch_size=2,
        max_consecutive_failures=3,
    )["conversations"]

    assert result.get("note") == "aborted-consecutive-failures"
    assert result["failed"] == 3, "stops right after tripping the breaker"
    assert result["processed"] == 3


def test_backfill_success_resets_the_consecutive_counter(store):
    """Failures interleaved with successes must never trip the breaker."""
    s, agent = store
    for i in range(10):
        s.append_turn(session_id="s", agent_identity=agent, role="user",
                       content=f"row {agent} {i}")

    calls = {"n": 0}

    def embed_fn(text):
        # Keep the up-front dimension probe out of the alternating-failure
        # count -- it must succeed so backfill proceeds to the row loop
        # rather than aborting as embed-unavailable.
        if text == "dimension probe":
            return [0.1] * 768
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            raise RuntimeError("flaky")
        return [0.1] * 768

    result = s.backfill_null_embeddings(
        embed_fn=embed_fn, tables=("conversations",), batch_size=10,
        max_consecutive_failures=2,
    )["conversations"]

    assert "note" not in result, "alternating failures never reach 2 in a row"
    assert result["failed"] == 5
    assert result["succeeded"] == 5


# --- M2: guarded UPDATE ------------------------------------------------------

def test_backfill_skips_a_row_changed_mid_run(store):
    """A replace() landing between backfill's SELECT and UPDATE must not
    stamp the NEW content with the OLD text's embedding."""
    s, agent = store
    rid = s.add(agent_identity=agent, target="memory", content="original text")
    assert rid is not None

    def embed_fn(text):
        # The up-front dimension probe (text == "dimension probe") must not
        # trigger the simulated race -- it runs before the real row is even
        # SELECTed, so mutating here would move the "concurrent" edit earlier
        # than the scenario this test is modeling.
        if text == "dimension probe":
            return [0.1] * 768
        # Simulate a race: mutate the row's content after it was read for
        # embedding, before backfill's guarded UPDATE runs.
        import psycopg
        with psycopg.connect(s._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memory_entries SET content = %s WHERE id = %s",
                    ("content changed by a concurrent replace()", rid),
                )
                conn.commit()
        return [0.1] * 768

    result = s.backfill_null_embeddings(
        embed_fn=embed_fn, tables=("memory_entries",), batch_size=10,
    )["memory_entries"]

    assert result["skipped_changed"] == 1
    assert result["succeeded"] == 0
    assert result["failed"] == 0

    rows = s.list_entries(agent_identity=agent, target="memory", limit=10)
    assert rows[0]["content"] == "content changed by a concurrent replace()"
    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding IS NULL FROM memory_entries WHERE id = %s", (rid,))
            assert cur.fetchone()[0] is True, "the new content must not get the old text's vector"


# --- M1 (code): ON CONFLICT DO NOTHING with no target; bulk continues -------

def test_add_duplicate_still_returns_none_with_no_conflict_target(store):
    s, agent = store
    first = s.add(agent_identity=agent, target="memory", content="same content twice")
    second = s.add(agent_identity=agent, target="memory", content="same content twice")
    assert isinstance(first, int)
    assert second is None
    assert s.count(agent_identity=agent) == 1


def test_bulk_upsert_md_continues_past_a_failing_entry(store, tmp_path, monkeypatch):
    """One bad entry must not abort the rest of the file (M1); `failed` is
    counted and a single warning is logged for the whole file."""
    s, agent = store
    md = tmp_path / "MEMORY.md"
    md.write_text(
        "first note" + s.ENTRY_DELIMITER +
        "POISON entry that fails to insert" + s.ENTRY_DELIMITER +
        "third note",
        encoding="utf-8",
    )

    real_add = s.add

    def _flaky_add(*, agent_identity, target, content, embedding=None, metadata=None):
        if content == "POISON entry that fails to insert":
            raise RuntimeError("simulated insert failure")
        return real_add(
            agent_identity=agent_identity, target=target, content=content,
            embedding=embedding, metadata=metadata,
        )

    monkeypatch.setattr(s, "add", _flaky_add)

    result = s.bulk_upsert_md(agent_identity=agent, target="memory", file_path=md, embed_fn=None)

    assert result == {"parsed": 3, "inserted": 2, "skipped": 0, "failed": 1}
    assert s.count(agent_identity=agent, target="memory") == 2


# --- H2 (store): exact_content on replace()/remove() -----------------------

def test_replace_exact_content_matches_precisely_never_like(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="15% YoY growth noted")
    s.add(agent_identity=agent, target="memory", content="unrelated 15 YoY growth noted too")

    n = s.replace(
        agent_identity=agent, target="memory",
        exact_content="15% YoY growth noted",
        new_content="15pct YoY growth noted",
    )
    assert n == 1
    contents = {r["content"] for r in s.list_entries(agent_identity=agent, target="memory", limit=10)}
    assert "15pct YoY growth noted" in contents
    assert "unrelated 15 YoY growth noted too" in contents


def test_replace_blank_exact_content_matches_nothing(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="keep me untouched")
    n = s.replace(
        agent_identity=agent, target="memory",
        exact_content="   ",
        new_content="should never appear",
    )
    assert n == 0
    contents = {r["content"] for r in s.list_entries(agent_identity=agent, target="memory", limit=10)}
    assert contents == {"keep me untouched"}


def test_remove_exact_content_matches_precisely_never_like(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="delete this exact row")
    s.add(agent_identity=agent, target="memory", content="delete this exact row and more")

    n = s.remove(agent_identity=agent, target="memory", exact_content="delete this exact row")
    assert n == 1
    contents = {r["content"] for r in s.list_entries(agent_identity=agent, target="memory", limit=10)}
    assert contents == {"delete this exact row and more"}


def test_remove_blank_exact_content_deletes_nothing_and_needs_no_old_text(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="row one")
    s.add(agent_identity=agent, target="memory", content="row two")

    n = s.remove(agent_identity=agent, target="memory", exact_content="")
    assert n == 0
    assert s.count(agent_identity=agent, target="memory") == 2


def test_remove_without_exact_content_still_requires_nonempty_old_text(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="row one")
    with pytest.raises(ValueError):
        s.remove(agent_identity=agent, target="memory", old_text="")
    assert s.count(agent_identity=agent, target="memory") == 1


# --- M6 (store): row_count_estimate + estimate_count ------------------------

def test_health_reports_row_count_estimate_not_row_count(store):
    s, _ = store
    h = s.health()
    assert h["ok"] is True
    assert "row_count" not in h
    assert h["row_count_estimate"] is None or isinstance(h["row_count_estimate"], int)


def test_estimate_count_whitelisted_table_never_raises(store):
    s, _ = store
    est = s.estimate_count("memory_entries")
    assert est is None or (isinstance(est, int) and est >= 0)


def test_estimate_count_rejects_non_whitelisted_table(store):
    s, _ = store
    assert s.estimate_count("events") is None
    assert s.estimate_count("not_a_real_table") is None


# --- L6: pure-vector search excludes NULL embeddings ------------------------

def test_search_excludes_null_embedding_rows(store):
    s, agent = store
    vec = [0.2] * 768
    s.add(agent_identity=agent, target="memory", content="has a real embedding", embedding=vec)
    s.add(agent_identity=agent, target="memory", content="text only, no embedding yet")

    rows = s.search(query_embedding=vec, agent_identity=agent, limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == "has a real embedding"


def test_search_turns_excludes_null_embedding_rows(store):
    s, agent = store
    vec = [0.2] * 768
    s.append_turn(session_id="s", agent_identity=agent, role="user",
                  content="has a real embedding", embedding=vec)
    s.append_turn(session_id="s", agent_identity=agent, role="user",
                  content="text only, no embedding yet")

    rows = s.search_turns(query_embedding=vec, agent_identity=agent, limit=10)
    assert len(rows) == 1
    assert rows[0]["content"] == "has a real embedding"


# --- L8 (store): bounded PII pattern + tables ------------------------------

def test_scan_pii_default_pattern_does_not_match_inside_a_longer_digit_run(store):
    s, agent = store
    s.append_turn(session_id="s", agent_identity=agent, role="user",
                  content="call 5550101234 about the order")
    s.append_turn(session_id="s", agent_identity=agent, role="user",
                  content="tracking id 123456789012345 has nothing to do with a phone")

    def _scoped_match_count(pattern):
        with s._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM conversations "
                    "WHERE agent_identity = %s AND content ~ %s",
                    (agent, pattern),
                )
                return cur.fetchone()[0]

    assert _scoped_match_count(r"\d{10,11}") == 2, "old unbounded pattern matches both rows"
    # Read the actual default off the store, not re-typed here, so a future
    # change to the constant can't silently desync the test.
    default_pattern = inspect.signature(s.scan_pii).parameters["pattern"].default
    assert _scoped_match_count(default_pattern) == 1, (
        "the bounded default must not match a 10-11 digit window inside "
        "the 15-digit tracking id"
    )

    counts = s.scan_pii(tables=("conversations",))
    assert counts["conversations"] >= 1


def test_scan_pii_honours_tables_whitelist(store):
    s, _ = store
    with pytest.raises(ValueError):
        s.scan_pii(tables=("relations",))

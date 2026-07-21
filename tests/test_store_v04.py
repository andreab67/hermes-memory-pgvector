"""DB-gated tests for v0.4.0 store methods. Skip without PG_TEST_DSN.

IMPORTANT: point PG_TEST_DSN at a THROWAWAY database, never production. Several
methods under test (backfill_null_embeddings, prune_conversations) operate across
the WHOLE table, not a single agent_identity, so they are only safe in isolation.
The 002-dependent tests skip unless migration 002 has been applied to the test DB.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector.store import MemoryStore  # noqa: E402


@pytest.fixture
def store():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    s = MemoryStore(dsn)
    s.ensure_schema()
    agent = "pytest-v04-" + os.urandom(4).hex()
    yield s, agent
    # Cleanup every identity that starts with our prefix (covers -old/-new/-child).
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity LIKE %s", (agent + "%",))
            cur.execute("DELETE FROM conversations WHERE agent_identity LIKE %s", (agent + "%",))
            conn.commit()


def _needs_002(s):
    if not s.ensure_migration_002_applied():
        pytest.skip("migration 002 not applied to the test DB")


# --- backfill -------------------------------------------------------------

def test_backfill_null_embeddings_recovers_searchability(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="alpha backfill row")
    s.add(agent_identity=agent, target="memory", content="beta backfill row")

    dry = s.backfill_null_embeddings(embed_fn=lambda t: [0.1] * 768, tables=("memory_entries",), dry_run=True)
    assert dry["memory_entries"]["remaining"] >= 2

    res = s.backfill_null_embeddings(embed_fn=lambda t: [0.1] * 768, tables=("memory_entries",))
    assert res["memory_entries"]["succeeded"] >= 2

    found = s.search(query_embedding=[0.1] * 768, agent_identity=agent, limit=10)
    assert len(found) >= 2  # rows are now reachable via the vector index path


def test_backfill_dim_guard_rejects_wrong_dims(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="dim guard row")
    with pytest.raises(ValueError):
        s.backfill_null_embeddings(embed_fn=lambda t: [0.1] * 384, tables=("memory_entries",))


def test_backfill_whitelist_rejects_foreign_table(store):
    s, _ = store
    with pytest.raises(ValueError):
        s.backfill_null_embeddings(embed_fn=lambda t: [0.1] * 768, tables=("events",), dry_run=True)


# --- delete_by_identity (cleanup) -----------------------------------------

def test_delete_by_identity_dry_run_then_execute(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="to be deleted")
    dry = s.delete_by_identity(identities=[agent], tables=("memory_entries",), dry_run=True)
    assert dry["memory_entries"] == 1
    assert s.count(agent_identity=agent) == 1  # dry-run did not delete

    real = s.delete_by_identity(identities=[agent], tables=("memory_entries",), dry_run=False)
    assert real["memory_entries"] == 1
    assert s.count(agent_identity=agent) == 0


def test_delete_by_identity_whitelist_guard(store):
    s, agent = store
    with pytest.raises(ValueError):
        s.delete_by_identity(identities=[agent], tables=("relations",), dry_run=True)


def test_scan_pii_finds_phone_in_content(store):
    s, agent = store
    s.append_turn(session_id="s", agent_identity=agent, role="user", content="call me at 17192714834 tomorrow")
    counts = s.scan_pii(tables=("conversations",))
    assert counts["conversations"] >= 1


# --- remap_identity -------------------------------------------------------

def test_remap_identity_merges_and_dedupes(store):
    s, agent = store
    old, new = agent + "-old", agent + "-new"
    s.add(agent_identity=old, target="memory", content="shared note")
    s.add(agent_identity=new, target="memory", content="shared note")   # duplicate content
    s.add(agent_identity=old, target="memory", content="unique to old")

    dry = s.remap_identity(old_identity=old, new_identity=new, dry_run=True)
    assert dry["memory_entries"]["dropped_duplicates"] == 1
    assert dry["memory_entries"]["moved"] == 1

    res = s.remap_identity(old_identity=old, new_identity=new, dry_run=False)
    assert res["memory_entries"]["dropped_duplicates"] == 1
    assert s.count(agent_identity=old) == 0
    assert s.count(agent_identity=new) == 2  # shared (once) + unique-to-old


# --- LIKE-literal substring semantics (v0.4.2) -----------------------------

def test_replace_treats_percent_as_literal(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="Q3 revenue grew 15 million YoY")
    s.add(agent_identity=agent, target="memory", content="Q3 revenue grew 15% YoY")
    n = s.replace(
        agent_identity=agent, target="memory",
        old_text="15% YoY", new_content="Q3 revenue grew 15pct YoY",
    )
    assert n == 1  # only the literal-'%' row — never the 'million' row via wildcard
    contents = {r["content"] for r in s.list_entries(agent_identity=agent, target="memory", limit=10)}
    assert "Q3 revenue grew 15 million YoY" in contents


def test_remove_underscore_and_backslash_literal(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="snake_case_name noted")
    s.add(agent_identity=agent, target="memory", content="snakeXcaseXname noted")  # unescaped '_' would match this too
    s.add(agent_identity=agent, target="memory", content=r"path C:\Users\andreab noted")
    assert s.remove(agent_identity=agent, target="memory", old_text="snake_case_name") == 1
    assert s.remove(agent_identity=agent, target="memory", old_text=r"C:\Users\andreab") == 1
    assert s.count(agent_identity=agent, target="memory") == 1  # only the X-row remains


# --- prune ----------------------------------------------------------------

def test_prune_conversations_deletes_old(store):
    s, agent = store
    s.append_turn(session_id="sess-prune", agent_identity=agent, role="user", content="old turn to prune")
    import psycopg
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET ts = now() - interval '100 days' WHERE agent_identity = %s",
                (agent,),
            )
            conn.commit()
    assert s.prune_conversations(older_than_days=90, dry_run=True) >= 1
    assert s.prune_conversations(older_than_days=90, dry_run=False) >= 1
    assert s.count_turns(agent_identity=agent) == 0


def test_prune_disabled_is_noop(store):
    s, agent = store
    s.append_turn(session_id="s", agent_identity=agent, role="user", content="keep me around")
    assert s.prune_conversations(older_than_days=0) == 0
    assert s.count_turns(agent_identity=agent) == 1


# --- 002-dependent (skip unless migration 002 applied) --------------------

def test_register_agent_and_attribution(store):
    s, agent = store
    _needs_002(s)
    s.register_agent(agent_identity=agent, kind="theme")
    s.register_agent(agent_identity=agent, kind="worker")  # upsert updates kind + last_seen
    rows = [r for r in s.agent_attribution() if r["agent_identity"] == agent]
    assert rows and rows[0]["kind"] == "worker"
    import psycopg
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_agents WHERE agent_identity = %s", (agent,))
            conn.commit()


def test_record_delegation_edge(store):
    s, agent = store
    _needs_002(s)
    eid = s.record_delegation(
        parent_identity=agent, child_identity=agent + "-child",
        parent_session_id="p1", child_session_id="c1", kind="delegated",
    )
    assert isinstance(eid, int)
    import psycopg
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_agent_edges WHERE parent_identity = %s", (agent,))
            conn.commit()


def test_parent_session_id_column_write(store):
    s, agent = store
    _needs_002(s)
    rid = s.append_turn(
        session_id="child-sess", agent_identity=agent, role="assistant",
        content="a delegated result worth recalling later", parent_session_id="parent-sess",
    )
    assert isinstance(rid, int)
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT parent_session_id FROM conversations WHERE id = %s", (rid,))
            assert cur.fetchone()["parent_session_id"] == "parent-sess"

"""Smoke tests for the pgvector memory plugin.

These tests target the standalone modules (embed.py, store.py). The
provider class itself imports hermes-agent internals (agent.memory_provider,
tools.registry, …) and is only exercised when the plugin runs inside
hermes-agent.

Run with:
    pytest tests/

DB tests skip when PG_TEST_DSN is unset; embed live tests skip when
PG_TEST_EMBED_URL is unset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Repo root on sys.path so `pgvector.*` imports resolve without installing.
# (v0.4.2: bare `from embed import ...` only worked when another test file
# happened to leak pgvector/ onto sys.path first — order-dependent.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# embed.py
# ---------------------------------------------------------------------------

def test_pgvector_literal_roundtrip():
    from pgvector.embed import to_pgvector_literal

    lit = to_pgvector_literal([0.1, -0.25, 0.333333])
    assert lit.startswith("[") and lit.endswith("]")
    assert "0.1" in lit and "-0.25" in lit


def test_embed_empty_input_raises():
    from pgvector.embed import embed, EmbeddingError

    with pytest.raises(EmbeddingError):
        embed("", base_url="http://localhost:11434")
    with pytest.raises(EmbeddingError):
        embed("   ", base_url="http://localhost:11434")


def test_escape_like_literalizes_metacharacters():
    from pgvector.store import _escape_like

    assert _escape_like("15% YoY") == "15\\% YoY"
    assert _escape_like("a_b") == "a\\_b"
    assert _escape_like("C:\\U") == "C:\\\\U"
    assert _escape_like("plain text") == "plain text"


@pytest.mark.skipif(
    not os.environ.get("PG_TEST_EMBED_URL"),
    reason="PG_TEST_EMBED_URL not set",
)
def test_embed_live_returns_768_dims():
    from pgvector.embed import embed

    base_url = os.environ["PG_TEST_EMBED_URL"]
    vec = embed("hello world", base_url=base_url, model="nomic-embed-text")
    assert isinstance(vec, list)
    assert len(vec) == 768
    assert all(isinstance(x, (int, float)) for x in vec)


# ---------------------------------------------------------------------------
# store.py
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """Live MemoryStore against PG_TEST_DSN, with isolated agent_identity.

    Cleans up its own rows on teardown. Skips if PG_TEST_DSN is unset.
    """
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")

    from pgvector.store import MemoryStore

    s = MemoryStore(dsn)
    s.ensure_schema()
    agent = "pytest-smoke-" + os.urandom(4).hex()

    yield s, agent

    # Cleanup
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (agent,))
            conn.commit()


def test_health_reports_ok(store):
    s, _ = store
    h = s.health()
    assert h["ok"] is True
    assert h["error"] == ""
    assert h["row_count"] >= 0


def test_add_dedupes_on_exact_content(store):
    s, agent = store

    first = s.add(agent_identity=agent, target="memory", content="teal is my favorite")
    second = s.add(agent_identity=agent, target="memory", content="teal is my favorite")
    assert isinstance(first, int)
    assert second is None  # duplicate → no-op
    assert s.count(agent_identity=agent) == 1


def test_targets_are_independent(store):
    s, agent = store

    s.add(agent_identity=agent, target="memory", content="same content")
    s.add(agent_identity=agent, target="user", content="same content")
    assert s.count(agent_identity=agent, target="memory") == 1
    assert s.count(agent_identity=agent, target="user") == 1
    assert s.count(agent_identity=agent) == 2


def test_replace_substring_match(store):
    s, agent = store

    s.add(agent_identity=agent, target="memory", content="prefer dark mode in vscode")
    n = s.replace(
        agent_identity=agent,
        target="memory",
        old_text="dark mode",
        new_content="prefer high-contrast mode in vscode",
    )
    assert n == 1
    rows = s.list_entries(agent_identity=agent, target="memory", limit=10)
    assert rows[0]["content"] == "prefer high-contrast mode in vscode"


def test_remove_substring_match(store):
    s, agent = store

    s.add(agent_identity=agent, target="memory", content="entry one")
    s.add(agent_identity=agent, target="memory", content="entry two")
    n = s.remove(agent_identity=agent, target="memory", old_text="one")
    assert n == 1
    assert s.count(agent_identity=agent, target="memory") == 1


def test_search_with_fake_embeddings(store):
    s, agent = store

    # 768-dim vectors. Not real embeddings — just verifying the
    # similarity math and SQL round-trip behave correctly.
    vec_a = [0.1] * 768
    vec_b = [-0.1] * 768
    s.add(agent_identity=agent, target="memory", content="A entry", embedding=vec_a)
    s.add(agent_identity=agent, target="memory", content="B entry", embedding=vec_b)

    rows = s.search(query_embedding=vec_a, agent_identity=agent, limit=5)
    assert len(rows) == 2
    assert rows[0]["content"] == "A entry"
    assert rows[0]["score"] > rows[1]["score"]


def test_search_cross_agent_scope(store):
    s, agent = store
    other_agent = agent + "-other"
    try:
        vec = [0.2] * 768
        s.add(agent_identity=agent, target="memory", content="mine", embedding=vec)
        s.add(agent_identity=other_agent, target="memory", content="theirs", embedding=vec)

        scoped = s.search(query_embedding=vec, agent_identity=agent, limit=10)
        unscoped = s.search(query_embedding=vec, agent_identity=None, limit=10)
        assert len(scoped) == 1
        assert len(unscoped) >= 2  # both ours and theirs (plus any other test residue)
    finally:
        # Cleanup the second agent — primary fixture only knows about `agent`
        import psycopg
        with psycopg.connect(s._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (other_agent,))
                conn.commit()


def test_count_filters(store):
    s, agent = store
    assert s.count(agent_identity=agent) == 0
    s.add(agent_identity=agent, target="memory", content="x")
    s.add(agent_identity=agent, target="user", content="y")
    assert s.count(agent_identity=agent) == 2
    assert s.count(agent_identity=agent, target="memory") == 1
    assert s.count(agent_identity=agent, target="user") == 1


# ---------------------------------------------------------------------------
# v0.1.1: bulk_upsert_md
# ---------------------------------------------------------------------------

def test_bulk_upsert_md_skips_existing(store, tmp_path):
    s, agent = store

    md = tmp_path / "MEMORY.md"
    md.write_text(
        "first durable note about the build system"
        "\n§\n"
        "second note: the gateway runs on port 8642"
        "\n§\n"
        "third note: prefer pgvector over Holographic"
    )

    # First run: inserts 3 rows. embed_fn=None → text-only writes.
    r1 = s.bulk_upsert_md(agent_identity=agent, target="memory", file_path=md, embed_fn=None)
    assert r1 == {"parsed": 3, "inserted": 3, "skipped": 0}
    assert s.count(agent_identity=agent, target="memory") == 3

    # Second run on same file: all entries present, zero new inserts.
    r2 = s.bulk_upsert_md(agent_identity=agent, target="memory", file_path=md, embed_fn=None)
    assert r2 == {"parsed": 3, "inserted": 0, "skipped": 3}
    assert s.count(agent_identity=agent, target="memory") == 3


def test_bulk_upsert_md_missing_file(store, tmp_path):
    s, agent = store
    nope = tmp_path / "does-not-exist.md"
    r = s.bulk_upsert_md(agent_identity=agent, target="memory", file_path=nope, embed_fn=None)
    assert r == {"parsed": 0, "inserted": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# v0.2: conversation turns
# ---------------------------------------------------------------------------

@pytest.fixture
def store_with_turn_cleanup(store):
    """Same as `store`, but also wipes conversations on teardown."""
    s, agent = store
    yield s, agent
    import psycopg
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE agent_identity = %s", (agent,))
            conn.commit()


def test_append_turn_and_count(store_with_turn_cleanup):
    s, agent = store_with_turn_cleanup
    s.append_turn(
        session_id="sess-a",
        agent_identity=agent,
        role="user",
        content="discussing the deploy pipeline",
    )
    s.append_turn(
        session_id="sess-a",
        agent_identity=agent,
        role="assistant",
        content="acknowledged; updating CI step",
    )
    s.append_turn(
        session_id="sess-b",
        agent_identity=agent,
        role="user",
        content="different session, different topic",
    )
    assert s.count_turns(agent_identity=agent) == 3
    assert s.count_turns(agent_identity=agent, session_id="sess-a") == 2
    assert s.count_turns(agent_identity=agent, session_id="sess-b") == 1


def test_search_turns_with_fake_embeddings(store_with_turn_cleanup):
    s, agent = store_with_turn_cleanup
    vec_a = [0.1] * 768
    vec_b = [-0.1] * 768
    s.append_turn(
        session_id="sess-a", agent_identity=agent, role="user",
        content="alpha topic talk", embedding=vec_a,
    )
    s.append_turn(
        session_id="sess-a", agent_identity=agent, role="assistant",
        content="beta topic reply", embedding=vec_b,
    )
    rows = s.search_turns(query_embedding=vec_a, agent_identity=agent, limit=5)
    assert len(rows) == 2
    assert rows[0]["content"] == "alpha topic talk"
    assert rows[0]["score"] > rows[1]["score"]


def test_search_turns_session_scope(store_with_turn_cleanup):
    s, agent = store_with_turn_cleanup
    vec = [0.2] * 768
    s.append_turn(session_id="sess-a", agent_identity=agent, role="user", content="in-A", embedding=vec)
    s.append_turn(session_id="sess-b", agent_identity=agent, role="user", content="in-B", embedding=vec)
    scoped = s.search_turns(query_embedding=vec, agent_identity=agent, session_id="sess-a", limit=10)
    assert len(scoped) == 1
    assert scoped[0]["content"] == "in-A"

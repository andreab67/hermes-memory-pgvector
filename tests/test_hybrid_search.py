"""DB-gated tests for v0.4.1 hybrid (vector + full-text, RRF) recall.

Skip without PG_TEST_DSN. Point it at a THROWAWAY database. These tests only
touch rows under a random per-test agent_identity prefix and clean them up.

Migration 003 (the GIN full-text indexes) is NOT required for correctness — the
FTS predicate works on a seq scan too — so these tests pass on a 001-only schema.
They assert the fusion picks the right rows, not that a particular index is used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector.store import MemoryStore  # noqa: E402


# A uniform embedding: every row is equidistant, so the vector ranker is a tie
# and the full-text ranker is what breaks it. Lets us prove FTS actually
# contributes to the fused result.
FLAT = [0.1] * 768


@pytest.fixture
def store():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    s = MemoryStore(dsn)
    s.ensure_schema()
    agent = "pytest-hybrid-" + os.urandom(4).hex()
    yield s, agent
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity LIKE %s", (agent + "%",))
            cur.execute("DELETE FROM conversations WHERE agent_identity LIKE %s", (agent + "%",))
            conn.commit()


# --- memory_entries -------------------------------------------------------

def test_hybrid_surfaces_lexical_match_over_vector_tie(store):
    s, agent = store
    # All three rows share the SAME embedding, so cosine cannot distinguish
    # them. Only one carries the rare query token.
    s.add(agent_identity=agent, target="memory", content="the marketing plan for autumn", embedding=FLAT)
    s.add(agent_identity=agent, target="memory", content="notes about the sales pipeline", embedding=FLAT)
    s.add(agent_identity=agent, target="memory", content="deploy runbook mentions quokkazephyr flag", embedding=FLAT)

    rows = s.hybrid_search(query_text="quokkazephyr", query_embedding=FLAT, agent_identity=agent, limit=5)
    assert rows, "hybrid search returned nothing"
    assert "quokkazephyr" in rows[0]["content"], "lexical match did not rank first"
    assert rows[0]["rrf_score"] > 0
    assert rows[0]["score"] is not None  # this row has an embedding


def test_hybrid_fts_only_recovers_null_embedding_row(store):
    s, agent = store
    # Row stored text-only (embed endpoint was down): no embedding, so a
    # semantic ranking can't score it — the full-text leg is what recovers it.
    s.add(agent_identity=agent, target="memory", content="incident xylophonic outage postmortem", embedding=None)

    # Pure-vector search() has no `embedding IS NOT NULL` filter, so the row is
    # still returned — but only with a NULL score (undefined cosine distance),
    # never as a real semantic hit. The hybrid vec leg is what excludes it.
    vec_rows = s.search(query_embedding=FLAT, agent_identity=agent, limit=5)
    assert len(vec_rows) == 1 and vec_rows[0]["score"] is None

    # ...full-text-only hybrid (no query embedding) recovers it as a real match...
    fts_only = s.hybrid_search(query_text="xylophonic", query_embedding=None, agent_identity=agent, limit=5)
    assert len(fts_only) == 1
    assert fts_only[0]["score"] is None            # no embedding -> null cosine
    assert fts_only[0]["rrf_score"] > 0

    # ...and so does full hybrid, whose vec leg skips the NULL-embedding row.
    both = s.hybrid_search(query_text="xylophonic", query_embedding=FLAT, agent_identity=agent, limit=5)
    assert any("xylophonic" in r["content"] for r in both)


def test_hybrid_respects_target_and_agent_filters(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="banana protocol memory row", embedding=FLAT)
    s.add(agent_identity=agent, target="user", content="banana protocol user row", embedding=FLAT)
    s.add(agent_identity=agent + "-other", target="memory", content="banana protocol other agent", embedding=FLAT)

    only_user = s.hybrid_search(query_text="banana protocol", query_embedding=FLAT, agent_identity=agent, target="user", limit=10)
    assert only_user and all(r["target"] == "user" for r in only_user)
    assert all(r["agent_identity"] == agent for r in only_user)


def test_hybrid_empty_query_returns_empty(store):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="something recallable", embedding=FLAT)
    assert s.hybrid_search(query_text="   ", query_embedding=FLAT, agent_identity=agent) == []


# --- conversations --------------------------------------------------------

def test_hybrid_turns_surfaces_lexical_match(store):
    s, agent = store
    s.append_turn(session_id="s1", agent_identity=agent, role="user", content="how is the campaign going", embedding=FLAT)
    s.append_turn(session_id="s1", agent_identity=agent, role="assistant", content="we shipped the wobblefish feature", embedding=FLAT)

    rows = s.hybrid_search_turns(query_text="wobblefish", query_embedding=FLAT, agent_identity=agent, limit=5)
    assert rows and "wobblefish" in rows[0]["content"]
    assert rows[0]["role"] == "assistant"
    assert rows[0]["session_id"] == "s1"


def test_hybrid_turns_session_filter(store):
    s, agent = store
    s.append_turn(session_id="keep", agent_identity=agent, role="user", content="pomegranate topic here", embedding=FLAT)
    s.append_turn(session_id="drop", agent_identity=agent, role="user", content="pomegranate topic elsewhere", embedding=FLAT)

    rows = s.hybrid_search_turns(query_text="pomegranate", query_embedding=FLAT, session_id="keep", limit=5)
    assert rows and all(r["session_id"] == "keep" for r in rows)

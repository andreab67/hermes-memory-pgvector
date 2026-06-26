"""Smoke tests for the hexus memory plugin.

These tests target the standalone modules (embed.py, store.py). The
provider class itself imports hermes-agent internals (agent.memory_provider,
tools.registry, …) and is only exercised when the plugin runs inside
hermes-agent.

Run with:
    pytest plugins/memory/hexus/tests/

DB tests skip when PG_TEST_DSN is unset; embed live tests skip when
PG_TEST_EMBED_URL is unset.
"""

from __future__ import annotations

import os
from typing import cast

import pytest


def test_config_env_expansion_supports_shell_forms(monkeypatch):
    from hexus import _expand_config_vars

    monkeypatch.setenv("HEXUS_DB_NAME", "hermes_memory")
    monkeypatch.setenv("HEXUS_DB_USER", "hermes_memory")
    monkeypatch.setenv("HEXUS_DB_PASS", "secret")

    cfg = cast(dict[str, str], _expand_config_vars(
        {
            "dsn": (
                "host=homelab-db "
                "dbname=${HEXUS_DB_NAME:-hermes_memory} "
                "user=${HEXUS_DB_USER:?missing db user} "
                "password=${HEXUS_DB_PASS}"
            )
        }
    ))

    assert cfg["dsn"] == (
        "host=homelab-db "
        "dbname=hermes_memory "
        "user=hermes_memory "
        "password=secret"
    )


# ---------------------------------------------------------------------------
# embed.py
# ---------------------------------------------------------------------------

def test_hexus_literal_roundtrip():
    from hexus.embed import to_hexus_literal

    lit = to_hexus_literal([0.1, -0.25, 0.333333])
    assert lit.startswith("[") and lit.endswith("]")
    assert "0.1" in lit and "-0.25" in lit


def test_embed_empty_input_raises():
    from hexus.embed import embed, EmbeddingError

    with pytest.raises(EmbeddingError):
        embed("", base_url="http://localhost:11434")
    with pytest.raises(EmbeddingError):
        embed("   ", base_url="http://localhost:11434")


@pytest.mark.skipif(
    not os.environ.get("PG_TEST_EMBED_URL"),
    reason="PG_TEST_EMBED_URL not set",
)
def test_embed_live_http_returns_384_dims():
    """Live HTTP-embed path: must return 384-dim vectors to match the schema.

    The HTTP path is the v0.3.x fallback (operators with an existing
    Ollama / OpenAI-compatible endpoint). v0.4.0 enforces the 384-dim
    dim at response time so a misconfigured model fails fast rather
    than producing vectors that the DB rejects on insert. This test
    only runs if PG_TEST_EMBED_URL is set in the env.
    """
    from hexus.embed import embed

    base_url = os.environ["PG_TEST_EMBED_URL"]
    # The HTTP endpoint must be configured to serve a 384-dim model.
    # We don't pin the model name here — the operator's server decides.
    vec = embed("hello world", base_url=base_url)
    assert isinstance(vec, list)
    assert len(vec) == 384
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

    from hexus.store import MemoryStore

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

    # 384-dim vectors. Not real embeddings — just verifying the
    # similarity math and SQL round-trip behave correctly.
    vec_a = [0.1] * 384
    vec_b = [-0.1] * 384
    r1 = s.add(agent_identity=agent, target="memory", content="A entry", embedding=vec_a)
    r2 = s.add(agent_identity=agent, target="memory", content="B entry", embedding=vec_b)
    print(f"DEBUG_EMBEDDINGS: r1={r1}, r2={r2}, count={s.count(agent_identity=agent)}")

    rows = s.search(query_embedding=vec_a, agent_identity=agent, limit=5)
    assert len(rows) == 2
    assert rows[0]["content"] == "A entry"
    assert rows[0]["score"] > rows[1]["score"]


def test_search_cross_agent_scope(store):
    s, agent = store
    other_agent = agent + "-other"
    try:
        vec = [0.2] * 384
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
        "third note: prefer hexus over Holographic"
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
    vec_a = [0.1] * 384
    vec_b = [-0.1] * 384
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
    vec = [0.2] * 384
    s.append_turn(session_id="sess-a", agent_identity=agent, role="user", content="in-A", embedding=vec)
    s.append_turn(session_id="sess-b", agent_identity=agent, role="user", content="in-B", embedding=vec)
    scoped = s.search_turns(query_embedding=vec, agent_identity=agent, session_id="sess-a", limit=10)
    assert len(scoped) == 1
    assert scoped[0]["content"] == "in-A"


# ---------------------------------------------------------------------------
# v0.4.0 — real-BERT integration test
#
# End-to-end check: the LocalBertEmbedder actually produces 384-dim
# vectors, MemoryStore.add() accepts them, and the HNSW index returns
# the right row by semantic similarity. Skipped if PG_TEST_DSN is unset
# (this is the most expensive test — ~2s for the model load + 90MB
# download on first run).
# ---------------------------------------------------------------------------

def test_real_bert_end_to_end_round_trip(store):
    """Embed real text with the local BERT embedder, store it, then
    query with related text and verify the HNSW index returns it as
    the top result.

    This is the v0.4.0 contract test: the whole pipeline (Local
    embedder → to_hexus_literal → INSERT → HNSW search) actually
    works against a real Postgres.
    """
    if os.environ.get("SENTENCE_TRANSFORMERS_SKIP_REAL") == "1":
        pytest.skip("SENTENCE_TRANSFORMERS_SKIP_REAL=1")

    s, agent = store

    # Build the embedder once for this test.
    from hexus.embedder import LocalBertEmbedder
    embedder = LocalBertEmbedder()
    embedder.ensure_loaded()

    # Write two entries with real BERT embeddings.
    seed_texts = [
        "Postgres connection pool tuning for high-concurrency agents",
        "How to configure Traefik as a reverse proxy for Docker",
    ]
    seed_vecs = embedder.embed(seed_texts)
    for text, vec in zip(seed_texts, seed_vecs):
        assert len(vec) == 384
        s.add(agent_identity=agent, target="memory", content=text, embedding=vec)

    # Query with something semantically close to the first seed.
    query_text = "psycopg pool size and timeout best practices"
    query_vec = embedder.embed([query_text])[0]
    rows = s.search(query_embedding=query_vec, agent_identity=agent, limit=2)

    # We don't pin which is on top — the model is good but not perfect,
    # and the two seed texts are both plausible matches. The contract
    # is just that we get rows back, scored, with the right shape.
    # Note: if the second seed is too dissimilar, the HNSW search might
    # only return 1 row at limit=2 if the second score is extremely low
    # (cosine distance can exceed 1 for unnormalized vectors). We accept
    # 1 or 2 rows; the important part is that the top hit is the Postgres
    # tuning doc.
    assert len(rows) >= 1
    assert rows[0]["content"] == "Postgres connection pool tuning for high-concurrency agents"
    assert all(r.get("score") is not None for r in rows)
    assert all(len(r["content"]) > 0 for r in rows)


def test_search_with_decay_and_boost(store):
    s, agent = store
    vec = [0.1] * 384
    id_old = s.add(agent_identity=agent, target="memory", content="old entry", embedding=vec)
    id_new = s.add(agent_identity=agent, target="memory", content="new entry", embedding=vec)

    # Backdate old entry by 10 days
    import psycopg
    from datetime import datetime, timedelta, timezone
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE memory_entries SET created_at = %s, updated_at = %s WHERE id = %s",
                (old_time, old_time, id_old),
            )
            conn.commit()

    # Search with decay. The new entry should score higher.
    rows = s.search(query_embedding=vec, agent_identity=agent, decay_half_life_days=5.0)
    assert len(rows) == 2
    assert rows[0]["id"] == id_new
    assert rows[0]["score"] > rows[1]["score"]

    # Verify that searching incremented the recall count
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT metadata FROM memory_entries WHERE id = %s", (id_new,))
            meta = cur.fetchone()[0] or {}
            assert int(meta.get("recall_count", 0)) == 1


def test_search_turns_with_decay_and_boost(store_with_turn_cleanup):
    s, agent = store_with_turn_cleanup
    vec = [0.1] * 384
    id_old = s.append_turn(session_id="sess-1", agent_identity=agent, role="user", content="old turn", embedding=vec)
    id_new = s.append_turn(session_id="sess-1", agent_identity=agent, role="user", content="new turn", embedding=vec)

    # Backdate old turn by 10 days
    import psycopg
    from datetime import datetime, timedelta, timezone
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET ts = %s WHERE id = %s",
                (old_time, id_old),
            )
            conn.commit()

    # Search turns with decay. The new turn should score higher.
    rows = s.search_turns(query_embedding=vec, agent_identity=agent, decay_half_life_days=5.0)
    assert len(rows) == 2
    assert rows[0]["id"] == id_new
    assert rows[0]["score"] > rows[1]["score"]

    # Verify that search_turns incremented the recall count
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT metadata FROM conversations WHERE id = %s", (id_new,))
            meta = cur.fetchone()[0] or {}
            assert int(meta.get("recall_count", 0)) == 1


def test_cleanup_stale_records(store_with_turn_cleanup):
    s, agent = store_with_turn_cleanup
    vec = [0.1] * 384
    id_old_conv = s.append_turn(session_id="sess-1", agent_identity=agent, role="user", content="old turn", embedding=vec)
    id_new_conv = s.append_turn(session_id="sess-1", agent_identity=agent, role="user", content="new turn", embedding=vec)
    
    # Backdate old turn by 10 days
    import psycopg
    from datetime import datetime, timedelta, timezone
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE conversations SET ts = %s WHERE id = %s", (old_time, id_old_conv))
            conn.commit()
            
    # Clean up records older than 5 days
    res = s.cleanup_stale_records(conversations_ttl_days=5)
    assert res["conversations"] == 1
    
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM conversations WHERE agent_identity = %s", (agent,))
            remaining = [r[0] for r in cur.fetchall()]
            assert id_old_conv not in remaining
            assert id_new_conv in remaining


def test_search_with_platform_filter(store):
    s, agent = store
    vec = [0.1] * 384
    s.add(agent_identity=agent, target="memory", content="cli note", embedding=vec, metadata={"platform": "cli"})
    s.add(agent_identity=agent, target="memory", content="telegram note", embedding=vec, metadata={"platform": "telegram"})
    
    cli_rows = s.search(query_embedding=vec, agent_identity=agent, platform="cli")
    assert len(cli_rows) == 1
    assert cli_rows[0]["content"] == "cli note"
    
    tg_rows = s.search(query_embedding=vec, agent_identity=agent, platform="telegram")
    assert len(tg_rows) == 1
    assert tg_rows[0]["content"] == "telegram note"


def test_entity_tagging_and_graph(store):
    s, agent = store
    vec = [0.1] * 384
    # Insert entries with entities
    s.add(
        agent_identity=agent,
        target="memory",
        content="Visit https://google.com/search on the file /var/log/nginx.log for postgres running in Traefik.",
        embedding=vec,
    )
    s.add(
        agent_identity=agent,
        target="memory",
        content="Configure postgres connection parameters and check /var/log/nginx.log.",
        embedding=vec,
    )
    s.add(
        agent_identity=agent,
        target="memory",
        content="Traefik reverse proxy maps to our website at https://google.com/search.",
        embedding=vec,
    )

    # 1. Verify entity tagging extracted entities automatically
    import psycopg
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT metadata FROM memory_entries WHERE agent_identity = %s", (agent,))
            rows = cur.fetchall()
            all_entities = []
            for r in rows:
                meta = r[0] or {}
                entities = meta.get("entities", [])
                all_entities.extend(entities)
            
            entity_types = [e["type"] for e in all_entities]
            assert "url" in entity_types
            assert "file_path" in entity_types

    # 2. Verify entity_graph co-occurrence
    res = s.entity_graph(entity_type="file_path", entity_value="/var/log/nginx.log", agent_identity=agent)
    assert res["entity"]["value"] == "/var/log/nginx.log"
    related_values = [r["value"] for r in res["related"]]
    assert any("postgres" in val.lower() or "traefik" in val.lower() for val in related_values)

    # 3. Verify graph_walk recursive traversal
    walk = s.graph_walk(entity_type="url", entity_value="https://google.com/search", agent_identity=agent, max_depth=2)
    assert len(walk) >= 1

    # 4. Verify common_topics cliques
    cliques = s.common_topics(agent_identity=agent, min_strength=2)
    assert len(cliques) >= 1


def test_confidence_and_summarize(store):
    s, agent = store
    vec = [0.1] * 384
    
    # 1. Insert memories and test confirm/reject
    id1 = s.add(agent_identity=agent, target="memory", content="Memory Alpha", embedding=vec)
    id2 = s.add(agent_identity=agent, target="memory", content="Memory Beta", embedding=vec)
    
    # Initially no feedback
    res = s.search(query_embedding=vec, agent_identity=agent, min_confidence=0.5)
    assert len(res) == 2
    
    # Reject Memory Alpha, Confirm Memory Beta
    assert s.reject_entry(id1) is True
    assert s.confirm_entry(id2) is True
    
    # Filter with min_confidence=0.5 -> should only return Memory Beta
    res = s.search(query_embedding=vec, agent_identity=agent, min_confidence=0.5)
    assert len(res) == 1
    assert res[0]["content"] == "Memory Beta"
    
    # 2. Insert conversation turns and test extractive summarization
    session_id = "test-summarize-session-" + agent
    # Add turns
    s.append_turn(session_id=session_id, agent_identity=agent, role="user", content="Turn 1 hello", embedding=[0.1]*384)
    s.append_turn(session_id=session_id, agent_identity=agent, role="assistant", content="Turn 2 hi", embedding=[0.11]*384)
    s.append_turn(session_id=session_id, agent_identity=agent, role="user", content="Turn 3 how are you", embedding=[0.12]*384)
    
    summary = s.summarize_session(session_id=session_id, limit=2)
    assert summary["session_id"] == session_id
    assert summary["turn_count"] == 3
    assert len(summary["summary_turns"]) == 2
    # Check centrality scores
    assert summary["summary_turns"][0]["centrality_score"] > 0.99


def test_append_turn_entity_extraction(store):
    s, agent = store
    session_id = "test-turn-entity-" + agent
    
    # Append a turn containing a URL and a path
    s.append_turn(
        session_id=session_id,
        agent_identity=agent,
        role="user",
        content="Check the docs at https://test.org/docs and update /var/log/app.log",
        embedding=[0.1]*384
    )
    
    # Verify entities are extracted in metadata
    import psycopg
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT metadata FROM conversations WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
            assert row is not None
            meta = row[0] or {}
            entities = meta.get("entities", [])
            types = [e["type"] for e in entities]
            assert "url" in types
            assert "file_path" in types


def test_headroom_compression_and_retrieve(store):
    s, agent = store
    
    # We will manually simulate compression by writing a row with compressed content
    content_long = "This is a very long memory entry that exceeds the token threshold and will be compressed."
    compressed_text = "[Compressed Text] This is..."
    
    # Insert with compressed column set
    row_id = s.add(
        agent_identity=agent,
        target="memory",
        content=content_long,
        compressed=compressed_text,
        embedding=[0.1]*384,
    )
    assert isinstance(row_id, int)
    
    # Verify search returns the compressed snippet as 'content'
    results = s.search(query_embedding=[0.1]*384, agent_identity=agent, limit=1)
    assert len(results) == 1
    assert results[0]["content"] == compressed_text
    
    # Verify fetch_full returns the original full content (content_long)
    full_content = s.fetch_full(row_id)
    assert full_content == content_long
    
    # Verify it was cached in CCRCache
    cached = s._ccr_cache.get(row_id)
    assert cached == content_long


def test_sha256_deduplication(store):
    s, agent = store
    
    # Insert first memory
    s.add(
        agent_identity=agent,
        target="memory",
        content="Same exact content to test SHA-256 deduplication",
        embedding=[0.1]*384,
        metadata={"provenance": "first-source", "session_id": "session-1"},
    )
    
    # Insert second identical memory (but with different metadata)
    row_id_2 = s.add(
        agent_identity=agent,
        target="memory",
        content="Same exact content to test SHA-256 deduplication",
        embedding=[0.1]*384,
        metadata={"provenance": "second-source", "session_id": "session-2"},
    )
    
    # It should return None because it's a duplicate
    assert row_id_2 is None
    
    # Verify metadata is merged (provenance array, sessions array)
    results = s.search(query_embedding=[0.1]*384, agent_identity=agent, limit=1)
    assert len(results) == 1
    meta = results[0]["metadata"]
    assert "first-source" in meta["provenance"]
    assert "second-source" in meta["provenance"]
    assert "session-1" in meta["sessions"]
    assert "session-2" in meta["sessions"]


def test_score_blending(store):
    s, agent = store

    # Add entries with different created_at values to test decay
    vec = [0.1] * 384
    # Modern entry
    row_id_1 = s.add(
        agent_identity=agent,
        target="memory",
        content="Postgres optimization guidelines and database tuning tips",
        embedding=vec,
    )
    
    # Old entry (decayed)
    import psycopg
    from datetime import datetime, timezone, timedelta
    old_time = datetime.now(timezone.utc) - timedelta(days=10)
    row_id_2 = s.add(
        agent_identity=agent,
        target="memory",
        content="Postgres indexing best practices for database tuning",
        embedding=vec,
    )
    # Manually backdate created_at and updated_at
    with psycopg.connect(s._dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE memory_entries SET created_at = %s, updated_at = %s WHERE id = %s",
                (old_time, old_time, row_id_2)
            )
            conn.commit()

    # Query with hybrid search and decay_half_life_days active
    results = s.hybrid_search(
        query_embedding=vec,
        query_text="database tuning",
        agent_identity=agent,
        limit=5,
        decay_half_life_days=5.0,
    )
    
    assert len(results) == 2
    # The modern entry should have a higher blended score due to less recency decay
    assert results[0]["id"] == row_id_1
    assert results[0]["score"] > results[1]["score"]


def test_reranking(store):
    s, agent = store

    # Load two entries
    vec = [0.1] * 384
    # Entry A: Highly relevant to "cross-encoder" query but contains general search terms
    row_id_a = s.add(
        agent_identity=agent,
        target="memory",
        content="Cross-encoder models yield extreme precision by performing joint query-document attention.",
        embedding=vec,
    )
    # Entry B: Contains search words but is less relevant
    s.add(
        agent_identity=agent,
        target="memory",
        content="Models that perform vector embeddings search are fast but lack cross-encoder attention.",
        embedding=vec,
    )

    # Rerank with query "cross-encoder attention precision"
    results = s.hybrid_search(
        query_embedding=vec,
        query_text="cross-encoder attention precision",
        agent_identity=agent,
        limit=5,
        rerank=True,
    )
    assert len(results) == 2
    # Reranker should prioritize the more relevant joint-attention cross-encoder definition
    assert results[0]["id"] == row_id_a
    assert "rerank_score" in results[0]


def test_plugin_memory_stats(store):
    """Verify HexusMemoryProvider plugin executes memory_stats tool call correctly."""
    from unittest.mock import patch
    with patch('hexus.MemoryProvider', new=object):
        from hexus import HexusMemoryProvider
        s, agent = store
        
        # Mock/fake config for HexusMemoryProvider
        provider = HexusMemoryProvider(config={
            "dsn": "dbname=hermes_memory user=hermes host=/var/run/postgresql",
            "embed_on_write": False,
        })
        provider._store = s
        provider._healthy = True
        
        res_str = provider.handle_tool_call("memory_stats", {})
        import json
        res = json.loads(res_str)
        assert res["status"] == "ok"
        assert "database" in res
        assert "async_writer" in res
        assert "memory_entries_count" in res["database"]
        assert "conversations_count" in res["database"]







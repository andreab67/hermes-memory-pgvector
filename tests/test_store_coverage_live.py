"""Store behavior against isolated, disposable PostgreSQL databases.

Whole-table operations use scratch_store, never a shared/live deployment.
"""

from __future__ import annotations

import psycopg
import pytest

from hermes_pgvector.store import MemoryStore
pytest_plugins = ["test_v060_migrations_cli"]


VECTOR = [1.0] + [0.0] * 767
ORTHOGONAL = [0.0, 1.0] + [0.0] * 766


def _rows(store, sql, params=()):
    with store._get_pool().connection() as conn:
        return conn.execute(sql, params).fetchall()


def test_schema_verification_does_not_create_missing_objects(scratch_db):
    store = MemoryStore(scratch_db["runtime_dsn"])
    try:
        with pytest.raises(MemoryStore.SchemaNotApplied, match="Apply migrations as DB admin"):
            store.ensure_schema()
        assert store.ensure_migration_002_applied() is False
        assert store.health()["ok"] is False
        assert _rows(store, "SELECT tablename FROM pg_tables WHERE schemaname='public'") == []
    finally:
        store.close()


def test_attribution_is_disabled_until_admin_applies_002(scratch_db):
    store = MemoryStore(scratch_db["runtime_dsn"])
    try:
        store.apply_migration_as_admin(admin_dsn=scratch_db["admin_dsn"])
        store.ensure_schema()
        store.apply_migration_as_admin(
            admin_dsn=scratch_db["admin_dsn"], migration="004_runtime_grants.sql",
            runtime_role="hermes",
        )
        assert store.ensure_migration_002_applied() is False
        store.apply_migration_as_admin(
            admin_dsn=scratch_db["admin_dsn"], migration="002_agent_attribution.sql",
            runtime_role="hermes",
        )
        assert store.ensure_migration_002_applied() is True
    finally:
        store.close()


def test_bulk_import_blank_file_never_contacts_embed_or_database(tmp_path):
    store = MemoryStore("unused-test-dsn")
    file = tmp_path / "MEMORY.md"
    file.write_text(" \n\u00a0\n§\n\t\n", encoding="utf-8")

    def unexpected(_):
        raise AssertionError("blank entries reached embedding endpoint")

    assert store.bulk_upsert_md(
        agent_identity="blank", target="memory", file_path=file, embed_fn=unexpected,
    ) == {"parsed": 0, "inserted": 0, "skipped": 0, "failed": 0}
    assert store._pool is None


def test_bulk_import_breaker_stores_remaining_text_after_three_failures(scratch_store, tmp_path, caplog):
    store, identity = scratch_store
    entries = [f"outage entry {i}" for i in range(6)]
    file = tmp_path / "MEMORY.md"
    file.write_text(store.ENTRY_DELIMITER.join(entries), encoding="utf-8")
    attempted = []

    def unavailable(text):
        attempted.append(text)
        raise TimeoutError("embedding endpoint unavailable")

    report = store.bulk_upsert_md(
        agent_identity=identity, target="memory", file_path=file, embed_fn=unavailable,
    )
    assert report == {"parsed": 6, "inserted": 6, "skipped": 0, "failed": 0}
    assert attempted == entries[:3]
    rows = _rows(store, "SELECT content, embedding, metadata FROM memory_entries ORDER BY id")
    assert [row[0] for row in rows] == entries
    assert all(row[1] is None for row in rows)
    assert all(row[2] == {"source": "bulk_import", "file": str(file)} for row in rows)
    assert caplog.text.count("3 consecutive embed failures") == 1


def test_bulk_import_success_resets_breaker_and_duplicate_race_is_skipped(scratch_store, tmp_path):
    store, identity = scratch_store
    entries = ["fail first", "successful reset", "fail second", "fail third", "concurrent insert"]
    file = tmp_path / "MEMORY.md"
    file.write_text(store.ENTRY_DELIMITER.join(entries), encoding="utf-8")
    attempted = []

    def sometimes_available(text):
        attempted.append(text)
        if text.startswith("fail"):
            raise TimeoutError("temporary outage")
        if text == "concurrent insert":
            store.add(agent_identity=identity, target="memory", content=text, embedding=VECTOR)
        return VECTOR

    report = store.bulk_upsert_md(
        agent_identity=identity, target="memory", file_path=file, embed_fn=sometimes_available,
    )
    assert attempted == entries
    assert report == {"parsed": 5, "inserted": 4, "skipped": 1, "failed": 0}
    assert store.count(agent_identity=identity) == 5
    rows = _rows(store, "SELECT content, embedding IS NOT NULL FROM memory_entries ORDER BY id")
    assert dict(rows) == {entry: not entry.startswith("fail") for entry in entries}


def test_scoped_list_and_cosine_threshold_filter_persisted_rows(scratch_store):
    store, identity = scratch_store
    near = store.add(agent_identity=identity, target="user", content="reliable incident preference", embedding=VECTOR)
    store.add(agent_identity=identity, target="memory", content="orthogonal runbook", embedding=ORTHOGONAL)
    store.add(agent_identity=identity + "-other", target="user", content="foreign preference", embedding=VECTOR)
    listed = store.list_entries(agent_identity=identity, target="user", limit=1)
    assert [row["id"] for row in listed] == [near]
    matches = store.search(query_embedding=VECTOR, agent_identity=identity, min_similarity=0.9)
    assert [row["id"] for row in matches] == [near]
    assert matches[0]["score"] == pytest.approx(1.0)
    assert [row["id"] for row in store.search(
        query_embedding=VECTOR, agent_identity=identity, target="user",
    )] == [near]
    for session, vector in [("near-session", VECTOR), ("far-session", ORTHOGONAL)]:
        store.append_turn(session_id=session, agent_identity=identity, role="assistant", content="recorded " + session, embedding=vector)
    turns = store.search_turns(query_embedding=VECTOR, agent_identity=identity, min_similarity=0.9)
    assert [row["session_id"] for row in turns] == ["near-session"]
    assert store.count_turns(session_id="far-session") == 1
    assert store.count(target="user") == 2


def test_explicit_hybrid_candidate_depth_preserves_session_scope(scratch_store):
    store, identity = scratch_store
    store.append_turn(session_id="wanted", agent_identity=identity, role="user", content="quokkazephyr good", embedding=VECTOR)
    store.append_turn(session_id="other", agent_identity=identity, role="user", content="quokkazephyr wrong session", embedding=VECTOR)
    rows = store.hybrid_search_turns(
        query_text="quokkazephyr", query_embedding=VECTOR, agent_identity=identity,
        session_id="wanted", pool=2,
    )
    assert [row["session_id"] for row in rows] == ["wanted"]
    assert rows[0]["rrf_score"] > 0


def test_estimate_uses_planner_statistics_after_analyze(scratch_store, scratch_db):
    store, identity = scratch_store
    assert store.estimate_count("memory_entries") is None
    for i in range(3):
        store.add(agent_identity=identity, target="memory", content=f"planner estimate {i}")
    with psycopg.connect(scratch_db["admin_dsn"], autocommit=True) as conn:
        conn.execute("ANALYZE memory_entries")
    assert store.estimate_count("memory_entries") == 3
    assert store.health() == {"ok": True, "error": "", "row_count_estimate": 3}


def test_remap_duplicate_guard_preserves_data_then_force_merges(scratch_store):
    store, identity = scratch_store
    old, new = identity + "-old", identity + "-new"
    for i in range(11):
        for owner in (old, new):
            store.add(agent_identity=owner, target="memory", content=f"shared {i}")
    store.add(agent_identity=old, target="user", content="unique original", metadata={"origin": old})
    store.append_turn(session_id="child-session", agent_identity=old, role="user", content="historical delegation", parent_session_id="parent-session")
    dry = store.remap_identity(old_identity=old, new_identity=new)
    assert dry == {
        "dry_run": True, "memory_entries": {"moved": 1, "dropped_duplicates": 11},
        "conversations": {"moved": 1},
    }
    with pytest.raises(RuntimeError, match="would drop 11"):
        store.remap_identity(old_identity=old, new_identity=new, dry_run=False)
    assert store.count(agent_identity=old) == 12
    assert store.count(agent_identity=new) == 11
    assert store.count_turns(agent_identity=old) == 1
    assert store.remap_identity(old_identity=old, new_identity=new, dry_run=False, force=True) == {
        "dry_run": False, "memory_entries": {"moved": 1, "dropped_duplicates": 11},
        "conversations": {"moved": 1},
    }
    assert store.count(agent_identity=old) == 0
    assert store.count(agent_identity=new) == 12
    assert store.count_turns(agent_identity=new) == 1
    assert _rows(store, "SELECT parent_session_id FROM conversations") == [("parent-session",)]
    assert store.list_entries(agent_identity=new, target="user")[0]["metadata"] == {"origin": old}


def test_audit_record_round_trips_provenance(scratch_store):
    store, identity = scratch_store
    details = {"agent": identity, "reason": "operator request", "ids": [1, 2]}
    store.log_maintenance(
        operation="remap", target_table="memory_entries", identity_pattern=identity,
        affected_count=2, dropped_dupes=1, dry_run=False, details=details,
    )
    assert _rows(store, "SELECT operation, target_table, identity_pattern, affected_count, dropped_dupes, dry_run, details FROM memory_maintenance_log") == [
        ("remap", "memory_entries", identity, 2, 1, False, details),
    ]


def test_unscoped_lexical_recall_recovers_text_only_across_agents(scratch_store):
    store, identity = scratch_store
    other = identity + "-other"
    for owner in (identity, other):
        store.add(agent_identity=owner, target="memory", content="quokkazephyr " + owner)
        store.append_turn(session_id="lexical", agent_identity=owner, role="user", content="quokkazephyr " + owner)
    memories = store.hybrid_search(query_text="quokkazephyr", pool=4)
    turns = store.hybrid_search_turns(query_text="quokkazephyr", pool=4)
    for rows in (memories, turns):
        assert {row["agent_identity"] for row in rows} == {identity, other}
        assert all(row["score"] is None and row["rrf_score"] > 0 for row in rows)

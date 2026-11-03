"""Provider contracts at offline storage, scheduler, and host boundaries."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest

import hermes_pgvector as pkg
from hermes_pgvector import EmbeddingError, PgvectorMemoryProvider
from hermes_pgvector.writer import _PendingWrite


def _provider(*, config=None, store=None, writer=None):
    provider = PgvectorMemoryProvider(config=config)
    provider._healthy = True
    provider._store = store
    provider._writer = writer
    provider._session_id = "session-1"
    provider._agent_identity = "marketing"
    provider._raw_identity = "marketing"
    return provider


def test_memory_hook_survives_writer_scheduler_failure(caplog):
    writer = Mock()
    writer.enqueue.side_effect = RuntimeError("thread creation failed")
    provider = _provider(writer=writer)

    with caplog.at_level(logging.DEBUG):
        provider.on_memory_write("add", "memory", "A useful durable entry")

    assert writer.enqueue.call_count == 1
    assert "thread creation failed" in caplog.text


def test_initialize_survives_store_constructor_failure(monkeypatch, caplog):
    constructor = Mock(side_effect=RuntimeError("pool construction failed"))
    constructor.SchemaNotApplied = pkg.MemoryStore.SchemaNotApplied
    monkeypatch.setattr(pkg, "MemoryStore", constructor)
    provider = PgvectorMemoryProvider()

    provider.initialize("session-1")

    assert provider._healthy is False
    assert provider.prefetch("query") == ""
    assert json.loads(provider.handle_tool_call("recall_memory", {"query": "query"}))["error"] == "pgvector unavailable"
    assert "pool construction failed" in caplog.text
    provider.shutdown()


@pytest.mark.parametrize("failure", ["schema", "health", "registry"])
def test_initialize_degrades_at_store_probes(monkeypatch, failure):
    store = Mock()
    store.health.return_value = {"ok": True}
    store.ensure_migration_002_applied.return_value = False
    if failure == "schema":
        store.ensure_schema.side_effect = pkg.MemoryStore.SchemaNotApplied("apply migration 001")
    elif failure == "health":
        store.health.side_effect = RuntimeError("database offline")
    else:
        store.ensure_migration_002_applied.side_effect = RuntimeError("registry offline")
    constructor = Mock(return_value=store)
    constructor.SchemaNotApplied = pkg.MemoryStore.SchemaNotApplied
    monkeypatch.setattr(pkg, "MemoryStore", constructor)
    writer = Mock()
    monkeypatch.setattr(pkg, "AsyncWriter", Mock(return_value=writer))
    provider = PgvectorMemoryProvider(config={"bulk_sync_on_init": False})

    provider.initialize("session-1")

    assert provider._healthy is (failure == "registry")
    assert provider._delegation_enabled is False
    writer.enqueue.assert_not_called()
    provider.shutdown()
    store.close.assert_called_once_with()


def test_initialize_bulk_import_requires_enabled_write_context(monkeypatch, tmp_path):
    store = Mock()
    store.health.return_value = {"ok": True}
    store.ensure_migration_002_applied.return_value = True
    store.bulk_upsert_md.return_value = {"inserted": 0}
    monkeypatch.setattr(pkg, "MemoryStore", Mock(return_value=store))
    writer = Mock()
    monkeypatch.setattr(pkg, "AsyncWriter", Mock(return_value=writer))
    provider = PgvectorMemoryProvider(config={"embed_on_write": False})

    provider.initialize("read-only", agent_context="subagent", hermes_home=str(tmp_path))
    store.bulk_upsert_md.assert_not_called()
    writer.enqueue.assert_not_called()
    provider.initialize("writable", agent_context="cron", hermes_home=str(tmp_path))

    assert store.bulk_upsert_md.call_count == 2
    assert {call.kwargs["target"] for call in store.bulk_upsert_md.call_args_list} == {"memory", "user"}
    writer.enqueue.assert_called_once()
    assert writer.enqueue.call_args.kwargs["action"] == "register_agent"
    assert provider._draining is False
    provider.shutdown()


def test_plugin_config_uses_empty_fallback_for_missing_or_invalid_yaml(monkeypatch, tmp_path):
    home = types.ModuleType("hermes_constants")
    home.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", home)
    assert pkg._load_plugin_config() == {}
    (tmp_path / "config.yaml").write_text("plugins: [unterminated", encoding="utf-8")
    assert pkg._load_plugin_config() == {}


@pytest.mark.parametrize("value", [42, object()])
def test_invalid_theme_and_alias_containers_do_not_establish_identity_policy(value):
    assert pkg._as_theme_list(value) is None
    assert pkg._as_alias_map(value) == {}


def test_multimodal_flattening_ignores_empty_or_unknown_parts():
    content = ["", "first", 13, {"type": "text", "text": 7},
               {"type": "audio"}, {"type": "text", "text": "second"}]
    assert pkg._flatten_user_content(content) == "first\nsecond"


def test_provider_availability_tracks_optional_database_driver(monkeypatch):
    provider = PgvectorMemoryProvider()
    assert provider.name == "pgvector"
    assert provider.is_available() is True
    monkeypatch.setitem(sys.modules, "psycopg", None)
    assert provider.is_available() is False


def test_registration_uses_loaded_configuration_and_exposes_only_read_tools(monkeypatch):
    monkeypatch.setattr(pkg, "_load_plugin_config", lambda: {"allowed_themes": "sales"})
    context = Mock()
    pkg.register(context)
    provider = context.register_memory_provider.call_args.args[0]
    assert provider._config["allowed_themes"] == "sales"
    assert {item["name"] for item in provider.get_tool_schemas()} == {"recall_memory", "recall_conversation"}


def test_saving_config_failure_preserves_existing_file(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    original = "plugins: [this is invalid YAML"
    path.write_text(original, encoding="utf-8")

    PgvectorMemoryProvider().save_config({"allowed_themes": "sales"}, str(tmp_path))

    assert path.read_text(encoding="utf-8") == original
    assert "save_config failed" in caplog.text


@pytest.mark.parametrize("failure", ["absent", "embed", "query", "budget"])
def test_prefetch_failure_returns_no_prompt_block(monkeypatch, failure):
    store = Mock()
    embed = Mock(return_value=[0.1, 0.2])
    monkeypatch.setattr(pkg, "_embed_with_config", embed)
    provider = _provider(store=store)
    if failure == "absent":
        provider._store = None
    elif failure == "embed":
        embed.side_effect = EmbeddingError("endpoint offline")
    elif failure == "query":
        store.search.side_effect = RuntimeError("database offline")
    else:
        monkeypatch.setattr(pkg.time, "monotonic", lambda: 10.0)

    assert provider._compute_prefetch_block("query", timeout=1, deadline=9 if failure == "budget" else None) == ""
    if failure != "query":
        store.search.assert_not_called()


def test_prefetch_formats_bounded_snippets_and_handles_blank_query(monkeypatch):
    store = Mock()
    store.search.return_value = [{"content": "a" * 281}, {"content": " line\none ", "score": 0.8, "target": "user"}]
    monkeypatch.setattr(pkg, "_embed_with_config", Mock(return_value=[0.1]))
    provider = _provider(store=store)

    assert provider.prefetch("") == ""
    store.search.assert_not_called()
    block = provider.prefetch("query")

    assert block == "## Recall (pgvector, marketing)\n- [0.00] (?) " + "a" * 280 + "…\n- [0.80] (user) line one"
    assert store.search.call_args.kwargs["agent_identity"] == "marketing"


def test_queue_prefetch_scheduler_failure_leaves_sync_fallback_available(monkeypatch, caplog):
    scheduler = Mock(side_effect=RuntimeError("cannot start thread"))
    monkeypatch.setattr(pkg, "spawn_context_thread", scheduler)
    store = Mock()
    store.search.return_value = []
    monkeypatch.setattr(pkg, "_embed_with_config", lambda *a, **k: [0.1])
    provider = _provider(store=store)
    with caplog.at_level(logging.DEBUG):
        provider.queue_prefetch("query", session_id="session-1")
    assert "cannot start thread" in caplog.text
    assert provider.prefetch("query", session_id="session-1") == ""
    store.search.assert_called_once()


@pytest.mark.parametrize("hook", ["sync_turn", "on_session_end", "on_delegation"])
def test_capture_hook_survives_writer_failure_without_marking_capture(hook, caplog):
    writer = Mock()
    writer.enqueue.side_effect = RuntimeError("writer offline")
    provider = _provider(writer=writer)
    provider._delegation_enabled = True
    content = "A substantive conversation entry that should remain retryable after failure."
    with caplog.at_level(logging.DEBUG):
        if hook == "sync_turn":
            provider.sync_turn(content, "")
        elif hook == "on_session_end":
            provider.on_session_end([{"role": "user", "content": content}])
        else:
            provider.on_delegation("summarize", content)
    assert "writer offline" in caplog.text
    assert provider._turn_fingerprints == set()


@pytest.mark.parametrize("hook,gate", [
    (hook, gate)
    for hook in ("sync_turn", "on_session_end", "on_delegation", "on_memory_write")
    for gate in ("unhealthy", "write_context")
] + [("sync_turn", "capture_disabled"), ("on_session_end", "capture_disabled")])
def test_capture_gates_prevent_writer_calls(hook, gate):
    writer = Mock()
    provider = _provider(writer=writer, config={"sync_turns": gate != "capture_disabled"})
    provider._delegation_enabled = gate != "capture_disabled"
    provider._healthy = gate != "unhealthy"
    provider._writes_enabled = gate != "write_context"
    if hook == "sync_turn":
        provider.sync_turn("a" * 80, "b" * 80)
    elif hook == "on_session_end":
        provider.on_session_end([{"role": "user", "content": "a" * 80}])
    elif hook == "on_delegation":
        provider.on_delegation("task", "result")
    else:
        provider.on_memory_write("add", "memory", "durable entry")
    writer.enqueue.assert_not_called()


def test_sync_turn_recovers_invalid_minimum_and_preserves_raw_identity():
    writer = Mock()
    writer.enqueue.return_value = True
    provider = _provider(writer=writer, config={"turn_min_chars": "malformed"})
    provider._raw_identity = "MARKETING"

    provider.sync_turn("", "A substantive assistant message exceeding the default minimum.")

    assert writer.enqueue.call_count == 1
    captured = writer.enqueue.call_args.kwargs
    assert captured["extra"]["role"] == "assistant"
    assert captured["metadata"]["raw_identity"] == "MARKETING"
    assert len(provider._turn_fingerprints) == 1


def test_session_backstop_ignores_other_roles_and_noise_after_invalid_minimum():
    writer = Mock()
    writer.enqueue.return_value = True
    provider = _provider(writer=writer, config={"turn_min_chars": None})
    provider.on_session_end([
        {"role": "tool", "content": "t" * 80},
        {"role": "assistant", "content": "ok"},
        {"role": "assistant", "content": "Useful substantive answer exceeding the default noise cutoff."},
    ])
    assert writer.enqueue.call_count == 1
    assert writer.enqueue.call_args.kwargs["content"].startswith("Useful substantive answer")


def test_turn_fingerprint_cap_bounds_long_lived_provider_without_rejecting_new_turn():
    provider = PgvectorMemoryProvider()
    provider._FINGERPRINT_CAP = 2
    provider._remember_turn("old-1")
    provider._remember_turn("old-2")
    provider._remember_turn("new")
    assert provider._turn_fingerprints == {"new"}


@pytest.mark.parametrize("content", ["   ", "ok"])
def test_turn_noise_filter_rejects_blank_and_acknowledgement_even_with_zero_minimum(content):
    assert PgvectorMemoryProvider._is_noise(content, min_chars=0) is True


@pytest.mark.parametrize("role,content,policy,expected", [
    ("user", "question", "none", False),
    ("user", "question", "substantive_only", True),
    ("assistant", "short answer", "substantive_only", False),
    ("assistant", "a" * 120, "substantive_only", True),
])
def test_turn_embedding_policy_preserves_substantive_user_and_assistant_content(role, content, policy, expected):
    assert PgvectorMemoryProvider._should_embed_turn(role, content, policy) is expected


def test_skill_helper_runtime_error_preserves_user_content(monkeypatch):
    helper = types.ModuleType("agent.skill_commands")
    helper.extract_user_instruction_from_skill_message = Mock(side_effect=RuntimeError("host bug"))
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.skill_commands", helper)
    assert PgvectorMemoryProvider._strip_skill_scaffolding("user instruction") == "user instruction"


@pytest.mark.parametrize("action,target", [("add", "other"), ("unknown", "memory")])
def test_memory_hook_rejects_unsupported_operations(action, target):
    writer = Mock()
    _provider(writer=writer).on_memory_write(action, target, "content")
    writer.enqueue.assert_not_called()


def test_memory_metadata_contains_parent_and_raw_identity_without_changing_caller_data():
    writer = Mock()
    provider = _provider(writer=writer)
    provider._delegation_enabled = True
    provider._parent_session_id = "parent-session"
    provider._raw_identity = "MARKETING"
    metadata = {"old_text": "old", "previous_content": "exact prior row"}
    provider.on_memory_write("replace", "memory", "replacement", metadata)
    captured = writer.enqueue.call_args.kwargs
    assert captured["metadata"] == {"old_text": "old", "session_id": "session-1", "parent_session_id": "parent-session", "raw_identity": "MARKETING"}
    assert captured["extra"] == {"old_text": "old", "previous_content": "exact prior row"}
    assert metadata == {"old_text": "old", "previous_content": "exact prior row"}


def test_worker_without_store_drops_work_without_embedding(monkeypatch):
    embed = Mock()
    monkeypatch.setattr(pkg, "_embed_with_config", embed)
    _provider()._worker(_PendingWrite(action="add", agent_identity="marketing", target="memory", content="entry"))
    embed.assert_not_called()


def test_worker_replace_without_old_text_adds_new_content():
    store = Mock()
    provider = _provider(store=store, config={"embed_on_write": False})
    provider._worker(_PendingWrite(action="replace", agent_identity="marketing", target="memory", content="replacement", metadata={"session_id": "original"}))
    store.replace.assert_not_called()
    store.add.assert_called_once_with(agent_identity="marketing", target="memory", content="replacement", embedding=None, metadata={"session_id": "original"})


@pytest.mark.parametrize("action,method,extra", [
    ("register_agent", "register_agent", {"kind": "theme"}),
    ("edge", "record_delegation", {"parent_identity": "marketing", "child_identity": "sales", "child_session_id": "child-session"}),
])
def test_worker_dispatches_provenance_without_embedding(monkeypatch, action, method, extra):
    store = Mock()
    embed = Mock()
    monkeypatch.setattr(pkg, "_embed_with_config", embed)
    provider = _provider(store=store)
    provider._worker(_PendingWrite(action=action, agent_identity="marketing", target="provenance", content="", extra=extra))
    assert getattr(store, method).call_count == 1
    if action == "edge":
        assert store.record_delegation.call_args.kwargs == {"parent_identity": "marketing", "child_identity": "sales", "parent_session_id": None, "child_session_id": "child-session", "kind": "delegated", "attrs": {}}
    else:
        store.register_agent.assert_called_once_with(agent_identity="marketing", kind="theme")
    embed.assert_not_called()


def test_worker_database_errors_warn_once_but_keep_processing(caplog):
    store = Mock()
    store.add.side_effect = [RuntimeError("first database error"), RuntimeError("second database error"), None]
    provider = _provider(store=store, config={"embed_on_write": False})
    item = _PendingWrite(action="add", agent_identity="marketing", target="memory", content="durable entry")
    with caplog.at_level(logging.DEBUG):
        provider._worker(item)
        provider._worker(item)
        provider._worker(item)
    assert store.add.call_count == 3
    assert [record.levelname for record in caplog.records if "worker" in record.message] == ["WARNING", "DEBUG"]
    assert "first database error" in caplog.text
    assert "second database error" in caplog.text


def test_bulk_import_uses_host_home_and_continues_after_one_file_failure(monkeypatch, tmp_path, caplog):
    home = types.ModuleType("hermes_constants")
    home.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", home)
    store = Mock()
    store.bulk_upsert_md.side_effect = [RuntimeError("bad MEMORY.md"), {"parsed": 2, "inserted": 1, "skipped": 1}]
    provider = _provider(store=store, config={"embed_on_write": False})
    with caplog.at_level(logging.INFO):
        provider._bulk_sync_from_disk(None)
    assert [call.kwargs["file_path"] for call in store.bulk_upsert_md.call_args_list] == [tmp_path / "memories" / "MEMORY.md", tmp_path / "memories" / "USER.md"]
    assert all(call.kwargs["embed_fn"] is None for call in store.bulk_upsert_md.call_args_list)
    assert "bad MEMORY.md" in caplog.text
    assert "parsed=2 inserted=1 skipped=1" in caplog.text


def test_bulk_import_skips_without_store_or_resolvable_home():
    provider = _provider()
    provider._bulk_sync_from_disk("unused")
    store = Mock()
    provider._store = store
    provider._bulk_sync_from_disk(None)
    store.bulk_upsert_md.assert_not_called()


def test_repeated_embed_failures_warn_once_per_failure_kind(monkeypatch, caplog):
    embed = Mock(side_effect=[EmbeddingError("offline"), EmbeddingError("offline"), ValueError("bad vector"), ValueError("bad vector")])
    monkeypatch.setattr(pkg, "_embed_with_config", embed)
    provider = PgvectorMemoryProvider()
    for _ in range(4):
        assert provider._maybe_embed("content") is None
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 2
    assert provider._embed_bug_warned == {"ValueError"}


@pytest.mark.parametrize("tool", ["recall_memory", "recall_conversation"])
@pytest.mark.parametrize("failure", ["embed_only", "lexical_db", "vector_db", "both_db"])
def test_recall_failures_preserve_structured_errors_and_fallback_policy(monkeypatch, tool, failure):
    hybrid = failure != "embed_only" and failure != "vector_db"
    embed = Mock(return_value=[0.1])
    if failure in {"embed_only", "lexical_db"}:
        embed.side_effect = EmbeddingError("endpoint offline")
    monkeypatch.setattr(pkg, "_embed_with_config", embed)
    store = Mock()
    hybrid_method = "hybrid_search" if tool == "recall_memory" else "hybrid_search_turns"
    vector_method = "search" if tool == "recall_memory" else "search_turns"
    getattr(store, hybrid_method).side_effect = RuntimeError("lexical query failed")
    getattr(store, vector_method).side_effect = RuntimeError("vector query failed")
    provider = _provider(store=store, config={"hybrid_search": hybrid})

    result = json.loads(provider.handle_tool_call(tool, {"query": "query", "scope": "all"}))

    assert result["results"] == []
    assert result["count"] == 0
    assert result["error"].startswith("embed:" if failure == "embed_only" else "db:")
    if failure == "embed_only":
        getattr(store, hybrid_method).assert_not_called()
        getattr(store, vector_method).assert_not_called()
    elif failure == "lexical_db":
        assert getattr(store, hybrid_method).call_args.kwargs["query_embedding"] is None
        getattr(store, vector_method).assert_not_called()
    else:
        assert getattr(store, vector_method).call_args.kwargs["exclude_identities"] == ["whatsapp-dm", "external-group", "_bench"]
        assert getattr(store, vector_method).call_args.kwargs["agent_identity"] is None


@pytest.mark.parametrize("tool", ["recall_memory", "recall_conversation"])
def test_hybrid_recall_database_failure_falls_back_to_vector_without_losing_filters(monkeypatch, tool):
    monkeypatch.setattr(pkg, "_embed_with_config", lambda *a, **k: [0.1])
    store = Mock()
    hybrid_method = "hybrid_search" if tool == "recall_memory" else "hybrid_search_turns"
    vector_method = "search" if tool == "recall_memory" else "search_turns"
    getattr(store, hybrid_method).side_effect = RuntimeError("full text unavailable")
    getattr(store, vector_method).return_value = []
    provider = _provider(store=store)

    result = json.loads(provider.handle_tool_call(tool, {"query": "query", "scope": "Sales", "limit": 9}))

    assert result == {"results": [], "count": 0}
    fallback = getattr(store, vector_method).call_args.kwargs
    assert fallback["query_embedding"] == [0.1]
    assert fallback["agent_identity"] == "sales"
    assert fallback["limit"] == 9
    assert fallback["exclude_identities"] is None


@pytest.mark.parametrize("tool", ["recall_memory", "recall_conversation"])
def test_recall_serializes_timestamps_null_scores_and_bounded_content(monkeypatch, tool):
    monkeypatch.setattr(pkg, "_embed_with_config", lambda *a, **k: [0.1])
    stamp = datetime(2026, 11, 3, tzinfo=timezone.utc)
    row = {"id": 7, "agent_identity": "sales", "target": "user", "role": "assistant",
           "session_id": "past-session", "updated_at": stamp, "ts": stamp,
           "score": None, "rrf_score": 0.012345678, "content": "a" * 2100}
    store = Mock()
    method = store.hybrid_search if tool == "recall_memory" else store.hybrid_search_turns
    method.return_value = [row, {"score": 0.123456, "content": None}]
    provider = _provider(store=store)

    result = json.loads(provider.handle_tool_call(tool, {"query": "query"}))

    assert result["count"] == 2
    first, second = result["results"]
    assert first["id"] == 7
    assert first["ts"] == stamp.isoformat()
    assert first["score"] is None
    assert first["rrf_score"] == 0.012346
    assert first["content"] == "a" * 2000
    assert second["score"] == 0.1235
    assert second["ts"] is None
    assert second["content"] == ""
    assert "rrf_score" not in second
    if tool == "recall_conversation":
        assert first["session_id"] == "past-session"
        assert first["role"] == "assistant"


@pytest.mark.parametrize("tool", ["recall_memory", "recall_conversation"])
def test_recall_requires_available_store_and_nonblank_query(tool):
    unavailable = PgvectorMemoryProvider()
    result = json.loads(unavailable.handle_tool_call(tool, {"query": "query"}))
    assert result == {"results": [], "count": 0, "error": "pgvector unavailable"}
    store = Mock()
    missing_query = json.loads(_provider(store=store).handle_tool_call(tool, {"query": "  "}))
    assert "Missing required arg: query" in missing_query["error"]
    assert store.mock_calls == []


def test_unknown_recall_tool_returns_explicit_error():
    result = json.loads(PgvectorMemoryProvider().handle_tool_call("write_memory", {}))
    assert "Unknown tool: write_memory" in result["error"]


def test_memory_recall_rejects_session_scope_without_embedding(monkeypatch):
    embed = Mock()
    monkeypatch.setattr(pkg, "_embed_with_config", embed)
    result = json.loads(_provider(store=Mock()).handle_tool_call("recall_memory", {"query": "query", "scope": "session"}))
    assert "only valid for recall_conversation" in result["error"]
    embed.assert_not_called()


def test_worker_ignores_unrecognized_queued_action():
    store = Mock()
    _provider(store=store)._worker(_PendingWrite(action="unknown", agent_identity="marketing", target="memory", content="unused"))
    assert store.mock_calls == []


def test_worker_missing_replace_match_mirrors_new_content_as_add():
    store = Mock()
    store.replace.return_value = 0
    provider = _provider(store=store, config={"embed_on_write": False})
    provider._worker(_PendingWrite(action="replace", agent_identity="marketing", target="user", content="new entry", extra={"old_text": "prior entry"}))
    store.replace.assert_called_once_with(agent_identity="marketing", target="user", old_text="prior entry", new_content="new entry", new_embedding=None)
    store.add.assert_called_once_with(agent_identity="marketing", target="user", content="new entry", embedding=None, metadata={})


def test_worker_exact_remove_does_not_retry_with_broader_pattern():
    store = Mock()
    store.remove.return_value = 1
    provider = _provider(store=store)
    provider._worker(_PendingWrite(action="remove", agent_identity="marketing", target="memory", content="", extra={"previous_content": "exact row", "old_text": "row"}))
    store.remove.assert_called_once_with(agent_identity="marketing", target="memory", exact_content="exact row")
    store.add.assert_not_called()


@pytest.mark.parametrize("embed", [True, False])
def test_worker_turn_dispatch_preserves_session_parent_and_embedding_policy(monkeypatch, embed):
    request = Mock(return_value=[0.2])
    monkeypatch.setattr(pkg, "_embed_with_config", request)
    store = Mock()
    provider = _provider(store=store)
    provider._worker(_PendingWrite(action="turn", agent_identity="marketing", target="conversations", content="captured turn", extra={"role": "assistant", "session_id": "original-session", "parent_session_id": "parent-session", "embed": embed}, metadata={"source": "hook"}))
    store.append_turn.assert_called_once_with(session_id="original-session", agent_identity="marketing", role="assistant", content="captured turn", embedding=[0.2] if embed else None, metadata={"source": "hook"}, parent_session_id="parent-session")
    assert request.call_count == int(embed)


def test_save_config_creates_initial_plugin_configuration(tmp_path):
    import yaml

    PgvectorMemoryProvider().save_config({"allowed_themes": "marketing,sales"}, str(tmp_path))

    assert yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")) == {"plugins": {"pgvector": {"allowed_themes": "marketing,sales"}}}


def test_provider_uses_host_base_class_and_error_adapter_when_available(monkeypatch):
    """Load a separate package instance so host binding cannot leak to other tests."""
    host_provider = types.ModuleType("agent.memory_provider")
    host_provider.MemoryProvider = type("HostMemoryProvider", (), {})
    host_provider.spawn_context_thread = Mock()
    registry = types.ModuleType("tools.registry")
    registry.tool_error = lambda message: json.dumps({"host_error": message})
    config = types.ModuleType("hermes_cli.config")
    config.cfg_get = Mock(return_value={})
    for name, module in {
        "agent": types.ModuleType("agent"), "agent.memory_provider": host_provider,
        "tools": types.ModuleType("tools"), "tools.registry": registry,
        "hermes_cli": types.ModuleType("hermes_cli"), "hermes_cli.config": config,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    alias = "_provider_host_contract"
    for sibling in ("embed", "identity", "store", "writer"):
        monkeypatch.setitem(sys.modules, f"{alias}.{sibling}", import_module(f"hermes_pgvector.{sibling}"))
    source = Path(pkg.__file__)
    spec = importlib.util.spec_from_file_location(alias, source, submodule_search_locations=[str(source.parent)])
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, loaded)
    spec.loader.exec_module(loaded)

    provider = loaded.PgvectorMemoryProvider()

    assert isinstance(provider, host_provider.MemoryProvider)
    assert json.loads(provider.handle_tool_call("unsupported", {})) == {"host_error": "Unknown tool: unsupported"}
    assert loaded.spawn_context_thread is host_provider.spawn_context_thread
    assert loaded.cfg_get is config.cfg_get

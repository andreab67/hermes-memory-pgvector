"""Operator command contracts at the store and subprocess boundaries.

The store is a recording double: these tests run destructive commands through
the real parser without needing or modifying an operator's database.
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_pgvector import DEFAULTS
from hermes_pgvector import __main__ as cli


@pytest.fixture
def store(monkeypatch):
    value = Mock()
    value.health.return_value = {"ok": True}
    value.count.return_value = 4
    value.count_turns.return_value = 7
    value.ensure_migration_002_applied.return_value = True
    value.agent_attribution.return_value = [
        {"agent_identity": "marketing", "kind": "theme", "memory_rows": 4, "conversation_rows": 7},
        {"agent_identity": "default"},
    ]
    value.backfill_null_embeddings.return_value = {
        "memory_entries": {"remaining": 0, "unembeddable": 2},
        "conversations": {"remaining": 1},
    }
    value.prune_conversations.return_value = 3
    value.scan_pii.return_value = {"memory_entries": 1}
    value.delete_by_identity.return_value = {"memory_entries": 2, "conversations": 3}
    value.remap_identity.return_value = {
        "memory_entries": {"moved": 2, "dropped_duplicates": 1},
    }
    monkeypatch.setattr(cli, "MemoryStore", Mock(return_value=value))
    return value


@pytest.mark.parametrize("contents", ["", "plugins: null", "plugins:\n  pgvector: null"])
def test_empty_config_sections_resolve_to_empty_mapping(tmp_path, contents):
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8-sig")
    assert cli._load_config_file(str(path)) == {}


def test_config_loader_reads_bom_and_only_plugin_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("plugins:\n  other: {dsn: wrong}\n  pgvector: {dsn: selected}\n", encoding="utf-8-sig")
    assert cli._load_config_file(str(path)) == {"dsn": "selected"}


@pytest.mark.parametrize("contents", ["[unterminated", "- not-a-mapping"])
def test_malformed_explicit_config_stops_before_store_creation(tmp_path, store, capsys, contents):
    path = tmp_path / "config.yaml"
    path.write_text(contents, encoding="utf-8")
    assert cli.main(["prune", "--config", str(path), "--execute"]) == 1
    cli.MemoryStore.assert_not_called()
    assert "could not read config" in capsys.readouterr().err


def test_resolution_preserves_explicit_false_zero_and_empty_values():
    assert cli._resolve(argparse.Namespace(ttl_days=0), {"ttl_days": 90}, "ttl_days") == 0
    assert cli._resolve(argparse.Namespace(dsn=""), {"dsn": "other"}, "dsn") == ""
    assert cli._resolve(argparse.Namespace(), {"embed_on_write": False}, "embed_on_write") is False
    assert cli._resolve(argparse.Namespace(), {"ttl_days": None}, "ttl_days") == DEFAULTS["ttl_days"]


@pytest.mark.parametrize("migrations,role", [([], None), (["001_schema.sql", "002_attribution.sql"], "agents")])
def test_migrate_uses_admin_dsn_reports_effective_role_and_closes(store, capsys, migrations, role):
    store.apply_all_migrations.return_value = migrations
    argv = ["migrate", "--admin-dsn", "admin-only"]
    if role:
        argv += ["--runtime-role", role]
    assert cli.main(argv) == 0
    cli.MemoryStore.assert_called_once_with("admin-only")
    store.apply_all_migrations.assert_called_once_with(admin_dsn="admin-only", runtime_role=role)
    store.close.assert_called_once_with()
    output = capsys.readouterr().out
    assert (", ".join(migrations) if migrations else "(none found)") in output
    assert f"runtime role: {role or 'hermes'}" in output
    assert "migration 002 present: True" in output


@pytest.mark.parametrize("attribution", [False, True])
def test_stats_reports_counts_embedding_gaps_and_attribution(store, capsys, attribution):
    store.ensure_migration_002_applied.return_value = attribution
    assert cli.main(["stats", "--dsn", "runtime"]) == 0
    output = capsys.readouterr().out
    assert "memory_entries: 4 rows" in output
    assert "conversations:  7 rows" in output
    assert "2 un-embeddable" in output
    assert "conversations: 1 null-embedding rows" in output
    if attribution:
        assert "marketing" in output and "kind=theme" in output
        assert "default" in output and "kind=-" in output
    else:
        assert "NOT applied" in output
        store.agent_attribution.assert_not_called()
    kw = store.backfill_null_embeddings.call_args.kwargs
    assert kw["dry_run"] is True
    assert len(kw["embed_fn"]("count-only")) == kw["expected_dim"]
    store.close.assert_called_once_with()


@pytest.mark.parametrize("result,dry_run,expected", [
    ({"memory_entries": {"remaining": 0, "failed": 0, "succeeded": 2}}, False, 0),
    ({"memory_entries": {"remaining": 2, "failed": 0}}, False, 3),
    ({"memory_entries": {"remaining": 0, "failed": 1}}, False, 3),
    ({"memory_entries": {"note": "embed-unavailable", "remaining": 2}}, False, 2),
    ({"memory_entries": {"note": "aborted-consecutive-failures"}}, True, 0),
])
def test_backfill_exit_code_and_audit_follow_result(store, result, dry_run, expected):
    store.backfill_null_embeddings.return_value = result
    argv = ["backfill", "--tables", "memory_entries,conversations", "--batch-size", "17"]
    if dry_run:
        argv.append("--dry-run")
    assert cli.main(argv) == expected
    kw = store.backfill_null_embeddings.call_args.kwargs
    assert kw["tables"] == ("memory_entries", "conversations")
    assert kw["batch_size"] == 17 and kw["dry_run"] == dry_run
    if dry_run:
        store.log_maintenance.assert_not_called()
    else:
        info = result["memory_entries"]
        store.log_maintenance.assert_called_once_with(
            operation="backfill", target_table="memory_entries",
            affected_count=info.get("succeeded"), dry_run=False, details=info,
        )
    store.close.assert_called_once_with()


@pytest.mark.parametrize("execute,days", [(False, 8), (True, 8), (True, 0)])
def test_prune_defaults_to_dry_run_and_audits_only_active_deletions(store, execute, days):
    argv = ["prune", "--days", str(days)] + (["--execute"] if execute else [])
    assert cli.main(argv) == 0
    store.prune_conversations.assert_called_once_with(older_than_days=days, dry_run=not execute)
    if execute and days > 0:
        store.log_maintenance.assert_called_once_with(
            operation="prune", target_table="conversations", affected_count=3,
            dry_run=False, details={"days": days},
        )
    else:
        store.log_maintenance.assert_not_called()
    store.close.assert_called_once_with()


def test_prune_reads_string_ttl_from_config_when_days_omitted(tmp_path, store):
    path = tmp_path / "config.yaml"
    path.write_text("plugins:\n  pgvector:\n    ttl_days: '12'\n    dsn: selected\n", encoding="utf-8")
    assert cli.main(["prune", "--config", str(path)]) == 0
    cli.MemoryStore.assert_called_once_with("selected")
    store.prune_conversations.assert_called_once_with(older_than_days=12, dry_run=True)


@pytest.mark.parametrize("execute", [False, True])
def test_cleanup_strips_empty_identities_and_audits_each_deleted_table(store, execute):
    argv = ["cleanup", "--identities", " marketing, ,sales, ", "--tables", "memory_entries"]
    assert cli.main(argv + (["--execute"] if execute else [])) == 0
    store.scan_pii.assert_called_once_with(tables=("memory_entries",))
    store.delete_by_identity.assert_called_once_with(
        identities=["marketing", "sales"], tables=("memory_entries",), dry_run=not execute,
    )
    assert store.log_maintenance.call_count == (2 if execute else 0)
    if execute:
        for call, table, count in zip(store.log_maintenance.call_args_list,
                                      ("memory_entries", "conversations"), (2, 3), strict=True):
            assert call.kwargs == dict(operation="cleanup_delete", target_table=table,
                                       identity_pattern="marketing,sales", affected_count=count,
                                       dry_run=False, details={"pii_scan": {"memory_entries": 1}})
    store.close.assert_called_once_with()


@pytest.mark.parametrize("execute", [False, True])
def test_remap_passes_force_and_audits_counts(store, execute):
    argv = ["remap", "--old", "old", "--new", "new", "--force"]
    assert cli.main(argv + (["--execute"] if execute else [])) == 0
    store.remap_identity.assert_called_once_with(
        old_identity="old", new_identity="new", dry_run=not execute, force=True,
    )
    if execute:
        store.log_maintenance.assert_called_once_with(
            operation="remap_identity", identity_pattern="old->new", affected_count=2,
            dropped_dupes=1, dry_run=False, details=store.remap_identity.return_value,
        )
    else:
        store.log_maintenance.assert_not_called()
    store.close.assert_called_once_with()


def test_late_audit_failure_still_closes_store_and_returns_failure(store, capsys):
    store.log_maintenance.side_effect = RuntimeError("audit unavailable")
    assert cli.main(["remap", "--old", "a", "--new", "b", "--execute"]) == 1
    store.close.assert_called_once_with()
    assert "error: audit unavailable" in capsys.readouterr().err


def test_home_resolution_supports_host_constants_and_standalone_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setitem(sys.modules, "hermes_constants", SimpleNamespace(get_hermes_home=lambda: tmp_path))
    assert cli._resolve_hermes_home(None) == tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert cli._resolve_hermes_home(None) == tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "environment"))
    assert cli._resolve_hermes_home(None) == tmp_path / "environment"
    assert cli._resolve_hermes_home(str(tmp_path / "explicit")) == tmp_path / "explicit"


def test_remove_missing_install_is_successful(tmp_path, capsys):
    assert cli.main(["install", "--hermes-home", str(tmp_path), "--remove"]) == 0
    assert "nothing to remove" in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["remove", "refuse", "replace"])
def test_install_symlink_handling_preserves_target(tmp_path, monkeypatch, operation):
    target = tmp_path / "manual-provider"
    target.mkdir()
    (target / "__init__.py").write_text("original", encoding="utf-8")
    shim = tmp_path / "plugins" / "pgvector"
    shim.parent.mkdir()
    try:
        shim.symlink_to(target, target_is_directory=True)
    except OSError:
        # Windows can forbid symlink creation without Developer Mode. Model
        # that one OS property with a disposable file; unlink and all later
        # installer writes still use the real filesystem.
        shim.write_text("stand-in link entry", encoding="utf-8")
        real_is_symlink = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: (
            (path == shim and path.is_file()) or real_is_symlink(path)
        ))
    monkeypatch.setattr(subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=0, stdout=cli.__file__, stderr="",
    )))
    argv = ["install", "--hermes-home", str(tmp_path)]
    argv += {"remove": ["--remove"], "refuse": [], "replace": ["--force"]}[operation]
    assert cli.main(argv) == (1 if operation == "refuse" else 0)
    assert (target / "__init__.py").read_text(encoding="utf-8") == "original"
    assert shim.is_symlink() == (operation == "refuse")
    if operation == "replace":
        assert cli.SHIM_MARKER in (shim / "__init__.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("probe,expected", [
    (SimpleNamespace(returncode=1, stdout="", stderr="traceback\npackage absent\n"), "package absent"),
    (SimpleNamespace(returncode=1, stdout="", stderr=""), "(no output)"),
    (SimpleNamespace(returncode=0, stdout=str(Path("C:/another-package/__init__.py")), stderr=""), "shadowing"),
    (TimeoutError("probe timeout"), "could not verify shim import"),
])
def test_install_reports_import_probe_failures_without_failing_install(tmp_path, monkeypatch, capsys, probe, expected):
    runner = Mock(side_effect=probe) if isinstance(probe, Exception) else Mock(return_value=probe)
    monkeypatch.setattr(subprocess, "run", runner)
    assert cli.main(["install", "--hermes-home", str(tmp_path)]) == 0
    assert expected in capsys.readouterr().err
    assert (tmp_path / "plugins" / "pgvector" / "__init__.py").exists()
    assert runner.call_args.kwargs["timeout"] == 60


def test_install_tolerates_missing_cosmetic_manifest(tmp_path, monkeypatch, capsys):
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path(cli.__file__).parent / "plugin.yaml":
            raise FileNotFoundError("manifest absent")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(subprocess, "run", Mock(return_value=SimpleNamespace(
        returncode=0, stdout=cli.__file__, stderr="",
    )))
    assert cli.main(["install", "--hermes-home", str(tmp_path)]) == 0
    assert "could not copy plugin.yaml" in capsys.readouterr().err


def test_module_entrypoint_exits_with_parser_version(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["hermes-pgvector", "--version"])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("hermes_pgvector.__main__", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("hermes-pgvector ")

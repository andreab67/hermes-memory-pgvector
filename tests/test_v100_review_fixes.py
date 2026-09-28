"""Regression tests for the 1.0 full-codebase review fixes in store.py /
__main__.py.

  STORE-1/OPS-1 -- remap_identity(old == new) used to delete every row of
                   that identity (INSERT ... ON CONFLICT DO NOTHING inserted
                   nothing, then DELETE removed the originals). Now refused
                   by the store (ValueError) and surfaced by the CLI as a
                   non-zero exit, for --execute AND dry-run.
  TESTDOC-7     -- the remap >10 duplicates force guard.
  STORE-2       -- replace(old_text=<blank>) matches nothing.
  OPS-2         -- an explicit --config that cannot be read aborts before any
                   connection instead of running against DEFAULTS["dsn"].
  TESTDOC-8     -- a ~6000-char incompressible entry (M1) adds once and
                   dedupes; CLI --execute paths write memory_maintenance_log.

DB-gated tests skip without PG_TEST_DSN; every row is scoped to a unique
throwaway identity prefix and removed on teardown.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import __main__ as cli  # noqa: E402
from hermes_pgvector.store import MemoryStore  # noqa: E402

PG_TEST_DSN = os.environ.get("PG_TEST_DSN")
PG_TEST_ADMIN_DSN = os.environ.get("PG_TEST_ADMIN_DSN")


@pytest.fixture
def store():
    if not PG_TEST_DSN:
        pytest.skip("PG_TEST_DSN not set")
    s = MemoryStore(PG_TEST_DSN)
    s.ensure_schema()
    agent = "pytest-v100-" + os.urandom(4).hex()
    yield s, agent
    with psycopg.connect(PG_TEST_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity LIKE %s", (agent + "%",))
            cur.execute("DELETE FROM conversations WHERE agent_identity LIKE %s", (agent + "%",))
            conn.commit()
    if PG_TEST_ADMIN_DSN:
        try:
            with psycopg.connect(PG_TEST_ADMIN_DSN) as conn:
                conn.execute(
                    "DELETE FROM memory_maintenance_log WHERE identity_pattern LIKE %s",
                    (agent + "%",),
                )
        except Exception:  # noqa: BLE001 -- best-effort log cleanup
            pass
    s.close()


def _seed(s, agent, n=3):
    for i in range(n):
        assert s.add(agent_identity=agent, target="memory", content=f"row {agent} {i}") is not None


# --- STORE-1: identity-equal remap ------------------------------------------


@pytest.mark.parametrize("dry_run", [True, False])
def test_remap_identity_same_identity_is_refused_and_rows_survive(store, dry_run):
    s, agent = store
    _seed(s, agent)
    with pytest.raises(ValueError):
        s.remap_identity(old_identity=agent, new_identity=agent, dry_run=dry_run)
    assert s.count(agent_identity=agent) == 3


def test_remap_identity_whitespace_padded_identity_is_a_real_remap(store):
    # Matching is by exact string, so "x" -> "x  " is not a self-remap
    # (P2STORE-3A): the rows move instead of being refused or deleted.
    s, agent = store
    _seed(s, agent)
    padded = agent + "  "
    try:
        s.remap_identity(old_identity=agent, new_identity=padded, dry_run=False)
        assert s.count(agent_identity=agent) == 0
        assert s.count(agent_identity=padded) == 3
    finally:
        with s._get_pool().connection() as conn:
            conn.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (padded,))
            conn.commit()


@pytest.mark.parametrize("old,new", [("", "x"), ("x", ""), ("  ", "x"), ("x", " ")])
def test_remap_identity_blank_identity_is_refused(store, old, new):
    s, _agent = store
    with pytest.raises(ValueError):
        s.remap_identity(old_identity=old, new_identity=new, dry_run=False)


@pytest.mark.parametrize("execute", [False, True])
def test_cli_remap_same_identity_fails_and_keeps_rows(store, execute, capsys):
    s, agent = store
    _seed(s, agent)
    argv = ["remap", "--dsn", PG_TEST_DSN, "--old", agent, "--new", agent]
    if execute:
        argv.append("--execute")
    rc = cli.main(argv)
    assert rc != 0
    assert "error:" in capsys.readouterr().err
    assert s.count(agent_identity=agent) == 3


# --- TESTDOC-7: remap force guard -------------------------------------------


def _seed_duplicates(s, old, new, n=11):
    for i in range(n):
        assert s.add(agent_identity=old, target="memory", content=f"dup {old} {i}") is not None
        assert s.add(agent_identity=new, target="memory", content=f"dup {old} {i}") is not None


def test_remap_identity_force_guard_refuses_over_ten_duplicates(store):
    s, agent = store
    old, new = agent + "-old", agent + "-new"
    _seed_duplicates(s, old, new)

    dry = s.remap_identity(old_identity=old, new_identity=new, dry_run=True)
    assert dry["memory_entries"]["dropped_duplicates"] == 11

    with pytest.raises(RuntimeError, match="force"):
        s.remap_identity(old_identity=old, new_identity=new, dry_run=False)
    assert s.count(agent_identity=old) == 11, "refused remap must change nothing"
    assert s.count(agent_identity=new) == 11

    res = s.remap_identity(old_identity=old, new_identity=new, dry_run=False, force=True)
    assert res["memory_entries"]["dropped_duplicates"] == 11
    assert s.count(agent_identity=old) == 0
    assert s.count(agent_identity=new) == 11, "duplicates dropped, not doubled"


def test_cli_remap_execute_needs_force_over_ten_duplicates(store):
    s, agent = store
    old, new = agent + "-old", agent + "-new"
    _seed_duplicates(s, old, new)
    base = ["remap", "--dsn", PG_TEST_DSN, "--old", old, "--new", new]

    assert cli.main(base + ["--execute"]) != 0
    assert s.count(agent_identity=old) == 11
    assert s.count(agent_identity=new) == 11

    assert cli.main(base + ["--execute", "--force"]) == 0
    assert s.count(agent_identity=old) == 0
    assert s.count(agent_identity=new) == 11


# --- STORE-2: blank replace ---------------------------------------------------


@pytest.mark.parametrize("blank", ["", "  ", "\n\t"])
def test_replace_blank_old_text_matches_nothing(store, blank):
    s, agent = store
    s.add(agent_identity=agent, target="memory", content="first entry")
    s.add(agent_identity=agent, target="memory", content="second entry")
    n = s.replace(agent_identity=agent, target="memory", old_text=blank, new_content="OVERWRITTEN")
    assert n == 0
    contents = {r["content"] for r in s.list_entries(agent_identity=agent, target="memory", limit=10)}
    assert contents == {"first entry", "second entry"}


# --- OPS-2: explicit --config must be readable --------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["stats"],
        ["prune", "--execute", "--days", "1"],
        ["cleanup", "--identities", "x", "--execute"],
        ["remap", "--old", "a", "--new", "b", "--execute"],
        ["backfill"],
    ],
)
def test_unreadable_explicit_config_aborts_before_connecting(monkeypatch, tmp_path, capsys, argv):
    constructed = {"n": 0}

    class _NoConnect:
        def __init__(self, *a, **kw):
            constructed["n"] += 1

    monkeypatch.setattr(cli, "MemoryStore", _NoConnect)

    missing = str(tmp_path / "does-not-exist.yaml")
    rc = cli.main([argv[0], "--config", missing, *argv[1:]])
    assert rc != 0
    assert constructed["n"] == 0
    assert "could not read config" in capsys.readouterr().err

    bad = tmp_path / "bad.yaml"
    bad.write_text("plugins: [unclosed\n", encoding="utf-8")
    rc = cli.main([argv[0], "--config", str(bad), *argv[1:]])
    assert rc != 0
    assert constructed["n"] == 0


def test_no_config_flag_still_uses_defaults():
    assert cli._load_config_file(None) == {}


# --- TESTDOC-8: M1 large entry + maintenance log --------------------------------


def test_large_incompressible_entry_adds_once_then_dedupes(store):
    s, agent = store
    big = os.urandom(3000).hex()  # 6000 hex chars, not compressible
    assert len(big) == 6000
    assert s.add(agent_identity=agent, target="memory", content=big) is not None
    assert s.add(agent_identity=agent, target="memory", content=big) is None
    assert s.count(agent_identity=agent) == 1


def test_cli_cleanup_execute_writes_maintenance_log(store):
    s, agent = store
    _seed(s, agent, n=2)
    rc = cli.main(["cleanup", "--dsn", PG_TEST_DSN, "--identities", agent, "--tables", "memory_entries", "--execute"])
    assert rc == 0
    assert s.count(agent_identity=agent) == 0
    with psycopg.connect(PG_TEST_DSN) as conn:
        rows = conn.execute(
            "SELECT operation, target_table, affected_count, dry_run "
            "FROM memory_maintenance_log WHERE identity_pattern = %s",
            (agent,),
        ).fetchall()
    assert rows == [("cleanup_delete", "memory_entries", 2, False)]

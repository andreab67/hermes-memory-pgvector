"""tests/test_v060_migrations_cli.py -- WP-B: migrations + CLI.

Covers H1 (runtime-role GRANT resolution + NOTICE-not-error), L10 (ASCII-only
migration comments + SQL_ASCII-database migrate), M1's schema half (005's
md5(content) unique index replacing the raw-text one), L4 (backfill exit
codes, every command closing its store), L8's CLI half (cleanup passes
--tables to scan_pii), and --version.

DB-gated tests skip cleanly without PG_TEST_ADMIN_DSN / PG_TEST_DSN. Every
DB test that needs a migrated database creates and drops its OWN throwaway
database in the shared test cluster (via the `scratch_db` fixture below) --
`backfill_null_embeddings` and `apply_all_migrations` both operate on a whole
database, not a single agent_identity, so sharing one DB across test
functions here would make them interfere with each other.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import __main__ as cli  # noqa: E402
from hermes_pgvector.store import MemoryStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "hermes_pgvector" / "migrations"

# 001 and 003 are frozen (operators have applied them) and may never be
# edited, including for L10 -- so the ASCII check must not cover them.
ASCII_EXEMPT = {"001_schema.sql", "003_hybrid_search_fts.sql"}

PG_TEST_ADMIN_DSN = os.environ.get("PG_TEST_ADMIN_DSN")
PG_TEST_DSN = os.environ.get("PG_TEST_DSN")

requires_db = pytest.mark.skipif(
    not (PG_TEST_ADMIN_DSN and PG_TEST_DSN),
    reason="PG_TEST_ADMIN_DSN / PG_TEST_DSN not set",
)


def _dsn_with_db(dsn: str, dbname: str) -> str:
    parts = [p for p in dsn.split() if not p.startswith("dbname=")]
    return " ".join(parts + [f"dbname={dbname}"])


def _random_name(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def _shipped_migration_names():
    return sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))


def _raw_insert(dsn: str, *, agent_identity: str, target: str, content: str) -> None:
    """INSERT directly, bypassing store.add() -- which in this branch still
    names the pre-005 conflict target (agent_identity, target, content) and
    is owned by WP-A, running in parallel. It errors on ANY insert once 005
    has dropped that constraint (a repro'd, expected cross-WP interaction --
    see the WP-B report), so seeding rows for these tests goes straight to
    SQL instead of depending on WP-A's fix landing first.
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO memory_entries (agent_identity, target, content) "
            "VALUES (%s, %s, %s)",
            (agent_identity, target, content),
        )


@pytest.fixture
def scratch_db():
    """A freshly created, empty (unmigrated) throwaway database in the
    shared test cluster; dropped on teardown. Tests apply whichever
    migrations they need themselves."""
    if not (PG_TEST_ADMIN_DSN and PG_TEST_DSN):
        pytest.skip("PG_TEST_ADMIN_DSN / PG_TEST_DSN not set")
    name = _random_name("wpb_test")
    maint = _dsn_with_db(PG_TEST_ADMIN_DSN, "postgres")
    with psycopg.connect(maint, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    info = {
        "name": name,
        "admin_dsn": _dsn_with_db(PG_TEST_ADMIN_DSN, name),
        "runtime_dsn": _dsn_with_db(PG_TEST_DSN, name),
    }
    try:
        yield info
    finally:
        with psycopg.connect(maint, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _migrate(admin_dsn: str, **kwargs):
    store = MemoryStore(admin_dsn)
    try:
        return store.apply_all_migrations(admin_dsn=admin_dsn, **kwargs)
    finally:
        store.close()


# --- L10: migration files stay ASCII ---------------------------------------


def test_migration_files_are_ascii_except_001_and_003():
    """Every migration EXCEPT 001/003 must be pure ASCII (L10) -- this is a
    glob, so a new 006_... file is covered automatically without editing
    this test."""
    checked = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in ASCII_EXEMPT:
            continue
        checked.append(path.name)
        try:
            path.read_bytes().decode("ascii")
        except UnicodeDecodeError as exc:
            pytest.fail(f"{path.name} contains a non-ASCII byte: {exc}")
    assert checked, "no non-exempt migration files were found to check"


def test_001_and_003_are_the_only_ascii_exemptions():
    """Guards against the exemption set silently growing to swallow a file
    that should be checked."""
    names = set(_shipped_migration_names())
    assert ASCII_EXEMPT <= names
    assert ASCII_EXEMPT != names, "every migration is exempt -- exemption set is too broad"


# --- L10: migrate works against a SQL_ASCII-encoded database ---------------


@requires_db
def test_migrate_succeeds_on_sql_ascii_database():
    name = _random_name("wpb_ascii")
    maint = _dsn_with_db(PG_TEST_ADMIN_DSN, "postgres")
    with psycopg.connect(maint, autocommit=True) as conn:
        conn.execute(
            f"CREATE DATABASE \"{name}\" ENCODING 'SQL_ASCII' "
            "LC_COLLATE 'C' LC_CTYPE 'C' TEMPLATE template0"
        )
    try:
        admin_dsn = _dsn_with_db(PG_TEST_ADMIN_DSN, name)
        applied = _migrate(admin_dsn)
        assert applied == _shipped_migration_names()
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            assert conn.execute("SELECT to_regclass('memory_entries')").fetchone()[0]
    finally:
        with psycopg.connect(maint, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


# --- H1: runtime-role resolution --------------------------------------------


@requires_db
def test_runtime_role_grants_dml_to_a_custom_role(scratch_db):
    role = _random_name("wpb_rt")
    maint = _dsn_with_db(PG_TEST_ADMIN_DSN, "postgres")
    with psycopg.connect(maint, autocommit=True) as conn:
        conn.execute(f'CREATE ROLE "{role}"')
    try:
        applied = _migrate(scratch_db["admin_dsn"], runtime_role=role)
        assert applied == _shipped_migration_names()
        with psycopg.connect(scratch_db["admin_dsn"], autocommit=True) as conn:
            for table in ("memory_entries", "conversations", "memory_agents",
                          "memory_agent_edges", "memory_maintenance_log"):
                ok = conn.execute(
                    "SELECT has_table_privilege(%s, %s, 'INSERT')", (role, table)
                ).fetchone()[0]
                assert ok, f"{role} lacks INSERT on {table}"
            # The default 'hermes' role (created by test-env.sh in this
            # shared cluster) must NOT have been granted -- only the role
            # actually named by --runtime-role.
            hermes_has_it = conn.execute(
                "SELECT has_table_privilege('hermes', 'memory_entries', 'INSERT')"
            ).fetchone()[0]
            assert not hermes_has_it
    finally:
        # DROP OWNED BY must run against the DB that holds the grants
        # (scratch_db's), before the role itself can be dropped from the
        # shared cluster -- otherwise DROP ROLE fails with
        # DependentObjectsStillExist.
        with psycopg.connect(scratch_db["admin_dsn"], autocommit=True) as conn:
            conn.execute(f'DROP OWNED BY "{role}"')
        with psycopg.connect(maint, autocommit=True) as conn:
            conn.execute(f'DROP ROLE IF EXISTS "{role}"')


@requires_db
def test_missing_runtime_role_is_a_notice_not_an_error(scratch_db):
    role = _random_name("wpb_missing")  # deliberately never created
    # If the DO $$ ... $$ block raised instead of RAISE NOTICE-ing, this
    # would propagate as a psycopg exception and fail the test right here.
    applied = _migrate(scratch_db["admin_dsn"], runtime_role=role)
    assert applied == _shipped_migration_names()
    with psycopg.connect(scratch_db["admin_dsn"], autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)
        ).fetchone()
        assert exists is None, "test bug: role should not exist"
        # 002/003/004/005 still ran (H1's whole point: a GRANT failing must
        # not abort the rest of that migration file, or the ones after it).
        for regclass in ("memory_agents", "ix_memory_entries_content_fts", "memory_entries_unique_md5"):
            assert conn.execute(
                "SELECT to_regclass(%s)", (regclass,)
            ).fetchone()[0] is not None


def test_default_runtime_role_falls_back_to_hermes_in_cli_output():
    """`migrate` with no --runtime-role prints 'hermes' as the effective
    role (independent of any database -- pure argument-resolution check)."""
    args = cli.build_parser().parse_args(
        ["migrate", "--admin-dsn", "dbname=irrelevant"]
    )
    assert args.runtime_role is None
    # cmd_migrate computes `args.runtime_role or "hermes"` -- exercised
    # end-to-end against a real database in the H1 tests above.


# --- M1 (schema): 005 present, old constraint gone, idempotent -------------


@requires_db
def test_005_replaces_the_raw_text_unique_constraint(scratch_db):
    admin_dsn = scratch_db["admin_dsn"]
    _migrate(admin_dsn)
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        assert conn.execute(
            "SELECT to_regclass('memory_entries_unique_md5')"
        ).fetchone()[0] is not None
        assert conn.execute(
            "SELECT 1 FROM pg_constraint WHERE conname = 'memory_entries_unique'"
        ).fetchone() is None

    # Idempotent: re-running is a clean no-op (no error, same end state).
    applied_again = _migrate(admin_dsn)
    assert applied_again == _shipped_migration_names()
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        assert conn.execute(
            "SELECT to_regclass('memory_entries_unique_md5')"
        ).fetchone()[0] is not None


@requires_db
def test_upgrade_from_v055_is_a_noop_except_005(scratch_db):
    """Apply the v0.5.5 migration set exactly as an existing fleet host has
    it, seed a couple of rows, then run the NEW migrate: row counts must be
    unchanged, and only the M1 constraint swap should differ."""
    v055 = _v055_migration_texts()
    if v055 is None:
        pytest.skip("v0.5.5 migrations not available from this checkout's git history")

    admin_dsn = scratch_db["admin_dsn"]
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        for _name, sql_text in v055:
            conn.execute(sql_text.encode("utf-8"))

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO memory_entries (agent_identity, target, content) VALUES "
            "('wpb-upgrade', 'memory', 'seed one'), ('wpb-upgrade', 'memory', 'seed two')"
        )
        conn.execute(
            "INSERT INTO conversations (session_id, agent_identity, role, content) "
            "VALUES ('sess-1', 'wpb-upgrade', 'user', 'hello')"
        )

    def _counts():
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            m = conn.execute("SELECT count(*) FROM memory_entries").fetchone()[0]
            c = conn.execute("SELECT count(*) FROM conversations").fetchone()[0]
        return m, c

    before = _counts()
    applied = _migrate(admin_dsn)
    assert applied == _shipped_migration_names()
    assert _counts() == before

    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        assert conn.execute(
            "SELECT to_regclass('memory_entries_unique_md5')"
        ).fetchone()[0] is not None
        assert conn.execute(
            "SELECT 1 FROM pg_constraint WHERE conname = 'memory_entries_unique'"
        ).fetchone() is None

    # Second run: still a clean no-op.
    _migrate(admin_dsn)
    assert _counts() == before


def _v055_migration_texts():
    """The v0.5.5 migration set, vendored verbatim under tests/fixtures/
    (from `git show v0.5.5:hermes_pgvector/migrations/...`) so this test does
    not depend on tags being present in the checkout (CI clones shallow)."""
    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "migrations_v055"
    files = sorted(fixture_dir.glob("*.sql"))
    if not files:
        return None
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


# --- L4: backfill exit codes + every command closes its store -------------


@requires_db
def test_backfill_exit_0_when_nothing_remains(scratch_db, fake_embed_server):
    _migrate(scratch_db["admin_dsn"])
    url = fake_embed_server()
    rc = cli.main([
        "backfill", "--dsn", scratch_db["runtime_dsn"],
        "--embed-url", url, "--tables", "memory_entries,conversations",
    ])
    assert rc == 0


@requires_db
def test_backfill_exit_2_when_embed_endpoint_unavailable(scratch_db):
    _migrate(scratch_db["admin_dsn"])
    rc = cli.main([
        "backfill", "--dsn", scratch_db["runtime_dsn"],
        # Nothing listens on TCP port 1 -- refused immediately.
        "--embed-url", "http://127.0.0.1:1",
        "--tables", "memory_entries,conversations",
    ])
    assert rc == 2


@requires_db
def test_backfill_exit_3_when_a_row_fails_to_embed(scratch_db, fake_embed_server):
    _migrate(scratch_db["admin_dsn"])
    _raw_insert(
        scratch_db["runtime_dsn"], agent_identity="wpb-backfill",
        target="memory", content="this content is denied by policy",
    )
    url = fake_embed_server(fail_substring="denied")
    rc = cli.main([
        "backfill", "--dsn", scratch_db["runtime_dsn"],
        "--embed-url", url, "--tables", "memory_entries",
    ])
    assert rc == 3


@requires_db
def test_backfill_dry_run_always_exits_0_even_with_failures_pending(scratch_db, fake_embed_server):
    _migrate(scratch_db["admin_dsn"])
    _raw_insert(
        scratch_db["runtime_dsn"], agent_identity="wpb-backfill-dry",
        target="memory", content="this content is denied by policy",
    )
    url = fake_embed_server(fail_substring="denied")
    rc = cli.main([
        "backfill", "--dsn", scratch_db["runtime_dsn"],
        "--embed-url", url, "--tables", "memory_entries", "--dry-run",
    ])
    assert rc == 0


@requires_db
def test_every_command_closes_its_store(scratch_db, monkeypatch):
    """migrate/stats/backfill/prune/cleanup/remap all call store.close() in
    a finally, even when the command's own body raises."""
    _migrate(scratch_db["admin_dsn"])
    closed = {"n": 0}
    real_close = MemoryStore.close

    def _tracking_close(self):
        closed["n"] += 1
        return real_close(self)

    monkeypatch.setattr(MemoryStore, "close", _tracking_close)

    rc = cli.main(["stats", "--dsn", scratch_db["runtime_dsn"]])
    assert rc == 0
    assert closed["n"] == 1

    rc = cli.main([
        "prune", "--dsn", scratch_db["runtime_dsn"], "--days", "9999",
    ])
    assert rc == 0
    assert closed["n"] == 2

    rc = cli.main([
        "cleanup", "--dsn", scratch_db["runtime_dsn"],
        "--identities", "nobody-here",
    ])
    assert rc == 0
    assert closed["n"] == 3

    rc = cli.main([
        "remap", "--dsn", scratch_db["runtime_dsn"],
        "--old", "nobody", "--new", "nobody-else",
    ])
    assert rc == 0
    assert closed["n"] == 4


def test_migrate_closes_its_store_even_on_failure(monkeypatch):
    """cmd_migrate must close the store in `finally` even when
    apply_all_migrations() raises (e.g. a bad --admin-dsn)."""
    closed = {"n": 0}
    real_close = MemoryStore.close

    def _tracking_close(self):
        closed["n"] += 1
        return real_close(self)

    def _boom(self, **kw):
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(MemoryStore, "close", _tracking_close)
    monkeypatch.setattr(MemoryStore, "apply_all_migrations", _boom)

    args = cli.build_parser().parse_args(
        ["migrate", "--admin-dsn", "dbname=irrelevant-never-connected"]
    )
    with pytest.raises(RuntimeError):
        cli.cmd_migrate(args)
    assert closed["n"] == 1


# --- L8 (CLI): cleanup passes --tables to the PII scan ----------------------


def test_cleanup_passes_tables_to_scan_pii(monkeypatch, capsys):
    calls = {}

    class _FakeStore:
        def scan_pii(self, *, tables=None):
            calls["tables"] = tables
            return {"memory_entries": 0}

        def delete_by_identity(self, *, identities, tables=None, dry_run=True):
            calls["delete_tables"] = tables
            return {"memory_entries": 0}

        def close(self):
            pass

    monkeypatch.setattr(cli, "_make_store", lambda args, cfg: _FakeStore())
    monkeypatch.setattr(cli, "_load_config_file", lambda path: {})
    args = cli.build_parser().parse_args([
        "cleanup", "--identities", "nobody", "--tables", "memory_entries",
    ])
    rc = cli.cmd_cleanup(args)
    assert rc == 0
    assert calls["tables"] == ("memory_entries",)
    assert calls["delete_tables"] == ("memory_entries",)
    out = capsys.readouterr().out
    assert "10-11 digit numbers" in out
    assert "\\d" not in out, "the old regex-literal wording must not reappear"


def test_cleanup_defaults_tables_to_none(monkeypatch):
    calls = {}

    class _FakeStore:
        def scan_pii(self, *, tables=None):
            calls["tables"] = tables
            return {}

        def delete_by_identity(self, *, identities, tables=None, dry_run=True):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(cli, "_make_store", lambda args, cfg: _FakeStore())
    monkeypatch.setattr(cli, "_load_config_file", lambda path: {})
    args = cli.build_parser().parse_args(["cleanup", "--identities", "nobody"])
    assert cli.cmd_cleanup(args) == 0
    assert calls["tables"] is None


# --- --version ---------------------------------------------------------


def test_version_flag_prints_hermes_pgvector_prefixed_string(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["--version"])
    assert ei.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("hermes-pgvector ")
    assert out != "hermes-pgvector unknown", (
        "the package is pip-installed (editable) in the test environment; "
        "this exercises the found-metadata path, not the fallback"
    )


def test_version_string_falls_back_to_unknown_without_crashing(monkeypatch):
    import importlib.metadata as im

    def _raise(name):
        raise im.PackageNotFoundError(name)

    monkeypatch.setattr(im, "version", _raise)
    assert cli._version_string() == "hermes-pgvector unknown"


def test_version_available_without_a_subcommand():
    """--version must work even though `command` is otherwise required."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["--version"])
    assert ei.value.code == 0  # not argparse's "command is required" error (code 2)

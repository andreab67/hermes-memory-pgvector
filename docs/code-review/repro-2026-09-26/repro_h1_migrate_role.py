#!/usr/bin/env python3
"""H1 regression check: `hermes-pgvector migrate` must work for a runtime role
that is not named `hermes`.

Exit 0 = fixed, exit 1 = bug present, exit 2 = environment problem.

Needs a THROWAWAY cluster in which no role named `hermes` exists (for example a
fresh `pgvector/pgvector:pg17` container). ADMIN_DSN must be a superuser DSN to
that cluster's maintenance database, e.g.

    docker run -d --rm --name h1-pg -e POSTGRES_PASSWORD=postgres -p 55433:5432 pgvector/pgvector:pg17
    ADMIN_DSN='host=127.0.0.1 port=55433 user=postgres password=postgres dbname=postgres' \
        python docs/code-review/repro-2026-09-26/repro_h1_migrate_role.py

The script creates role `h1_runtime` and database `h1_repro`, runs migrate
(passing --runtime-role h1_runtime when the CLI supports it), then checks that
every shipped migration was applied and that h1_runtime can write.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROLE = "h1_runtime"
DB = "h1_repro"


def _dsn_with_db(dsn: str, db: str) -> str:
    parts = [p for p in dsn.split() if not p.startswith("dbname=")]
    return " ".join(parts + [f"dbname={db}"])


def main() -> int:
    admin = os.environ.get("ADMIN_DSN")
    if not admin:
        print("ADMIN_DSN not set", file=sys.stderr)
        return 2
    try:
        import psycopg
        import hermes_pgvector
    except ImportError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2

    with psycopg.connect(admin, autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = 'hermes'").fetchone():
            print("a role named 'hermes' exists in this cluster; use a fresh one", file=sys.stderr)
            return 2
        conn.execute(f"DROP DATABASE IF EXISTS {DB}")
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,)).fetchone():
            conn.execute(f"CREATE ROLE {ROLE} LOGIN PASSWORD '{ROLE}'")
        conn.execute(f"CREATE DATABASE {DB}")

    db_admin = _dsn_with_db(admin, DB)
    cli = [sys.executable, "-m", "hermes_pgvector", "migrate", "--admin-dsn", db_admin]
    help_text = subprocess.run(cli[:4] + ["--help"], capture_output=True, text=True).stdout
    if "--runtime-role" in help_text:
        cli += ["--runtime-role", ROLE]
    proc = subprocess.run(cli, capture_output=True, text=True)
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(f"FAIL: migrate exited {proc.returncode}: {proc.stderr.strip()}")
        print("BUG PRESENT")
        return 1

    shipped = sorted(p.name for p in (Path(hermes_pgvector.__file__).parent / "migrations").glob("*.sql"))
    missing = [m for m in shipped if m not in proc.stdout]
    ok = not missing
    if missing:
        print(f"FAIL: migrations not reported as applied: {missing}")

    with psycopg.connect(db_admin, autocommit=True) as conn:
        checks = {
            "memory_entries INSERT": "SELECT has_table_privilege(%s, 'memory_entries', 'INSERT')",
            "conversations INSERT": "SELECT has_table_privilege(%s, 'conversations', 'INSERT')",
            "memory_agents INSERT": "SELECT has_table_privilege(%s, 'memory_agents', 'INSERT')",
        }
        for label, sql in checks.items():
            if not conn.execute(sql, (ROLE,)).fetchone()[0]:
                print(f"FAIL: {ROLE} lacks {label}")
                ok = False

    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {DB}")
    print("PASS" if ok else "BUG PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

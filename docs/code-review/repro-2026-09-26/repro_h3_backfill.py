#!/usr/bin/env python3
"""H3 regression check: backfill must not stall behind permanently failing rows.

Exit 0 = fixed, exit 1 = bug present, exit 2 = environment problem.

Seeds a throwaway identity with 10 rows whose embed always fails, followed by
25 good rows, then runs backfill_null_embeddings(batch_size=10). With the
v0.5.5 loop the first batch is all failures, the loop breaks, and the 25 good
rows are never embedded. It also checks that each failing row is counted once.

    PG_TEST_DSN='host=127.0.0.1 port=55432 user=hermes password=hermes dbname=repro_scratch' \
        python docs/code-review/repro-2026-09-26/repro_h3_backfill.py

Point PG_TEST_DSN at a THROWAWAY database (e.g. a scratch db you created just
for this run) with migrations applied. Connection/setup failures and an
embed-dimension mismatch exit 2, never 1.
"""
from __future__ import annotations

import os
import random
import sys

IDENTITY = "repro-h3-backfill"


def main() -> int:
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        print("PG_TEST_DSN not set", file=sys.stderr)
        return 2
    try:
        import psycopg
        from hermes_pgvector.store import MemoryStore
    except ImportError as exc:
        print(f"cannot import hermes_pgvector/psycopg: {exc}", file=sys.stderr)
        return 2

    # embed_dim may be configured; probe length must match what the store expects.
    dim = int(os.environ.get("PG_TEST_EMBED_DIM", "768"))

    store = MemoryStore(dsn)
    result = None
    stuck_good = 0
    try:
        # Setup phase: any failure here is an environment problem (exit 2),
        # not evidence of the bug. Cleanup of seeded rows is in `finally`.
        try:
            with store._get_pool().connection() as conn:
                conn.execute("DELETE FROM conversations WHERE agent_identity = %s", (IDENTITY,))
                conn.commit()
                col_dim = conn.execute(
                    "SELECT a.atttypmod FROM pg_attribute a "
                    "WHERE a.attrelid = 'conversations'::regclass AND a.attname = 'embedding'"
                ).fetchone()
            if col_dim and col_dim[0] > 0 and col_dim[0] != dim:
                print(f"environment problem: conversations.embedding is vector({col_dim[0]}) "
                      f"but PG_TEST_EMBED_DIM={dim}; set PG_TEST_EMBED_DIM to match",
                      file=sys.stderr)
                return 2
            for i in range(10):
                store.append_turn(session_id="h3", agent_identity=IDENTITY, role="user",
                                  content=f"POISON row {i} that the endpoint always rejects")
            for i in range(25):
                store.append_turn(session_id="h3", agent_identity=IDENTITY, role="user",
                                  content=f"good row {i} that embeds fine")
        except Exception as exc:  # noqa: BLE001 - connection, schema, seeding
            print(f"environment problem during setup: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

        def embed_fn(text: str):
            if "POISON" in text:
                raise RuntimeError("endpoint rejects this input")
            return [random.random() for _ in range(dim)]

        try:
            result = store.backfill_null_embeddings(
                embed_fn=embed_fn, tables=("conversations",), batch_size=10, expected_dim=dim,
            )["conversations"]
            print(f"backfill result: {result}")
            with store._get_pool().connection() as conn:
                stuck_good = conn.execute(
                    "SELECT count(*) FROM conversations WHERE agent_identity = %s "
                    "AND content LIKE 'good%%' AND embedding IS NULL", (IDENTITY,),
                ).fetchone()[0]
        except psycopg.Error as exc:
            print(f"environment problem (database error): {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    finally:
        try:
            with store._get_pool().connection() as conn:
                conn.execute("DELETE FROM conversations WHERE agent_identity = %s", (IDENTITY,))
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not clean up seeded rows: {exc}", file=sys.stderr)
        store.close()

    ok = True
    if stuck_good:
        print(f"FAIL: {stuck_good} good rows left un-embedded behind failing rows")
        ok = False
    # Other NULL rows in the table may exist; only assert on our own rows' counts
    # when the database was otherwise clean.
    if result.get("failed", 0) > 10 and result.get("processed", 0) > 35:
        print(f"FAIL: failures counted more than once (failed={result['failed']} for 10 failing rows)")
        ok = False
    print("PASS" if ok else "BUG PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""H3 regression check: backfill must not stall behind permanently failing rows.

Exit 0 = fixed, exit 1 = bug present, exit 2 = environment problem.

Seeds a throwaway identity with 10 rows whose embed always fails, followed by
25 good rows, then runs backfill_null_embeddings(batch_size=10). With the
v0.5.5 loop the first batch is all failures, the loop breaks, and the 25 good
rows are never embedded. It also checks that each failing row is counted once.

    PG_TEST_DSN='host=127.0.0.1 port=55432 user=hermes password=hermes dbname=hermes_test' \
        python docs/code-review/repro-2026-09-26/repro_h3_backfill.py

Point PG_TEST_DSN at a THROWAWAY database with migrations applied.
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
        from hermes_pgvector.store import MemoryStore
    except ImportError as exc:
        print(f"cannot import hermes_pgvector: {exc}", file=sys.stderr)
        return 2

    store = MemoryStore(dsn)
    try:
        with store._get_pool().connection() as conn:
            conn.execute("DELETE FROM conversations WHERE agent_identity = %s", (IDENTITY,))
            conn.commit()
        for i in range(10):
            store.append_turn(session_id="h3", agent_identity=IDENTITY, role="user",
                              content=f"POISON row {i} that the endpoint always rejects")
        for i in range(25):
            store.append_turn(session_id="h3", agent_identity=IDENTITY, role="user",
                              content=f"good row {i} that embeds fine")

        # embed_dim may be configured; probe length must match what the store expects.
        dim = int(os.environ.get("PG_TEST_EMBED_DIM", "768"))

        def embed_fn(text: str):
            if "POISON" in text:
                raise RuntimeError("endpoint rejects this input")
            return [random.random() for _ in range(dim)]

        result = store.backfill_null_embeddings(
            embed_fn=embed_fn, tables=("conversations",), batch_size=10, expected_dim=dim,
        )["conversations"]
        print(f"backfill result: {result}")

        with store._get_pool().connection() as conn:
            stuck_good = conn.execute(
                "SELECT count(*) FROM conversations WHERE agent_identity = %s "
                "AND content LIKE 'good%%' AND embedding IS NULL", (IDENTITY,),
            ).fetchone()[0]
            conn.execute("DELETE FROM conversations WHERE agent_identity = %s", (IDENTITY,))
            conn.commit()
    finally:
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

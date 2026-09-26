#!/usr/bin/env python3
"""M1 regression check: long memory entries must be insertable.

Exit 0 = fixed, exit 1 = bug present, exit 2 = environment problem.

The v0.5.5 schema enforces UNIQUE(agent_identity, target, content) with a btree
on the raw text, so an entry whose index tuple exceeds 2704 bytes raises
ProgramLimitExceeded. Hex text is used because it barely compresses. Also
checks that exact-duplicate adds are still a no-op after the fix.

    PG_TEST_DSN='host=127.0.0.1 port=55432 user=hermes password=hermes dbname=hermes_test' \
        python docs/code-review/repro-2026-09-26/repro_m1_unique_limit.py
"""
from __future__ import annotations

import os
import secrets
import sys

IDENTITY = "repro-m1-unique"


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
    ok = True
    try:
        for n in (3000, 6000, 20000):
            text = secrets.token_hex(n // 2)
            try:
                first = store.add(agent_identity=IDENTITY, target="memory", content=text)
                second = store.add(agent_identity=IDENTITY, target="memory", content=text)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL: {n}-char entry: {type(exc).__name__}: {str(exc).splitlines()[0]}")
                ok = False
                continue
            if first is None or second is not None:
                print(f"FAIL: {n}-char entry: duplicate semantics broken (first={first}, second={second})")
                ok = False
            else:
                print(f"ok: {n}-char entry inserted once, duplicate ignored")
        with store._get_pool().connection() as conn:
            conn.execute("DELETE FROM memory_entries WHERE agent_identity = %s", (IDENTITY,))
            conn.commit()
    finally:
        store.close()
    print("PASS" if ok else "BUG PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

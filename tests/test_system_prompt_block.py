"""Unit test for PgvectorMemoryProvider.system_prompt_block() on a count()
failure. Pure unit test, no DB, no embed endpoint -- constructs the provider
via the standalone-import fallback (see test_tool_args_hardening.py) and
substitutes a stub store whose count() raises.

Regression coverage: system_prompt_block() calls self._store.count(...)
twice (scoped, then total) to decide between the "Empty store" message and
the normal entry-count message. A failed count() (DB blip, pool exhaustion,
connection drop after a healthy init -- _healthy is only ever probed once,
at initialize()) is NOT the same fact as "the store has zero rows". Treating
an exception as count_all == 0 asserts something FALSE to the model: it
tells the agent the store is active and empty, when it's actually
unreachable. The fix: catch the exception and return "" (no system-prompt
claim at all) instead of falling through to the empty-store branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import PgvectorMemoryProvider  # noqa: E402


class _RaisingCountStore:
    """Store stand-in whose count() always raises -- simulates a DB that
    went unreachable sometime after a healthy initialize()."""

    def count(self, **kwargs):
        raise RuntimeError("connection pool exhausted")


def test_system_prompt_block_returns_empty_string_when_count_fails():
    provider = PgvectorMemoryProvider()
    provider._healthy = True
    provider._store = _RaisingCountStore()
    provider._agent_identity = "default"

    result = provider.system_prompt_block()

    # The regression: pre-fix, a caught-and-defaulted count() failure fell
    # through to the "Empty store" branch and returned a string claiming the
    # store was active-and-empty. A merely unreachable store must say
    # NOTHING to the model, not something false.
    assert result == "", (
        f"expected empty string on a count() failure, got {result!r} -- "
        f"a failed count is not an empty store"
    )
    assert "Empty store" not in result

"""Regression lock for the on_session_switch(parent_session_id=...) contract.

Pure unit test on PgvectorMemoryProvider — no DB, no embed endpoint, no
hermes-agent runtime (constructs the provider via the standalone-import
fallback, same as test_turn_dedup.py / test_tool_args_hardening.py).

BACKGROUND — READ BEFORE "FIXING" on_session_switch AGAIN:

A previous change added this line to on_session_switch():

    self._parent_session_id = kwargs.get("parent_session_id") or None

It was REVERTED because it conflates two unrelated things that happen to
share a parameter name:

  * initialize()'s `parent_session_id` kwarg is a DELEGATION parent — the
    session_id of the PARENT AGENT that spawned this one as a subagent. This
    is what `_parent_session_id` is supposed to hold; it feeds
    conversations.parent_session_id, which migration 002
    (pgvector/migrations/002_agent_attribution.sql) defines as delegation
    traceback for cross-agent provenance.

  * on_session_switch()'s `parent_session_id` kwarg is something else
    entirely: the host (hermes_cli/cli_session_mixin.py) passes the
    PREVIOUS SESSION IN THIS SAME AGENT'S OWN LINEAGE, not a delegation
    parent:
      - cli_session_mixin.py:585 passes `old_session_id` on a normal
        rotation (/new, /resume, /branch, context-compression rollover).
      - cli_session_mixin.py:903 passes `""` on /undo.

If on_session_switch() ever assigns that value into `_parent_session_id`
again:
  1. Every write in the new session after a plain /new or /resume gets
     stamped with a FABRICATED delegation edge pointing at the agent's own
     previous session — there was no delegation, no subagent, nothing to
     trace back.
  2. The /undo path passes "" -> kwargs.get(...) or None -> None, which
     would ERASE a real subagent's parent_session_id that initialize() had
     already set for this session, silently severing genuine delegation
     provenance.

These two tests pin the revert: on_session_switch() must leave
`_parent_session_id` completely alone, no matter what it's called with. If
someone "helpfully" re-adds the assignment, both tests fail immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector import PgvectorMemoryProvider  # noqa: E402


def _provider_with_delegation_parent() -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider()
    # Simulate initialize() having already set a real delegation parent for
    # this session (e.g. this provider instance belongs to a subagent that
    # was spawned by another agent's session).
    p._parent_session_id = "delegation-parent-1"
    return p


def test_on_session_switch_does_not_overwrite_delegation_parent():
    """The common rotation case: /new or /resume pass the agent's own
    previous session_id as parent_session_id. If on_session_switch() copied
    that into _parent_session_id (the reverted line), this would fail with
    _parent_session_id == "previous-lineage-sess" instead of the untouched
    delegation parent."""
    p = _provider_with_delegation_parent()
    p.on_session_switch("new-sess", parent_session_id="previous-lineage-sess")
    assert p._parent_session_id == "delegation-parent-1", (
        "on_session_switch() must NEVER assign kwargs['parent_session_id'] "
        "into self._parent_session_id -- that kwarg is the agent's own "
        "previous session in this lineage (host: cli_session_mixin.py:585), "
        "not a delegation parent. See module docstring."
    )


def test_on_session_switch_undo_empty_string_does_not_erase_delegation_parent():
    """The /undo case (cli_session_mixin.py:903 passes parent_session_id="").
    kwargs.get("parent_session_id") or None -> None for "" -- if the reverted
    line were present, this call would NULL OUT a real subagent's delegation
    parent. Must stay untouched here too."""
    p = _provider_with_delegation_parent()
    p.on_session_switch("s", parent_session_id="")
    assert p._parent_session_id == "delegation-parent-1", (
        "on_session_switch(parent_session_id='') (the /undo path) must not "
        "erase a genuine delegation parent set by initialize()."
    )


def test_on_session_switch_still_resets_per_session_log_and_dedup_state():
    """The resets on_session_switch() DOES perform are correct and
    intentional (same rationale as initialize()'s per-session reset): stale
    one-shot warning flags would silence real signal for a new session, and a
    stale fingerprint set would make on_session_end() wrongly skip turns that
    belong to the NEW session. This is the counterpart assertion to the two
    tests above -- on_session_switch must be surgical: reset exactly these
    three, and nothing else (in particular, not _parent_session_id)."""
    p = _provider_with_delegation_parent()
    p._embed_warned = True
    p._db_warned = True
    p._turn_fingerprints = {"stale-fingerprint-from-old-session"}

    p.on_session_switch("new-sess", parent_session_id="previous-lineage-sess")

    assert p._embed_warned is False
    assert p._db_warned is False
    assert p._turn_fingerprints == set()
    # And still untouched, in the same call that performed the resets above.
    assert p._parent_session_id == "delegation-parent-1"

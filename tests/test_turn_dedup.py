"""Unit tests for PgvectorMemoryProvider._turn_fingerprint — pure static
method, no DB or embed endpoint, no hermes-agent runtime needed.

GROUP C regression coverage: the host calls BOTH sync_turn() (per exchange,
as the conversation happens) and on_session_end() (replays the WHOLE message
list again, e.g. on session rotation). `conversations` has no unique
constraint (unlike memory_entries, which dedupes on exact content), so
without a fingerprint-based guard the same turn is written to Postgres
twice -- doubling the table, the embedding cost, and the rows
recall_conversation returns for that turn. _turn_fingerprint() is the key
that guard is keyed on; these tests lock in that it is stable per (role,
content) and that dedup-by-set actually collapses repeats.

The tests above only cover the PRODUCER side (_turn_fingerprint itself, and
that sync_turn() only fingerprints accepted writes). The bottom section
covers the CONSUMER side: that on_session_end() actually consults
self._turn_fingerprints and skips a turn sync_turn() already enqueued this
session, instead of just building the set and never reading it back.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import PgvectorMemoryProvider  # noqa: E402


def test_fingerprint_is_stable_for_identical_role_and_content():
    a = PgvectorMemoryProvider._turn_fingerprint("user", "what time is the deploy?")
    b = PgvectorMemoryProvider._turn_fingerprint("user", "what time is the deploy?")
    assert a == b


def test_fingerprint_differs_on_content():
    a = PgvectorMemoryProvider._turn_fingerprint("user", "what time is the deploy?")
    b = PgvectorMemoryProvider._turn_fingerprint("user", "what time is the standup?")
    assert a != b


def test_fingerprint_differs_on_role():
    a = PgvectorMemoryProvider._turn_fingerprint("user", "acknowledged")
    b = PgvectorMemoryProvider._turn_fingerprint("assistant", "acknowledged")
    assert a != b


def test_fingerprint_is_callable_as_a_staticmethod_without_an_instance():
    # sync_turn()/on_session_end() call it as self._turn_fingerprint(...), but
    # it must not actually require an instance -- confirms it stays a
    # @staticmethod (not accidentally becoming a bound method that needs
    # self, which would break the class-level call sites already in the repo
    # if it were ever converted incorrectly).
    fp = PgvectorMemoryProvider._turn_fingerprint("assistant", "on it")
    assert isinstance(fp, str) and fp


def test_dedup_contract_same_turn_fingerprinted_twice_collapses_in_a_set():
    # This is the actual guard sync_turn()/on_session_end() rely on:
    # `self._turn_fingerprints` is a set, and on_session_end() skips any
    # fingerprint already present from this session's sync_turn() calls.
    seen: set = set()
    seen.add(PgvectorMemoryProvider._turn_fingerprint("user", "ship it"))
    # Same (role, content) captured again later (e.g. replayed by
    # on_session_end after sync_turn already enqueued it) must be the exact
    # same fingerprint, so the set does not grow.
    seen.add(PgvectorMemoryProvider._turn_fingerprint("user", "ship it"))
    assert len(seen) == 1

    # A genuinely different turn does grow the set -- the guard isn't just
    # collapsing everything.
    seen.add(PgvectorMemoryProvider._turn_fingerprint("assistant", "shipped"))
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# Drop-safety: a turn the writer REJECTED must not be fingerprinted.
#
# sync_turn() records a fingerprint so on_session_end() can skip turns already
# captured. But AsyncWriter.enqueue() returns False when its bounded queue is
# full and the write is DROPPED. Fingerprinting such a turn anyway would make
# on_session_end skip a row that never reached Postgres, converting a
# recoverable drop into permanent data loss -- the opposite of the backstop
# on_session_end is supposed to be.
# ---------------------------------------------------------------------------

class _RejectingWriter:
    """Writer stand-in whose queue is always full (enqueue always drops).

    Counts only action="turn" so on_session_end's register_agent enqueue does
    not inflate the total."""

    def __init__(self):
        self.calls = 0

    def enqueue(self, **kwargs) -> bool:
        if kwargs.get("action") == "turn":
            self.calls += 1
        return False


class _AcceptingWriter:
    """Counts only action="turn" -- see _RejectingWriter."""

    def __init__(self):
        self.calls = 0

    def enqueue(self, **kwargs) -> bool:
        if kwargs.get("action") == "turn":
            self.calls += 1
        return True


def _provider_for_turns(writer):
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._writer = writer
    p._agent_identity = "default"
    p._raw_identity = "default"
    p._session_id = "sess-drop"
    p._turn_fingerprints = set()
    return p


_LONG_USER = "This is a substantive user turn well past the noise threshold."
_LONG_ASSISTANT = "And this is a substantive assistant reply, also long enough."


def test_dropped_turn_is_not_fingerprinted():
    """A full queue must leave the fingerprint set empty, so on_session_end
    still re-captures the turn instead of skipping a write that never landed."""
    writer = _RejectingWriter()
    p = _provider_for_turns(writer)
    p.sync_turn(_LONG_USER, _LONG_ASSISTANT)
    assert writer.calls == 2, "both turns should have been offered to the writer"
    assert p._turn_fingerprints == set(), (
        "dropped turns must NOT be fingerprinted -- on_session_end would then "
        "skip them and the turn would be lost permanently"
    )


def test_accepted_turn_is_fingerprinted():
    """The accepted path still records fingerprints, so on_session_end skips
    the turns sync_turn already persisted (the double-write guard)."""
    writer = _AcceptingWriter()
    p = _provider_for_turns(writer)
    p.sync_turn(_LONG_USER, _LONG_ASSISTANT)
    assert writer.calls == 2
    assert len(p._turn_fingerprints) == 2, "accepted turns must be fingerprinted"
    expected = {
        PgvectorMemoryProvider._turn_fingerprint("user", _LONG_USER),
        PgvectorMemoryProvider._turn_fingerprint("assistant", _LONG_ASSISTANT),
    }
    assert p._turn_fingerprints == expected


# ---------------------------------------------------------------------------
# Consumer side: on_session_end() must actually SKIP a turn already
# fingerprinted by sync_turn(), not just let sync_turn() populate a set
# nobody reads. Requires _delegation_enabled=True -- on_session_end() is a
# no-op until migration 002 is applied (see pgvector/__init__.py).
# ---------------------------------------------------------------------------

class _CountingWriter:
    """Writer stand-in that accepts every write and records each call's
    `action`, so a test can isolate action=="turn" enqueues from the
    action=="register_agent" enqueue on_session_end() always issues."""

    def __init__(self):
        self.actions: list = []

    def enqueue(self, **kwargs) -> bool:
        self.actions.append(kwargs.get("action"))
        return True


def _provider_for_session_end(writer) -> PgvectorMemoryProvider:
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._delegation_enabled = True  # on_session_end() no-ops otherwise
    p._writer = writer
    p._agent_identity = "default"
    p._raw_identity = "default"
    p._session_id = "sess-consumer-dedup"
    p._turn_fingerprints = set()
    return p


def test_on_session_end_skips_a_turn_already_captured_by_sync_turn():
    """The actual consumer-side contract: sync_turn() fingerprints the
    (user, assistant) exchange as it happens; on_session_end() then replays
    the full message list (the host calls both hooks, and calls
    on_session_end() again on session rotation). Without on_session_end()
    checking `fp in self._turn_fingerprints` and skipping, the same two
    turns get enqueued a second time -- 4 action=="turn" enqueues instead of
    2. If that skip were ever removed, this test fails: 4 != 2."""
    writer = _CountingWriter()
    p = _provider_for_session_end(writer)

    p.sync_turn(_LONG_USER, _LONG_ASSISTANT)
    p.on_session_end(
        [
            {"role": "user", "content": _LONG_USER},
            {"role": "assistant", "content": _LONG_ASSISTANT},
        ]
    )

    turn_calls = [a for a in writer.actions if a == "turn"]
    assert len(turn_calls) == 2, (
        f"expected exactly 2 action=='turn' enqueues (sync_turn's own two), "
        f"got {len(turn_calls)} -- on_session_end() is re-enqueuing turns "
        f"sync_turn() already fingerprinted this session"
    )
    # Sanity: on_session_end() does independently enqueue one
    # action=="register_agent" every call -- not part of the dedup contract,
    # just confirms the writer stub is actually wired into both hooks.
    register_calls = [a for a in writer.actions if a == "register_agent"]
    assert len(register_calls) == 1


# ---------------------------------------------------------------------------
# Drop-safety, on_session_end side.
#
# The same contract as sync_turn, and arguably more important here: this hook
# replays an entire transcript at once through a 256-slot queue, so a full
# queue is MOST likely exactly at this moment. Fingerprinting a dropped turn
# would make the next on_session_end -- session rotation replays the same
# message list -- skip it, converting a recoverable drop into permanent loss.
# ---------------------------------------------------------------------------

_LONG_U = "A substantive user turn well past the noise threshold for capture."
_LONG_A = "A substantive assistant reply, also long enough to be captured here."
_MSGS = [{"role": "user", "content": _LONG_U}, {"role": "assistant", "content": _LONG_A}]


def _end_provider(writer):
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._delegation_enabled = True          # on_session_end no-ops without 002
    p._writer = writer
    p._agent_identity = "default"
    p._raw_identity = "default"
    p._session_id = "sess-end"
    p._turn_fingerprints = set()
    return p


def test_on_session_end_does_not_fingerprint_dropped_turns():
    writer = _RejectingWriter()
    p = _end_provider(writer)
    p.on_session_end(_MSGS)
    assert p._turn_fingerprints == set(), (
        "a turn the writer refused must stay un-fingerprinted, or the next "
        "on_session_end will skip a row that never reached Postgres"
    )


def test_on_session_end_retries_a_previously_dropped_turn():
    """End-to-end of the above: drop, then a later replay must re-offer it."""
    p = _end_provider(_RejectingWriter())
    p.on_session_end(_MSGS)

    accepting = _AcceptingWriter()
    p._writer = accepting
    p.on_session_end(_MSGS)

    turns = accepting.calls
    assert turns == 2, f"dropped turns must be retried on replay, got {turns}"
    assert len(p._turn_fingerprints) == 2


def test_on_session_end_still_skips_turns_that_were_accepted():
    """The dedup itself must survive the drop-safety change."""
    accepting = _AcceptingWriter()
    p = _end_provider(accepting)
    p.on_session_end(_MSGS)
    first = accepting.calls
    p.on_session_end(_MSGS)          # rotation replays the same list
    assert accepting.calls == first, "accepted turns must not be written twice"

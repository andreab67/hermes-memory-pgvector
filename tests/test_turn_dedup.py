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
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector import PgvectorMemoryProvider  # noqa: E402


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
    """Writer stand-in whose queue is always full (enqueue always drops)."""

    def __init__(self):
        self.calls = 0

    def enqueue(self, **kwargs) -> bool:
        self.calls += 1
        return False


class _AcceptingWriter:
    def __init__(self):
        self.calls = 0

    def enqueue(self, **kwargs) -> bool:
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

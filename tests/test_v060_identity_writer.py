"""v0.6.0 coverage for identity.py (M7, L3) and writer.py (L5, M3).

No DB or embed endpoint needed — identity.py has no I/O and AsyncWriter's
worker_fn here is a plain in-process stub, so these run everywhere.

M7: identities are case-folded in ``normalize_identity`` -- ``Marketing`` and
``marketing`` resolve to the same theme, alias keys/values and allow-list
entries are compared case-insensitively, and the canonical value returned is
always lowercase. See ``tests/test_identity.py`` for the pre-existing (now
phone-number-scrubbed, L1) baseline coverage this file adds to, not replaces.

L5/M3: ``AsyncWriter`` no longer carries a dead ``_stop`` Event (it was
constructed and cleared but never set by anything) and exposes a read-only
``draining`` property so a caller (the memory provider, in a later WP) can
skip slow work while a graceful shutdown drain is in progress.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector.identity import (  # noqa: E402
    BENCH_BUCKET,
    DEFAULT_IDENTITY,
    DM_BUCKET,
    GROUP_BUCKET,
    normalize_identity,
)
from hermes_pgvector.writer import AsyncWriter  # noqa: E402


# ---------------------------------------------------------------------------
# M7 -- case-folding of plain themes.
# ---------------------------------------------------------------------------

def test_plain_theme_is_lowercased():
    canon, normalized, reason = normalize_identity("Marketing")
    assert canon == "marketing"
    assert normalized is True
    assert reason == "lowercased"


@pytest.mark.parametrize("raw", ["Marketing", "MARKETING", "mArKeting"])
def test_mixed_case_variants_all_collapse_to_the_same_canonical(raw):
    canon, normalized, reason = normalize_identity(raw)
    assert canon == "marketing"
    assert normalized is True
    assert reason == "lowercased"


def test_already_lowercase_theme_reason_stays_unchanged():
    """Sanity check that 'lowercased' is only reported when case actually
    changed -- an already-lowercase theme keeps the pre-v0.6.0 'unchanged'
    reason, not a new 'lowercased' one."""
    canon, normalized, reason = normalize_identity("marketing")
    assert canon == "marketing"
    assert normalized is False
    assert reason == "unchanged"


@pytest.mark.parametrize(
    "raw", ["Marketing", "MARKETING", "Sales", "  Ops  ", "ops", "default"]
)
def test_normalized_flag_matches_canonical_vs_raw_invariant(raw):
    """The caller (``__init__.py``) relies on this: `normalized` must be True
    whenever the final canonical differs from the ORIGINAL raw input, in
    particular for a case-only change -- it decides whether raw_identity
    metadata is recorded."""
    canon, normalized, _ = normalize_identity(raw)
    assert normalized == (canon != raw)


# ---------------------------------------------------------------------------
# M7 -- alias keys and target values are case-folded.
# ---------------------------------------------------------------------------

def test_alias_key_matches_a_differently_cased_raw_identity():
    canon, normalized, reason = normalize_identity(
        "Agent-Hermes", aliases={"agent-hermes": "hermes"}
    )
    assert canon == "hermes"
    assert normalized is True
    assert reason == "alias"


def test_alias_key_configured_mixed_case_still_matches_lowercase_raw():
    canon, _, reason = normalize_identity(
        "agent-hermes", aliases={"Agent-Hermes": "hermes"}
    )
    assert canon == "hermes"
    assert reason == "alias"


def test_alias_target_value_is_lowercased_too():
    """An operator-configured alias target in mixed case must not leak
    mixed-case canonical identities back out."""
    canon, _, reason = normalize_identity(
        "agent-hermes", aliases={"agent-hermes": "HERMES"}
    )
    assert canon == "hermes"
    assert reason == "alias"


# ---------------------------------------------------------------------------
# M7 -- allow-list entries are case-folded (mixed-case allow-list, mixed-case
# identity, and every combination of the two).
# ---------------------------------------------------------------------------

def test_mixed_case_identity_matches_mixed_case_allowlist_entry():
    canon, normalized, reason = normalize_identity(
        "Marketing", allowed_themes=["Marketing", "Sales"]
    )
    assert canon == "marketing"
    assert normalized is True
    assert reason == "lowercased"


def test_uppercase_identity_matches_lowercase_allowlist_entry():
    canon, normalized, reason = normalize_identity(
        "SALES", allowed_themes=["marketing", "sales"]
    )
    assert canon == "sales"
    assert normalized is True
    assert reason == "lowercased"


def test_lowercase_identity_matches_uppercase_allowlist_entry():
    canon, normalized, reason = normalize_identity(
        "sales", allowed_themes=["Marketing", "SALES"]
    )
    assert canon == "sales"
    assert normalized is False
    assert reason == "unchanged"


def test_unknown_identity_under_mixed_case_allowlist_falls_back_to_default():
    canon, normalized, reason = normalize_identity(
        "Typo-Theme", allowed_themes=["Marketing", "Sales"]
    )
    assert canon == DEFAULT_IDENTITY
    assert normalized is True
    assert reason == "not-in-allowlist"


def test_governed_sink_typed_in_mixed_case_still_survives_a_strict_allowlist():
    """Isolation buckets stay always-allowed after case folding too -- typing
    the bucket name in the wrong case must not route PII/bench traffic to
    'default', where every theme could read it."""
    canon, normalized, reason = normalize_identity(
        "WHATSAPP-DM", allowed_themes=["marketing"]
    )
    assert canon == DM_BUCKET
    assert normalized is True
    assert reason == "lowercased"


# ---------------------------------------------------------------------------
# M7 -- DM/group/bench buckets are unaffected by case (the regexes already
# used re.IGNORECASE; this pins that the CANONICAL RETURN VALUE is also
# unaffected -- always the lowercase bucket constant, whatever case the raw
# session key arrived in).
# ---------------------------------------------------------------------------

def test_dm_bucket_matches_regardless_of_case():
    canon, normalized, reason = normalize_identity(
        "Agent:Main:WhatsApp:DM:15550100123"
    )
    assert canon == DM_BUCKET
    assert normalized is True
    assert reason == "dm-bucket"


def test_group_bucket_matches_regardless_of_case():
    canon, normalized, reason = normalize_identity(
        "Agent:Main:WhatsApp:Group:120363@G.us:15550100123"
    )
    assert canon == GROUP_BUCKET
    assert normalized is True
    assert reason == "group-bucket"


def test_bench_bucket_matches_regardless_of_case():
    canon, normalized, reason = normalize_identity("Skill-Bench")
    assert canon == BENCH_BUCKET
    assert normalized is True
    assert reason == "bench-bucket"


# ---------------------------------------------------------------------------
# L5/M3 -- AsyncWriter: no `_stop`, `draining` property.
# ---------------------------------------------------------------------------

def test_stop_event_is_gone_but_writer_still_drains_and_stops():
    """L5: `_stop` was constructed and cleared but never set by anything, so
    it never gated the drain loop. Confirm it is actually removed, and that
    the writer still drains every enqueued item and stops its thread on
    shutdown without it."""
    received: list = []

    def sink(item):
        received.append(item)

    writer = AsyncWriter(sink, maxsize=32)
    assert not hasattr(writer, "_stop")

    for i in range(10):
        assert writer.enqueue(
            action="turn", agent_identity="a", target="conversations", content=f"item-{i}"
        )

    writer.shutdown(timeout=5.0)

    assert len(received) == 10
    assert {item.content for item in received} == {f"item-{i}" for i in range(10)}
    assert writer.stats()["thread_alive"] is False


def test_draining_is_false_before_any_shutdown():
    writer = AsyncWriter(lambda item: None, maxsize=8)
    assert writer.draining is False
    assert writer.enqueue(action="turn", agent_identity="a", target="conversations", content="x")
    # Still false: a plain enqueue is not a drain.
    assert writer.draining is False
    writer.shutdown(timeout=5.0)


def test_draining_is_true_while_shutdown_drains_and_persists_after():
    """During: `draining` flips True as soon as shutdown() starts draining,
    while the worker is still blocked on an in-flight item. After: shutdown()
    has returned (the drain is complete) and the flag -- a plain reflection
    of `self._draining.is_set()`, per the M3 spec -- still reads True; a
    fresh enqueue is what resets it, exercised below."""
    release = threading.Event()
    first_item_seen = threading.Event()
    received: list = []

    def sink(item):
        if not first_item_seen.is_set():
            first_item_seen.set()
            release.wait(timeout=5.0)
        received.append(item)

    writer = AsyncWriter(sink, maxsize=8)
    assert writer.draining is False

    assert writer.enqueue(action="turn", agent_identity="a", target="conversations", content="item-0")
    first_item_seen.wait(timeout=5.0)  # worker is now blocked inside sink()
    assert writer.draining is False  # shutdown() has not been called yet

    shutdown_thread = threading.Thread(target=lambda: writer.shutdown(timeout=5.0))
    shutdown_thread.start()

    # shutdown() sets the drain flag synchronously before it does anything
    # else, so this should flip almost immediately -- poll with a deadline
    # rather than a fixed sleep so this can't be flaky under slow CI.
    deadline = time.monotonic() + 2.0
    while not writer.draining and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.draining is True  # during: worker still blocked, drain in progress

    release.set()  # let the worker finish so shutdown() can complete
    shutdown_thread.join(timeout=5.0)
    assert not shutdown_thread.is_alive()

    assert received  # sanity: the item was drained, not abandoned
    assert writer.draining is True  # after: shutdown() returned, flag persists

    # A fresh enqueue restarts the drain thread, which clears the flag.
    assert writer.enqueue(action="turn", agent_identity="a", target="conversations", content="item-1")
    assert writer.draining is False
    writer.shutdown(timeout=5.0)

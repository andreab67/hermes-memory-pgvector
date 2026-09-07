"""DB-gated tests for the read-side exclusion filter (v0.5.0).

Skip without PG_TEST_DSN. Point it at a THROWAWAY database. These tests only
touch rows under a random per-test prefix and clean them up.

Why these exist as LIVE tests. The gate's *policy* (which themes are restricted,
and when) is covered DB-free in tests/test_read_side_gate.py against a recording
store. What that cannot cover is whether the SQL is actually right:
`agent_identity <> ALL(%s)` bound with a Python list behaves differently in the
positional form (search / search_turns) and the named form
(hybrid_search / hybrid_search_turns, where it is spliced into BOTH the vector
and full-text CTEs of the RRF query). A filter that silently matched nothing --
or silently excluded everything -- would pass every DB-free test while leaking
in production, or while returning no rows at all.

So: real Postgres, real rows, real recall.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector.store import MemoryStore  # noqa: E402


FLAT = [0.1] * 768


@pytest.fixture
def store():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    s = MemoryStore(dsn)
    s.ensure_schema()
    prefix = "pytest-excl-" + os.urandom(4).hex()
    yield s, prefix
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity LIKE %s", (prefix + "%",))
            cur.execute("DELETE FROM conversations WHERE agent_identity LIKE %s", (prefix + "%",))
            conn.commit()


def _seed_entries(s, prefix):
    """One row per identity, all sharing a searchable token."""
    ok, sink_a, sink_b = f"{prefix}-ok", f"{prefix}-sinkA", f"{prefix}-sinkB"
    for ident in (ok, sink_a, sink_b):
        s.add(
            agent_identity=ident,
            target="memory",
            content=f"quarantine marker token for {ident}",
            embedding=FLAT,
        )
    return ok, sink_a, sink_b


def _seed_turns(s, prefix):
    ok, sink_a, sink_b = f"{prefix}-ok", f"{prefix}-sinkA", f"{prefix}-sinkB"
    for ident in (ok, sink_a, sink_b):
        s.append_turn(
            agent_identity=ident,
            session_id=f"sess-{ident}",
            role="user",
            content=f"quarantine marker token for {ident}",
            embedding=FLAT,
        )
    return ok, sink_a, sink_b


def _identities(rows):
    return {r.get("agent_identity") for r in rows}


# --- memory_entries --------------------------------------------------------

def test_search_excludes_listed_identities(store):
    s, prefix = store
    ok, sink_a, sink_b = _seed_entries(s, prefix)

    unfiltered = s.search(query_embedding=FLAT, limit=50)
    assert {ok, sink_a, sink_b} <= _identities(unfiltered), "seed rows must be visible without the filter"

    rows = s.search(query_embedding=FLAT, limit=50, exclude_identities=[sink_a, sink_b])
    found = _identities(rows)
    assert ok in found, "the exclusion must not drop everything"
    assert sink_a not in found and sink_b not in found


def test_hybrid_search_excludes_listed_identities(store):
    """The named-param form splices the filter into BOTH RRF legs; a leg that
    silently dropped it would still return the excluded row via fusion."""
    s, prefix = store
    ok, sink_a, sink_b = _seed_entries(s, prefix)

    unfiltered = s.hybrid_search(query_text="quarantine marker token", query_embedding=FLAT, limit=50)
    assert {ok, sink_a, sink_b} <= _identities(unfiltered)

    rows = s.hybrid_search(
        query_text="quarantine marker token",
        query_embedding=FLAT,
        limit=50,
        exclude_identities=[sink_a, sink_b],
    )
    found = _identities(rows)
    assert ok in found
    assert sink_a not in found and sink_b not in found


def test_hybrid_search_full_text_only_leg_still_excludes(store):
    """query_embedding=None is the degraded path taken when the query itself
    fails to embed. The exclusion must hold there too."""
    s, prefix = store
    ok, sink_a, sink_b = _seed_entries(s, prefix)

    rows = s.hybrid_search(
        query_text="quarantine marker token",
        query_embedding=None,
        limit=50,
        exclude_identities=[sink_a, sink_b],
    )
    found = _identities(rows)
    assert ok in found, "full-text-only recall must still work"
    assert sink_a not in found and sink_b not in found


# --- conversations ---------------------------------------------------------

def test_search_turns_excludes_listed_identities(store):
    s, prefix = store
    ok, sink_a, sink_b = _seed_turns(s, prefix)

    unfiltered = s.search_turns(query_embedding=FLAT, limit=50)
    assert {ok, sink_a, sink_b} <= _identities(unfiltered)

    rows = s.search_turns(query_embedding=FLAT, limit=50, exclude_identities=[sink_a, sink_b])
    found = _identities(rows)
    assert ok in found
    assert sink_a not in found and sink_b not in found


def test_hybrid_search_turns_excludes_listed_identities(store):
    s, prefix = store
    ok, sink_a, sink_b = _seed_turns(s, prefix)

    rows = s.hybrid_search_turns(
        query_text="quarantine marker token",
        query_embedding=FLAT,
        limit=50,
        exclude_identities=[sink_a, sink_b],
    )
    found = _identities(rows)
    assert ok in found
    assert sink_a not in found and sink_b not in found


# --- degenerate inputs -----------------------------------------------------

def test_empty_exclusion_list_is_a_no_op(store):
    """`exclude_identities=[]` must behave like None, not like 'exclude all'.
    The provider passes `restricted or None`, but the store is public API."""
    s, prefix = store
    ok, sink_a, sink_b = _seed_entries(s, prefix)

    rows = s.search(query_embedding=FLAT, limit=50, exclude_identities=[])
    assert {ok, sink_a, sink_b} <= _identities(rows)


def test_exclusion_of_an_absent_identity_changes_nothing(store):
    s, prefix = store
    ok, sink_a, sink_b = _seed_entries(s, prefix)

    rows = s.search(query_embedding=FLAT, limit=50, exclude_identities=["no-such-theme-anywhere"])
    assert {ok, sink_a, sink_b} <= _identities(rows)


def test_exclusion_survives_a_literal_with_sql_metacharacters(store):
    """The value is a bound parameter, not interpolated: a quote or percent in
    a theme name must neither error nor widen the match."""
    s, prefix = store
    ok, sink_a, _ = _seed_entries(s, prefix)

    rows = s.search(
        query_embedding=FLAT,
        limit=50,
        exclude_identities=["o'brien%", "%", sink_a],
    )
    found = _identities(rows)
    assert ok in found, "a literal % must not behave as a wildcard here"
    assert sink_a not in found

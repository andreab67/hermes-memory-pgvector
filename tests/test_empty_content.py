"""Empty-content rows must never be created, and never re-tried forever.

No DB required for the write-path half; the backfill half is live-mode.

Found in production. One row in `memory_entries` sat with `embedding IS NULL`
and `length(content) = 0` -- its metadata (`old_text`, `tool_name`) showing it
arrived through the built-in tool's `replace` path with empty new content.

Two separate defects met there:

  * nothing on the write path rejected empty content, so the row was created;
  * `backfill_null_embeddings` selected `WHERE embedding IS NULL` with no
    content filter, so every nightly sweep re-fetched it, called embed(), got
    EmbeddingError("empty input") -- which is unconditional for empty text --
    marked it `failed`, and moved on. Forever.

The second is the damaging one. It pins `failed` above zero and makes
`remaining == 0` unreachable, so "are there un-embedded rows?" stops being
answerable: you cannot tell one permanently-stuck row from one new genuine
failure. Skipping them silently would be just as bad, so they are reported
separately as `unembeddable`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector import PgvectorMemoryProvider  # noqa: E402
from hermes_pgvector.store import MemoryStore  # noqa: E402


class _RecordingWriter:
    def __init__(self):
        self.items = []

    def enqueue(self, **kwargs) -> bool:
        self.items.append(kwargs)
        return True


def _provider():
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._writer = _RecordingWriter()
    p._agent_identity = "marketing"
    p._raw_identity = "marketing"
    p._session_id = "sess-empty"
    return p


# --- write path -------------------------------------------------------------

@pytest.mark.parametrize("content", ["", "   ", "\n", "\t  \n "])
@pytest.mark.parametrize("action", ["add", "replace"])
def test_empty_add_or_replace_is_not_mirrored(action, content):
    p = _provider()
    p.on_memory_write(action=action, target="memory", content=content)
    assert p._writer.items == [], (
        f"{action!r} with {content!r} must not be mirrored -- it creates a row "
        "that can never be embedded and that backfill retries forever"
    )


def test_remove_with_empty_content_is_still_mirrored():
    """`remove` legitimately arrives with empty content -- the removal target
    travels in metadata as old_text, not in content -- so the empty-content
    guard must not swallow it.

    NOTE: an earlier version of this docstring said the row is targeted "via
    old_text" and left it there, which quietly asserted a path that was in fact
    broken downstream: _worker read item.content, not extra["old_text"], so the
    forwarded remove deleted EVERYTHING. See the _worker tests below -- this
    test only pins that the enqueue happens, never that it is well-formed."""
    p = _provider()
    p.on_memory_write(
        action="remove", target="memory", content="",
        metadata={"old_text": "the entry being removed"},
    )
    assert len(p._writer.items) == 1
    assert p._writer.items[0]["action"] == "remove"


def test_normal_add_is_unaffected():
    p = _provider()
    p.on_memory_write(action="add", target="memory", content="a real durable note")
    assert len(p._writer.items) == 1
    assert p._writer.items[0]["content"] == "a real durable note"


# --- backfill ---------------------------------------------------------------

@pytest.fixture
def store():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    s = MemoryStore(dsn)
    s.ensure_schema()
    agent = "pytest-empty-" + os.urandom(4).hex()
    yield s, agent
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity LIKE %s", (agent + "%",))
            conn.commit()


def _insert_raw(s, agent, content):
    """Bypass add() to plant the shape production actually had: NULL embedding
    plus empty content."""
    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory_entries (agent_identity, target, content, embedding, metadata)"
                " VALUES (%s, 'memory', %s, NULL, '{}'::jsonb) RETURNING id",
                (agent, content),
            )
            rid = cur.fetchone()[0]
            conn.commit()
    return rid


def test_backfill_skips_empty_content_and_reports_it(store):
    s, agent = store
    empty_id = _insert_raw(s, agent, "")
    real_id = _insert_raw(s, agent, "a genuinely embeddable durable note")

    seen = []

    def _embed_fn(text):
        seen.append(text)
        return [0.1] * 768

    report = s.backfill_null_embeddings(embed_fn=_embed_fn, tables=["memory_entries"])

    assert "" not in seen, "the empty row must never reach embed()"
    assert any("genuinely embeddable" in x for x in seen), "the real row must be embedded"
    assert report["memory_entries"]["unembeddable"] >= 1, "skipped rows must be reported, not hidden"

    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding IS NULL FROM memory_entries WHERE id = %s", (empty_id,))
            assert cur.fetchone()[0] is True, "empty row is left alone, not failed"
            cur.execute("SELECT embedding IS NULL FROM memory_entries WHERE id = %s", (real_id,))
            assert cur.fetchone()[0] is False, "real row got embedded"


def _scoped_backlog(s, agent):
    """`remaining`, computed for THIS test's rows only.

    backfill_null_embeddings reports table-wide numbers, but the fixture only
    cleans its own agent_identity prefix -- so asserting on the table-wide
    figure makes the test depend on every other row in the database and fail
    permanently after one interrupted run. Assert the same property, scoped."""
    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                r"SELECT count(*) FROM memory_entries "
                r"WHERE agent_identity = %s AND embedding IS NULL AND content ~ '\S'",
                (agent,),
            )
            return int(cur.fetchone()[0])


def test_backfill_remaining_can_reach_zero_despite_an_empty_row(store):
    """The point of the fix: an un-embeddable row must not pin the backlog
    above zero forever, or 'is the backlog clear?' becomes unanswerable."""
    s, agent = store
    _insert_raw(s, agent, "")
    _insert_raw(s, agent, "another embeddable note for the sweep")
    assert _scoped_backlog(s, agent) == 1, "one embeddable row to start"

    before = s.backfill_null_embeddings(
        embed_fn=lambda t: [0.1] * 768, tables=["memory_entries"], dry_run=True
    )["memory_entries"]["failed"]

    report = s.backfill_null_embeddings(
        embed_fn=lambda t: [0.1] * 768, tables=["memory_entries"]
    )

    assert _scoped_backlog(s, agent) == 0, (
        "the backlog must exclude un-embeddable rows so it can actually reach 0"
    )
    assert report["memory_entries"]["failed"] == before, (
        "an un-embeddable row must be skipped, not counted as a failure"
    )
    assert report["memory_entries"]["unembeddable"] >= 1


def test_backfill_skips_whitespace_only_content(store):
    """Not just the empty string. Postgres trim() strips SPACES ONLY, so a row
    holding a newline or tab used to pass the filter, reach embed(), and fail
    forever -- the same bug for a different shape. The predicate now matches
    Python's str.strip()."""
    s, agent = store
    for blank in ("", "   ", chr(10), chr(9) + chr(10) + " "):
        _insert_raw(s, agent, blank)
    real_id = _insert_raw(s, agent, "a real note alongside the blank ones")

    seen = []

    def _embed_fn(text):
        seen.append(text)
        return [0.1] * 768

    report = s.backfill_null_embeddings(embed_fn=_embed_fn, tables=["memory_entries"])

    assert all(x.strip() for x in seen), f"a blank row reached embed(): {seen!r}"
    assert report["memory_entries"]["unembeddable"] >= 4
    assert _scoped_backlog(s, agent) == 0

    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding IS NULL FROM memory_entries WHERE id = %s", (real_id,))
            assert cur.fetchone()[0] is False


# ---------------------------------------------------------------------------
# CRITICAL: the remove path must never be able to wipe a whole scope.
#
# store.remove() matches with `content LIKE %<old_text>%`. With an empty
# old_text that is LIKE '%%', which matches EVERY row -- so a remove that lost
# its target deletes the entire mirror for that (agent_identity, target)
# instead of one entry. Verified against Postgres: DELETE ... WHERE c LIKE '%%'
# removed all rows.
#
# _worker used to pass old_text=item.content. The built-in tool's remove op
# takes old_text and leaves content empty (tools/memory_tool.py:87), and the
# host forwards old_text through METADATA (memory_manager.notify_memory_tool_write),
# so item.content was ALWAYS '' for a remove -- making every mirrored
# `memory remove` a full wipe of that theme.
# ---------------------------------------------------------------------------

class _FakeStore:
    def __init__(self):
        self.removes = []

    def remove(self, *, agent_identity, target, old_text):
        self.removes.append(old_text)
        return 1


def _worker_provider():
    p = PgvectorMemoryProvider()
    p._healthy = True
    p._store = _FakeStore()
    return p


class _Item:
    def __init__(self, content="", extra=None):
        self.action = "remove"
        self.agent_identity = "marketing"
        self.target = "memory"
        self.content = content
        self.extra = extra or {}
        self.metadata = {}


def test_worker_remove_uses_old_text_not_content():
    """The regression: content is empty for every real remove, so reading it
    produced LIKE '%%' and deleted the whole scope."""
    p = _worker_provider()
    p._worker(_Item(content="", extra={"old_text": "the entry to delete"}))
    assert p._store.removes == ["the entry to delete"]


def test_worker_remove_refuses_when_no_target_is_available():
    """Belt and braces: with neither content nor old_text there is nothing to
    match, and the store must not be called at all."""
    for item in (_Item(content="", extra={}), _Item(content="   ", extra={"old_text": "  "})):
        p = _worker_provider()
        p._worker(item)
        assert p._store.removes == [], "a target-less remove must never reach the store"


def test_worker_remove_falls_back_to_content_when_old_text_absent():
    """Older/other callers that put the target in content still work."""
    p = _worker_provider()
    p._worker(_Item(content="target in content", extra={}))
    assert p._store.removes == ["target in content"]


def test_store_remove_rejects_empty_old_text():
    """Store-level guard, independent of any caller: an empty pattern is
    LIKE '%%' and must be impossible to reach by omission."""
    s = MemoryStore.__new__(MemoryStore)   # no connection needed; guard is first
    for bad in ("", "   ", chr(10), chr(9) + " "):
        with pytest.raises(ValueError, match="non-empty old_text"):
            s.remove(agent_identity="a", target="memory", old_text=bad)


def test_store_remove_guard_runs_before_any_db_work(store):
    """Live: the guard must fire before touching the pool, and delete nothing."""
    s, agent = store
    _insert_raw(s, agent, "row one that must survive")
    _insert_raw(s, agent, "row two that must survive")

    with pytest.raises(ValueError):
        s.remove(agent_identity=agent, target="memory", old_text="")

    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM memory_entries WHERE agent_identity = %s", (agent,))
            assert cur.fetchone()[0] == 2, "an empty remove must delete nothing"


def test_store_remove_still_deletes_a_real_match(store):
    s, agent = store
    _insert_raw(s, agent, "delete me please")
    _insert_raw(s, agent, "keep me around")
    n = s.remove(agent_identity=agent, target="memory", old_text="delete me")
    assert n == 1
    with s._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM memory_entries WHERE agent_identity = %s", (agent,))
            assert [r[0] for r in cur.fetchall()] == ["keep me around"]

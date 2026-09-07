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
    """`remove` legitimately arrives with empty content: it targets the row via
    old_text, so the empty-content guard must not swallow it."""
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


def test_backfill_remaining_can_reach_zero_despite_an_empty_row(store):
    """The point of the fix: an un-embeddable row must not pin `remaining`
    above zero forever, or 'is the backlog clear?' becomes unanswerable."""
    s, agent = store
    _insert_raw(s, agent, "")
    _insert_raw(s, agent, "another embeddable note for the sweep")

    report = s.backfill_null_embeddings(
        embed_fn=lambda t: [0.1] * 768, tables=["memory_entries"]
    )
    assert report["memory_entries"]["remaining"] == 0, (
        "remaining must exclude un-embeddable rows so it can actually reach 0"
    )
    assert report["memory_entries"]["failed"] == 0, (
        "an un-embeddable row must be skipped, not counted as a failure"
    )

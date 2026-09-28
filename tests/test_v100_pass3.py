"""1.0 review pass-3 fixes: provider layer (P3PROV-1, P3PROV-2). DB-free."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hermes_pgvector as pkg  # noqa: E402
from hermes_pgvector import PgvectorMemoryProvider  # noqa: E402
from hermes_pgvector.identity import GROUP_BUCKET  # noqa: E402
from hermes_pgvector.writer import _PendingWrite  # noqa: E402


class _UnhealthyStore:
    def __init__(self, dsn):
        self.dsn = dsn

    def ensure_schema(self):
        return None

    def health(self):
        return {"ok": False, "error": "fake", "row_count_estimate": None}

    def close(self):
        return None


class _AddStore:
    def __init__(self):
        self.rows = []

    def add(self, **kwargs):
        self.rows.append(kwargs)
        return 1


def test_forum_session_key_resolves_to_group_bucket_and_is_restricted(monkeypatch):
    monkeypatch.setattr(pkg, "MemoryStore", _UnhealthyStore)
    p = PgvectorMemoryProvider(config={"bulk_sync_on_init": False})
    p.initialize(
        "s1",
        gateway_session_key="agent:main:telegram:forum:-1001234567890:42:987654321",
        agent_identity="default",
        agent_workspace="hermes",
    )
    assert p._agent_identity == GROUP_BUCKET
    assert "-1001234567890" not in p._agent_identity
    assert GROUP_BUCKET not in p._restricted_identities()  # the bucket itself may read itself
    q = PgvectorMemoryProvider(config={"bulk_sync_on_init": False})
    q.initialize("s2", agent_identity="marketing")
    assert GROUP_BUCKET in q._restricted_identities()
    p.shutdown()
    q.shutdown()


def _add_item():
    return _PendingWrite(action="add", agent_identity="x", target="memory",
                         content="hello world", extra={}, metadata={})


def test_worker_add_survives_blank_embed_url():
    p = PgvectorMemoryProvider(config={"embed_url": None, "embed_model": None,
                                       "embed_write_retries": 0})
    p._store = _AddStore()
    p._worker(_add_item())
    assert len(p._store.rows) == 1
    assert p._store.rows[0]["embedding"] is None


def test_maybe_embed_treats_any_exception_as_no_embedding(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("embed-side bug")

    monkeypatch.setattr(pkg, "_embed_with_config", boom)
    p = PgvectorMemoryProvider(config={})
    assert p._maybe_embed("some content") is None

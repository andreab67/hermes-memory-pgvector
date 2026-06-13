"""tests/test_mcp_server.py — tests for the MCP tool surface.

Forked from andreab67/hermes-hexus (BSD-3-Clause).

Two layers:
  1. **Tool-function tests** — exercise each pure function in
     mcp_server.tools directly with a live `MemoryStore`. Cheap, fast,
     catches the meat of the contract.
  2. **FastMCP wiring tests** — verify that every tool is registered
     with the right name + input schema. Runs in-process against
     `FastMCP._tool_cache`, so it doesn't need to spin up a real stdio
     transport or a streamable-http server.

The whole file is skipped if `PG_TEST_DSN` is unset (no DB available).
The tool layer is also skipped if `mcp` isn't installed (the test
container is expected to have it via the `[mcp]` extra in the
Dockerfile's pip install).
"""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid

import pytest

# Skip the whole module if there's no DSN to talk to.
pytestmark = pytest.mark.skipif(
    not os.environ.get("PG_TEST_DSN"),
    reason="PG_TEST_DSN not set — live DB tests skipped",
)

# Skip just the wiring layer if mcp isn't importable. The pure-function
# layer below doesn't need mcp at all.
mcp_required = pytest.mark.skipif(
    True,  # placeholder; replaced below after import attempt
    reason="mcp SDK not importable",
)

mcp_available = True
try:
    import mcp  # noqa: F401
except ImportError:
    mcp_available = False


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def store():
    """A fresh MemoryStore per test, pointing at the live DB.

    Yields the store with a per-test unique agent_identity stamped on
    it (`store._test_agent`) so retain/recall don't see other tests' rows.
    Cleans up its own rows on teardown.
    """
    from hexus.store import MemoryStore

    dsn = os.environ["PG_TEST_DSN"]
    s = MemoryStore(dsn)
    s.ensure_schema()
    s._test_agent = f"mcp-test-{uuid.uuid4().hex[:8]}"  # type: ignore[attr-defined]
    yield s
    # Best-effort cleanup of any rows this test wrote.
    try:
        with s._get_pool().connection() as conn:  # noqa: SLF001
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memory_entries WHERE agent_identity = %s",
                    (s._test_agent,),  # type: ignore[attr-defined]
                )
                cur.execute(
                    "DELETE FROM conversations WHERE agent_identity = %s",
                    (s._test_agent,),  # type: ignore[attr-defined]
                )
            conn.commit()
    finally:
        s.close()


def agent_of(store) -> str:
    """Return the per-test unique agent_identity attached to a store."""
    return store._test_agent  # type: ignore[attr-defined]


# -----------------------------------------------------------------------
# Layer 1 — pure function tests
# -----------------------------------------------------------------------


def test_default_agent_identity_resolution(monkeypatch):
    """`default_agent_identity()` reads env, falls back to 'default'."""
    from mcp_server import tools

    monkeypatch.delenv("HEXUS_AGENT_IDENTITY", raising=False)
    assert tools.default_agent_identity() == "default"
    monkeypatch.setenv("HEXUS_AGENT_IDENTITY", "intraday-trading")
    assert tools.default_agent_identity() == "intraday-trading"


def test_memory_health_reports_ok(store):
    from mcp_server import tools

    out = tools.memory_health(store, {})
    assert out["status"] == "ok"
    assert out["schema_ok"] is True
    assert out["embedder"]["dim"] == 384
    assert "all-MiniLM-L6-v2" in out["embedder"]["model"]


def test_memory_retain_inserts_rows(store):
    from mcp_server import tools

    out = tools.memory_retain(
        store,
        {
            "contents": ["alpha bravo", "charlie delta"],
            "target": "memory",
            "agent_identity": agent_of(store),  # noqa: SLF001
            "metadata": {"doc_type": "document", "source_url": "https://example.com/a"},
        },
    )
    assert out["inserted"] == 2
    assert out["duplicates"] == 0
    assert out["errors"] == []


def test_memory_retain_dedupes_on_exact_repeat(store):
    from mcp_server import tools

    args = {
        "contents": ["unique content one"],
        "target": "memory",
        "agent_identity": agent_of(store),  # noqa: SLF001
    }
    first = tools.memory_retain(store, args)
    second = tools.memory_retain(store, args)
    assert first["inserted"] == 1
    assert second["inserted"] == 0
    assert second["duplicates"] == 1


def test_memory_retain_rejects_empty_contents(store):
    from mcp_server import tools

    with pytest.raises(ValueError, match="contents must be a non-empty list"):
        tools.memory_retain(store, {"contents": []})
    with pytest.raises(ValueError, match="non-empty string"):
        tools.memory_retain(store, {"contents": ["", "valid"]})


def test_memory_retain_rejects_bad_target(store):
    from mcp_server import tools

    with pytest.raises(ValueError, match="target must be"):
        tools.memory_retain(
            store,
            {
                "contents": ["x"],
                "target": "bogus",
                "agent_identity": agent_of(store),  # noqa: SLF001
            },
        )


def test_empty_string_target_defaults_to_both_stores(store):
    """MCP clients often send default string fields as '' rather than
    omitting them. Empty target should mean 'both memory and user'."""
    from mcp_server import tools

    tools.memory_retain(
        store,
        {
            "contents": ["Postgres + hexus is great for semantic search."],
            "target": "memory",
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )

    out = tools.memory_recall(
        store,
        {
            "query": "hexus semantic search",
            "top_k": 3,
            "agent_identity": agent_of(store),  # noqa: SLF001
            "target": "",
        },
    )
    assert out["count"] == 1
    assert "hexus" in out["results"][0]["content"]

    count = tools.memory_count(
        store,
        {
            "agent_identity": agent_of(store),  # noqa: SLF001
            "target": "",
        },
    )
    assert count["memory_entries"] == 1


def test_memory_recall_round_trip(store):
    """Retain 3 related docs, recall with a related query, expect top hit
    to be one of the 3."""
    from mcp_server import tools

    docs = [
        "Postgres + hexus is great for semantic search.",
        "The weather in Austin is warm in June.",
        "Sentence-transformers provides local BERT embeddings.",
    ]
    tools.memory_retain(
        store,
        {
            "contents": docs,
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )

    out = tools.memory_recall(
        store,
        {
            "query": "what does hexus do",
            "top_k": 3,
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert out["count"] == 3
    # The first hit should be the hexus doc.
    assert "hexus" in out["results"][0]["content"]
    # All results carry a score in [0, 1]. Note: cosine distance can be
    # slightly > 1 for unnormalized vectors, making 1 - distance slightly
    # negative. We clamp at a small negative tolerance.
    for r in out["results"]:
        assert -0.01 <= r["score"] <= 1.0


def test_memory_hybrid_search_round_trip(store):
    from mcp_server import tools

    docs = [
        "Postgres + hexus is great for semantic search.",
        "The weather in Austin is warm in June.",
        "Sentence-transformers provides local BERT embeddings.",
    ]
    tools.memory_retain(
        store,
        {
            "contents": docs,
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )

    out = tools.memory_hybrid_search(
        store,
        {
            "query": "weather in Austin",
            "top_k": 3,
            "vector_weight": 0.5,
            "text_weight": 0.5,
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert out["count"] >= 1
    assert "Austin" in out["results"][0]["content"]
    assert "vector_score" in out["results"][0]
    assert "text_score" in out["results"][0]
    assert out["results"][0]["text_score"] > 0.0


def test_memory_hybrid_recall_turns_round_trip(store):
    from mcp_server import tools

    tools.memory_append_turn(
        store,
        {
            "session_id": "session-123",
            "role": "user",
            "content": "My favorite database is Postgres.",
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    tools.memory_append_turn(
        store,
        {
            "session_id": "session-123",
            "role": "assistant",
            "content": "I prefer local BERT embeddings.",
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )

    out = tools.memory_hybrid_recall_turns(
        store,
        {
            "query": "favorite database Postgres",
            "top_k": 2,
            "vector_weight": 0.5,
            "text_weight": 0.5,
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert out["count"] >= 1
    assert "Postgres" in out["results"][0]["content"]
    assert "vector_score" in out["results"][0]
    assert "text_score" in out["results"][0]
    assert out["results"][0]["text_score"] > 0.0


def test_memory_recall_respects_min_similarity(store):
    from mcp_server import tools

    tools.memory_retain(
        store,
        {
            "contents": ["Postgres is a relational database."],
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    out = tools.memory_recall(
        store,
        {
            "query": "Postgres relational database",
            "top_k": 5,
            "agent_identity": agent_of(store),  # noqa: SLF001
            "min_similarity": 0.5,
        },
    )
    # Top hit should still be present (real BERT score should be high).
    assert out["count"] >= 1


def test_memory_recall_caps_top_k(store):
    from mcp_server import tools

    tools.memory_retain(
        store,
        {
            "contents": [f"entry {i}" for i in range(10)],
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    out = tools.memory_recall(
        store,
        {
            "query": "entry",
            "top_k": 10000,  # way over the cap
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    # Capped at 100, but we only have 10 rows so count is 10.
    assert out["count"] == 10


def test_memory_recall_rejects_empty_query(store):
    from mcp_server import tools

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        tools.memory_recall(store, {"query": ""})


def test_memory_recall_cross_agent_returns_other_agents_rows(store):
    """agent_identity=None on recall searches the whole store — including
    rows written by other agents. This is the multi-agent read path."""
    from mcp_server import tools

    # Write a row with this test's agent.
    tools.memory_retain(
        store,
        {
            "contents": ["the meaning of life is forty two"],
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    # Recall with agent_identity="" (None) — should find it.
    out = tools.memory_recall(
        store,
        {"query": "meaning of life", "top_k": 5, "agent_identity": ""},
    )
    assert out["count"] >= 1
    assert any("forty two" in r["content"] for r in out["results"])


def test_memory_search_browse(store):
    from mcp_server import tools

    tools.memory_retain(
        store,
        {
            "contents": [f"row {i}" for i in range(5)],
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    out = tools.memory_search(
        store,
        {
            "agent_identity": agent_of(store),  # noqa: SLF001
            "limit": 3,
        },
    )
    assert out["count"] == 3
    assert out["limit"] == 3
    assert out["offset"] == 0
    # No embedding field — we strip it for compact JSON.
    for r in out["rows"]:
        assert "embedding" not in r

    # Test pagination / offset
    out_page2 = tools.memory_search(
        store,
        {
            "agent_identity": agent_of(store),  # noqa: SLF001
            "limit": 2,
            "offset": 3,
        },
    )
    assert out_page2["count"] == 2
    assert out_page2["limit"] == 2
    assert out_page2["offset"] == 3
    contents = [r["content"] for r in out_page2["rows"]]
    assert contents == ["row 1", "row 0"]


def test_memory_forget_dry_run_by_default(store):
    from mcp_server import tools

    out = tools.memory_retain(
        store,
        {
            "contents": ["to be deleted"],
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert out["inserted"] == 1
    # Look up the row id.
    listed = store.list_entries(agent_identity=agent_of(store), limit=1)  # noqa: SLF001
    row_id = listed[0]["id"]

    dry = tools.memory_forget(
        store,
        {"id": row_id, "agent_identity": agent_of(store), "confirm": False},  # noqa: SLF001
    )
    assert dry["dry_run"] is True
    assert dry["deleted"] == 0
    # Row is still there.
    after = store.list_entries(agent_identity=agent_of(store), limit=10)  # noqa: SLF001
    assert any(r["id"] == row_id for r in after)


def test_memory_forget_actually_deletes_with_confirm(store):
    from mcp_server import tools

    out = tools.memory_retain(
        store,
        {
            "contents": ["goodbye cruel world"],
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert out["inserted"] == 1
    row_id = store.list_entries(agent_identity=agent_of(store), limit=1)[0]["id"]  # noqa: SLF001

    real = tools.memory_forget(
        store,
        {"id": row_id, "agent_identity": agent_of(store), "confirm": True},  # noqa: SLF001
    )
    assert real["dry_run"] is False
    assert real["deleted"] == 1
    # And the row is gone.
    after = store.list_entries(agent_identity=agent_of(store), limit=10)  # noqa: SLF001
    assert not any(r["id"] == row_id for r in after)


def test_memory_forget_rejects_bad_id(store):
    from mcp_server import tools

    with pytest.raises(ValueError, match="positive integer"):
        tools.memory_forget(store, {"id": 0, "agent_identity": "x", "confirm": True})
    with pytest.raises(ValueError, match="positive integer"):
        tools.memory_forget(store, {"id": -3, "agent_identity": "x", "confirm": True})
    with pytest.raises(ValueError, match="positive integer"):
        tools.memory_forget(store, {"id": "abc", "agent_identity": "x", "confirm": True})


def test_memory_append_turn_and_recall_turns(store):
    from mcp_server import tools

    out = tools.memory_append_turn(
        store,
        {
            "session_id": "sess-abc",
            "role": "user",
            "content": "I love using Postgres for memory",
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert isinstance(out["id"], int) and out["id"] > 0
    out2 = tools.memory_append_turn(
        store,
        {
            "session_id": "sess-abc",
            "role": "assistant",
            "content": "Glad to hear! hexus is a great fit.",
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert out2["id"] > out["id"]

    recall = tools.memory_recall_turns(
        store,
        {
            "query": "Postgres memory",
            "top_k": 5,
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    assert recall["count"] >= 1
    roles = {r["role"] for r in recall["results"]}
    assert "user" in roles and "assistant" in roles


def test_memory_append_turn_validates_role(store):
    from mcp_server import tools

    with pytest.raises(ValueError, match="role must be one of"):
        tools.memory_append_turn(
            store,
            {
                "session_id": "s1",
                "role": "pirate",
                "content": "arrr",
                "agent_identity": agent_of(store),  # noqa: SLF001
            },
        )


def test_memory_count_scopes_correctly(store):
    from mcp_server import tools

    tools.memory_retain(
        store,
        {
            "contents": [f"row {i}" for i in range(3)],
            "agent_identity": agent_of(store),  # noqa: SLF001
        },
    )
    out = tools.memory_count(store, {"agent_identity": agent_of(store)})
    assert out["memory_entries"] == 3
    assert out["conversations"] == 0


# -----------------------------------------------------------------------
# Multi-agent isolation — the headline guarantee
# -----------------------------------------------------------------------


def test_multi_agent_isolation(store):
    """Two agents write to the same store, neither can see the other's
    scoped rows by default. With agent_identity="" on recall, both
    agents' rows are returned."""
    from mcp_server import tools

    agent_a = f"{agent_of(store)}-A"
    agent_b = f"{agent_of(store)}-B"

    # Write distinct rows for each agent.
    tools.memory_retain(
        store,
        {"contents": ["alpha's secret"], "agent_identity": agent_a},
    )
    tools.memory_retain(
        store,
        {"contents": ["beta's secret"], "agent_identity": agent_b},
    )

    # Agent A only sees alpha's row.
    out_a = tools.memory_search(store, {"agent_identity": agent_a, "limit": 50})
    contents_a = {r["content"] for r in out_a["rows"]}
    assert "alpha's secret" in contents_a
    assert "beta's secret" not in contents_a

    # Agent B only sees beta's row.
    out_b = tools.memory_search(store, {"agent_identity": agent_b, "limit": 50})
    contents_b = {r["content"] for r in out_b["rows"]}
    assert "beta's secret" in contents_b
    assert "alpha's secret" not in contents_b

    # Cross-agent recall (agent_identity="" → search all) sees both.
    out_all = tools.memory_recall(
        store,
        {
            "query": "secret",
            "top_k": 50,
            "agent_identity": "",
        },
    )
    seen_agents = {r["agent_identity"] for r in out_all["results"]}
    assert agent_a in seen_agents
    assert agent_b in seen_agents

    # Cleanup.
    for a in (agent_a, agent_b):
        with store._get_pool().connection() as conn:  # noqa: SLF001
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memory_entries WHERE agent_identity = %s", (a,)
                )
            conn.commit()


# -----------------------------------------------------------------------
# Layer 2 — FastMCP wiring (skipped if mcp SDK isn't installed)
# -----------------------------------------------------------------------


@pytest.mark.skipif(not mcp_available, reason="mcp SDK not importable")
class TestMcpWiring:
    """Verify the FastMCP instance registers every tool with the right
    name + a non-empty inputSchema. We don't run the stdio transport
    in tests (would require a child process and a wire-protocol client)
    — we just check the registration surface."""

    def _server(self, store):
        from mcp_server.server import _build_server
        return _build_server(store, name="hexus-test")

    def _get_tool_cache(self, mcp):
        """FastMCP wraps a Server internally; tools are cached there.
        
        FastMCP 1.x stores tools in different locations depending on version:
        - mcp._tool_manager._tools (newer)
        - mcp._server._tool_cache (older)
        - mcp.get_tools() (public API in some versions)
        """
        # Try the newer FastMCP 1.x _tool_manager
        if hasattr(mcp, "_tool_manager") and hasattr(mcp._tool_manager, "_tools"):
            return mcp._tool_manager._tools
        # Try the older _server attribute
        if hasattr(mcp, "_server") and hasattr(mcp._server, "_tool_cache"):
            return mcp._server._tool_cache
        # Try the public get_tools() method if available
        if hasattr(mcp, "get_tools"):
            tools = mcp.get_tools()
            if isinstance(tools, dict):
                return tools
            # If it's a list, convert to dict
            return {t.name: t for t in tools}
        raise AttributeError("Cannot find tool cache in FastMCP instance")

    def test_all_expected_tools_are_registered(self, store):
        mcp = self._server(store)
        cache = self._get_tool_cache(mcp)
        names = set(cache.keys())
        # The ten tools the server is supposed to expose.
        assert {
            "memory_health",
            "memory_retain",
            "memory_recall",
            "memory_search",
            "memory_forget",
            "memory_recall_turns",
            "memory_append_turn",
            "memory_count",
            "memory_hybrid_search",
            "memory_hybrid_recall_turns",
        }.issubset(names), f"missing tools: {names - {n for n in names if n.startswith('memory_')}}"

    def test_every_tool_has_description_and_input_schema(self, store):
        mcp = self._server(store)
        cache = self._get_tool_cache(mcp)
        for name, tool in cache.items():
            assert tool.description, f"tool {name!r} has empty description"
            # FastMCP's Tool uses `parameters` for the input JSON schema
            assert hasattr(tool, "parameters"), f"tool {name!r} has no parameters (input schema)"
            params = tool.parameters
            assert params, f"tool {name!r} has empty parameters"
            # MCP-required keys
            assert params.get("type") == "object", (
                f"tool {name!r} parameters.type is {params.get('type')!r}"
            )
            assert "properties" in params, (
                f"tool {name!r} parameters has no properties"
            )

    def test_memory_retain_schema_includes_contents(self, store):
        mcp = self._server(store)
        tool = self._get_tool_cache(mcp)["memory_retain"]
        props = tool.parameters["properties"]
        assert "contents" in props
        assert props["contents"]["type"] == "array"
        # contents is required.
        assert "contents" in tool.parameters.get("required", [])

    def test_memory_recall_schema_includes_query(self, store):
        mcp = self._server(store)
        tool = self._get_tool_cache(mcp)["memory_recall"]
        props = tool.parameters["properties"]
        assert "query" in props
        assert "top_k" in props
        assert "agent_identity" in props

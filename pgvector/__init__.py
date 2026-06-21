"""pgvector — Postgres + pgvector memory provider for hermes-agent.

Mirrors hermes-agent's built-in `memory` tool entries (MEMORY.md / USER.md
in tools/memory_tool.py) into a single Postgres table, adds 768-dim
embeddings for semantic recall, and scopes by `agent_identity` so each
named agent (marketing / sales / trading / incident / …) has its own
theme.

Design philosophy: this is a STORAGE LAYER for hermes-agent's native
memory model, not a new memory model. We don't invent facts, entities,
trust scores, deriver pipelines, or dialectic synthesis. We give the
built-in `memory` tool a durable Postgres backing + semantic search,
nothing more. Honcho went heavy and exploded; this stays lean.

Config in $HERMES_HOME/config.yaml under plugins.pgvector:

    plugins:
      pgvector:
        dsn:        "dbname=hermes_memory user=hermes host=/var/run/postgresql"
        embed_url:  "http://192.168.100.50:11434"
        embed_model: "nomic-embed-text"
        prefetch_limit: 5
        min_similarity: 0.30
        embed_on_write: true
        scope_default: "current"   # 'current' | 'all'

Tools exposed: `recall_memory` (one explicit search tool). All built-in
memory writes (add/replace/remove) are mirrored automatically via the
on_memory_write hook — no agent-facing change.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
    from hermes_cli.config import cfg_get
except ImportError:  # pragma: no cover
    # Standalone context (the `python -m pgvector` CLI / unit tests) — hermes-agent
    # is not importable. Provide minimal fallbacks so the package still imports;
    # the provider class is never instantiated outside the agent, only the store/
    # embed/identity helpers + the maintenance CLI are used here.
    MemoryProvider = object  # type: ignore[assignment,misc]

    def tool_error(msg: str) -> str:  # type: ignore[misc]
        return json.dumps({"error": msg})

    def cfg_get(data, *keys, default=None):  # type: ignore[misc]
        cur = data
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
            if cur is None:
                return default
        return cur

from .embed import embed, EmbeddingError
from .identity import classify_kind, normalize_identity
from .store import MemoryStore
from .writer import AsyncWriter, _PendingWrite


# Boilerplate / acknowledgement-only turns that are not worth embedding or
# storing. Case-insensitive whole-string match after strip. Combined with
# a length floor (default 40 chars) in _is_noise.
_NOISE_RE = re.compile(
    r"^("
    r"ok(ay)?|thanks?( you)?|thx|ty|np|"
    r"yes|no|sure|got it|done|cool|nice|great|"
    r"continue|please|exit|cancel|stop|quit|"
    r"yeah|yep|nope|alright"
    r")[\s\.\!\?]*$",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schema — one explicit search over memory_entries
# ---------------------------------------------------------------------------

RECALL_CONVERSATION_SCHEMA = {
    "name": "recall_conversation",
    "description": (
        "Semantic search over past chat turns (every substantive "
        "user/assistant exchange across all sessions). Use this when "
        "the user references something you discussed earlier — last week, "
        "yesterday, in another session — and you need the actual turn "
        "text, not just a durable memory entry. Returns top-K matching "
        "turns with role, content, session_id, and timestamp.\n\n"
        "SCOPES: 'current' (your theme — default), 'session' (current "
        "session only), 'all' (every theme).\n\n"
        "Skip for in-session continuity (already in your context). Skip "
        "for durable facts (use recall_memory instead — that's the "
        "MEMORY.md / USER.md entries the agent decided to remember)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text query describing what to recall.",
            },
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'session', 'all', or a named agent.",
                "default": "current",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


RECALL_MEMORY_SCHEMA = {
    "name": "recall_memory",
    "description": (
        "Semantic search over durable memory entries (the same entries the "
        "built-in `memory` tool writes to MEMORY.md / USER.md, stored "
        "durably in Postgres with embeddings).\n\n"
        "WHEN TO USE: when the answer might be in a past memory entry that "
        "is NOT already in your system prompt's memory block — older "
        "entries, or entries from another named agent. The current scope's "
        "recent entries are already injected ambient; only use this tool "
        "for deeper / cross-scope recall.\n\n"
        "SCOPES:\n"
        "  'current' — your own theme (default; e.g. 'marketing')\n"
        "  'all'     — across all agent themes\n"
        "  '<name>'  — a specific theme: 'marketing', 'sales', 'trading', 'incident', …"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text query describing what to recall.",
            },
            "scope": {
                "type": "string",
                "description": "Theme scope: 'current', 'all', or a named agent.",
                "default": "current",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user", "both"],
                "description": "Which store to search. Default 'both'.",
                "default": "both",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (1-20, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "dsn": "dbname=hermes_memory user=hermes host=/var/run/postgresql connect_timeout=5",
    "embed_url": "http://192.168.100.50:11434",
    "embed_model": "nomic-embed-text",
    "prefetch_limit": 5,
    "min_similarity": 0.30,
    "embed_on_write": True,
    "scope_default": "current",
    "write_queue_maxsize": 256,
    # v0.1.1 — bulk sync MEMORY.md / USER.md on init
    "bulk_sync_on_init": True,
    # v0.2 — conversation turn capture
    "sync_turns": True,
    "turn_min_chars": 40,  # turns shorter than this are noise unless > 200 chars or contain tool refs
    # v0.4 — identity governance
    "allowed_themes": None,        # None/empty = governance off; set a list to enforce an allow-list
    "identity_aliases": {},        # {raw: canonical} remaps applied before normalization
    "bench_mode": "bucket",        # 'bucket' -> _bench (isolated) | 'reject' -> default
    # v0.4 — conversation embedding policy + TTL
    "conversation_embed_policy": "all",   # all | substantive_only | none
    "ttl_days": 0,                 # 0 = TTL off; prune is CLI-only regardless of this value
    # v0.4 — writer-path embed retries (hot path stays single-attempt)
    "embed_write_retries": 2,
    "embed_write_backoff": 0.1,
}


def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        import yaml
        with open(config_path, encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh) or {}
        return cfg_get(data, "plugins", "pgvector", default={}) or {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class PgvectorMemoryProvider(MemoryProvider):
    """Postgres mirror of built-in memory entries, with semantic recall."""

    def __init__(self, config: dict | None = None):
        self._config = {**DEFAULTS, **(config or {})}
        self._store: Optional[MemoryStore] = None
        self._writer: Optional[AsyncWriter] = None
        self._agent_identity: str = "default"
        self._raw_identity: str = "default"          # pre-normalization (for metadata/trace)
        self._parent_session_id: Optional[str] = None
        self._session_id: str = ""
        self._healthy: bool = False
        self._delegation_enabled: bool = False        # set in initialize() iff migration 002 applied
        self._embed_warned: bool = False

    @property
    def name(self) -> str:
        return "pgvector"

    # -- Lifecycle -----------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import psycopg  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        # Per-agent theme scoping — priority order:
        #   1. gateway_session_key — from the `X-Hermes-Session-Key` header on
        #      API requests. This is the EXPLICIT minion-scope signal sent by
        #      systemd-run callers (marketing-daily, sales-daily, intraday
        #      workers, …) and takes precedence over the profile fallback
        #      because the gateway always sets agent_identity='default' for
        #      API traffic — without prioritising the header, every minion
        #      collapses to one shared 'default' scope.
        #   2. agent_identity ≠ 'default' — explicit profile name from CLI
        #      (`hermes --profile marketing`). Skipped when it's the
        #      auto-default sentinel to allow header (#1) to win.
        #   3. agent_workspace — shared workspace name from some platforms.
        #   4. agent_identity == 'default' — accept it now (no other source).
        #   5. 'default'        — last-resort bucket for unscoped traffic.
        explicit_identity = kwargs.get("agent_identity")
        if explicit_identity == "default":
            explicit_identity = None  # sentinel — let header take over
        self._agent_identity = (
            kwargs.get("gateway_session_key")
            or explicit_identity
            or kwargs.get("agent_workspace")
            or kwargs.get("agent_identity")  # accept 'default' if nothing else set
            or "default"
        )

        # v0.4 identity governance — normalize the resolved identity ONCE, here,
        # AFTER the priority chain (so a typo'd header still reaches the allow-list)
        # and never at read time (rows keep their historical identity, else they
        # become unrecallable). Strips PII/cardinality DM keys, isolates bench
        # traffic, and enforces the optional allow-list. See identity.py.
        self._raw_identity = self._agent_identity
        canonical, normalized, reason = normalize_identity(
            self._agent_identity,
            allowed_themes=self._config.get("allowed_themes"),
            aliases=self._config.get("identity_aliases") or {},
            bench_mode=self._config.get("bench_mode", "bucket"),
        )
        if normalized:
            logger.info(
                "pgvector identity normalized: %r -> %r (%s)",
                self._raw_identity, canonical, reason,
            )
        self._agent_identity = canonical
        # Parent-session linkage for delegation traceback (subagents only).
        self._parent_session_id = kwargs.get("parent_session_id") or None

        # Re-initialization guard (v0.3.1): one registered provider instance
        # can have initialize() called again for a new session — the gateway
        # reuses the registered provider rather than constructing a fresh one
        # per session. Without tearing down the previous session's writer +
        # pool first, each re-init abandoned a ConnectionPool whose warm
        # connection lingered in Postgres until idle_session_timeout — the
        # v0.3.0 connection leak that saturated the server's slots under a
        # burst of concurrent sessions. shutdown() is idempotent and drains
        # in-flight writes, so calling it unconditionally here is safe.
        if self._store is not None or self._writer is not None:
            self.shutdown()

        self._store = MemoryStore(self._config["dsn"])
        try:
            # Schema is verify-only at runtime — admin applies the
            # migration out-of-band (see plugin README install step).
            self._store.ensure_schema()
            health = self._store.health()
            self._healthy = bool(health.get("ok"))
            if not self._healthy:
                logger.warning("pgvector unhealthy on init: %s", health.get("error"))
        except MemoryStore.SchemaNotApplied as exc:
            logger.error("pgvector schema not applied — %s", exc)
            self._healthy = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector init failed: %s", exc)
            self._healthy = False

        # Background writer — bounded queue, lazy thread start. Decouples
        # on_memory_write + sync_turn from the (potentially slow) embed +
        # DB write so the agent loop never blocks on a stalled embed
        # endpoint.
        self._writer = AsyncWriter(
            self._worker,
            maxsize=int(self._config.get("write_queue_maxsize", 256)),
        )

        # v0.4: agent attribution + delegation require migration 002. Probe it
        # SEPARATELY from ensure_schema() (which stays 001-only) so a v0.4 binary
        # on a not-yet-migrated v0.3 schema runs degraded — the delegation hooks
        # become no-ops — instead of crashing at the first write.
        self._delegation_enabled = False
        if self._healthy and self._store is not None:
            try:
                self._delegation_enabled = self._store.ensure_migration_002_applied()
            except Exception:  # noqa: BLE001
                self._delegation_enabled = False
            if not self._delegation_enabled:
                logger.info(
                    "pgvector: migration 002 not applied — agent attribution/"
                    "delegation disabled (apply 002_agent_attribution.sql to enable)"
                )
            else:
                # Register this agent in the provenance registry (best-effort, async).
                self._writer.enqueue(
                    action="register_agent",
                    agent_identity=self._agent_identity,
                    target="memory_agents",
                    content="",
                    extra={"kind": classify_kind(self._agent_identity)},
                    metadata={},
                )

        # v0.1.1: bulk import existing MEMORY.md / USER.md content so the
        # plugin sees pre-plugin entries + direct file edits, not just the
        # new writes captured via on_memory_write.
        if self._healthy and self._config.get("bulk_sync_on_init", True):
            self._bulk_sync_from_disk(kwargs.get("hermes_home"))

    def shutdown(self) -> None:
        # Drain the in-flight writes first so we don't drop work...
        if self._writer:
            self._writer.shutdown(timeout=5.0)
            self._writer = None
        # ...then close the pool the writer was draining into.
        if self._store:
            self._store.close()
            self._store = None
        self._healthy = False

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id

    # -- System prompt + ambient recall --------------------------------------

    def system_prompt_block(self) -> str:
        if not self._healthy or not self._store:
            return ""
        try:
            count_scoped = self._store.count(agent_identity=self._agent_identity)
            count_all = self._store.count()
        except Exception:  # noqa: BLE001
            count_scoped = count_all = 0
        if count_all == 0:
            return (
                "# pgvector memory\n"
                "Active. Empty store. Use the built-in `memory` tool to save "
                "durable notes — entries are mirrored to Postgres with "
                "embeddings for semantic recall across sessions."
            )
        return (
            "# pgvector memory\n"
            f"Active. {count_scoped} entries for '{self._agent_identity}', "
            f"{count_all} total across all themes. "
            "Use `recall_memory(query, scope='all'|'<theme>')` for deeper / "
            "cross-theme recall beyond what's in the built-in memory block."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._healthy or not self._store or not query:
            return ""
        try:
            vec = embed(
                query,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            logger.debug("pgvector prefetch embed failed: %s", exc)
            return ""

        # Ambient prefetch is scoped to the current agent_identity by
        # default — keeps marketing turns from polluting trading recall.
        try:
            rows = self._store.search(
                query_embedding=vec,
                agent_identity=self._agent_identity,
                limit=int(self._config.get("prefetch_limit", 5)),
                min_similarity=float(self._config.get("min_similarity", 0.30)),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("pgvector prefetch query failed: %s", exc)
            return ""
        if not rows:
            return ""

        lines = [f"## Recall (pgvector, {self._agent_identity})"]
        for r in rows:
            score = r.get("score") or 0.0
            tgt = r.get("target") or "?"
            content = (r.get("content") or "").strip().replace("\n", " ")
            if len(content) > 280:
                content = content[:280] + "…"
            lines.append(f"- [{score:.2f}] ({tgt}) {content}")
        return "\n".join(lines)

    # -- Turn capture (v0.2) -------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Persist a (user, assistant) turn pair to the conversations table.

        Non-blocking — enqueues writes; the async writer drains, embeds,
        and INSERTs. Boilerplate / very short turns are filtered out so
        the recall table stays high-signal.
        """
        if not self._healthy or not self._writer:
            return
        if not self._config.get("sync_turns", True):
            return

        sid = session_id or self._session_id or "default"
        min_chars = int(self._config.get("turn_min_chars", 40))
        policy = self._config.get("conversation_embed_policy", "all")
        psid = self._parent_session_id if self._delegation_enabled else None

        for role, content in (("user", user_content), ("assistant", assistant_content)):
            if not content:
                continue
            if self._is_noise(content, min_chars=min_chars):
                continue
            meta = {"session_id": sid}
            if self._raw_identity != self._agent_identity:
                meta["raw_identity"] = self._raw_identity
            self._writer.enqueue(
                action="turn",
                agent_identity=self._agent_identity,
                target="conversations",  # synthetic; worker dispatches on action
                content=content,
                extra={
                    "role": role,
                    "session_id": sid,
                    "embed": self._should_embed_turn(role, content, policy),
                    "parent_session_id": psid,
                },
                metadata=meta,
            )

    @staticmethod
    def _is_noise(content: str, *, min_chars: int) -> bool:
        """True for short / boilerplate content we don't want in recall."""
        stripped = (content or "").strip()
        if not stripped:
            return True
        if len(stripped) < min_chars:
            return True
        if _NOISE_RE.match(stripped):
            return True
        return False

    @staticmethod
    def _should_embed_turn(role: str, content: str, policy: str) -> bool:
        """Whether to embed a captured turn, per conversation_embed_policy.

        'all' (default) embeds every turn that already passed the noise filter
        — recall-safe. 'substantive_only' embeds user turns + assistant turns
        >= 120 chars (skips short acks/replies to cap HNSW growth). 'none'
        stores text-only (conversation recall disabled). The turn is ALWAYS
        stored regardless; this only controls embedding.
        """
        if policy == "none":
            return False
        if policy == "substantive_only":
            return role == "user" or len(content or "") >= 120
        return True

    # -- Agent attribution + delegation hooks (v0.4 — M4) -------------------

    def on_delegation(self, task: str, result: str, **kwargs) -> None:
        """Parent-side capture of a subagent delegation (task -> result).

        Records a provenance edge (memory_agent_edges) and stores the
        delegation as a recallable conversation turn under the parent's theme,
        linked to the child session via parent_session_id. STRICTLY non-blocking
        and fail-soft (invariants #2/#4): enqueue-only, never an inline DB/embed
        call, never raises into the agent loop. No-op until migration 002 is
        applied (self._delegation_enabled)."""
        if not self._healthy or not self._writer or not self._delegation_enabled:
            return
        try:
            child_identity = kwargs.get("child_identity") or kwargs.get("agent_identity")
            child_session_id = kwargs.get("child_session_id") or kwargs.get("session_id")
            self._writer.enqueue(
                action="edge",
                agent_identity=self._agent_identity,
                target="memory_agent_edges",
                content="",
                extra={
                    "parent_identity": self._agent_identity,
                    "child_identity": child_identity,
                    "parent_session_id": self._session_id,
                    "child_session_id": child_session_id,
                    "kind": "delegated",
                    "attrs": {"task": (task or "")[:500], "result_chars": len(result or "")},
                },
                metadata={},
            )
            # Store the delegation as a searchable turn under the parent theme.
            combined = f"[delegation] task: {task}\nresult: {result}".strip()
            if combined and not self._is_noise(combined, min_chars=1):
                policy = self._config.get("conversation_embed_policy", "all")
                self._writer.enqueue(
                    action="turn",
                    agent_identity=self._agent_identity,
                    target="conversations",
                    content=combined[:8000],
                    extra={
                        "role": "assistant",
                        "session_id": self._session_id or "default",
                        "embed": self._should_embed_turn("assistant", combined, policy),
                        "parent_session_id": child_session_id,
                    },
                    metadata={"kind": "delegation", "child_session_id": child_session_id},
                )
        except Exception as exc:  # noqa: BLE001 — never escape into the agent loop
            logger.debug("pgvector on_delegation failed (ignored): %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Best-effort: capture session turns + bump last_seen. Fail-soft,
        non-blocking. No LLM summary (invariant #2). No-op until 002 applied."""
        if not self._healthy or not self._writer or not self._delegation_enabled:
            return
        try:
            self._writer.enqueue(
                action="register_agent",
                agent_identity=self._agent_identity,
                target="memory_agents",
                content="",
                extra={"kind": classify_kind(self._agent_identity)},
                metadata={},
            )
            # Capture substantive user/assistant turns for conversation recall.
            if messages:
                policy = self._config.get("conversation_embed_policy", "all")
                for msg in messages:
                    role = (msg.get("role") or "").lower()
                    if role not in ("user", "assistant"):
                        continue
                    raw = msg.get("content") or ""
                    content = (
                        raw if isinstance(raw, str)
                        else " ".join(
                            p.get("text", "") for p in raw
                            if isinstance(p, dict) and p.get("type") == "text"
                        ) if isinstance(raw, list) else str(raw)
                    )
                    if self._is_noise(content, min_chars=40):
                        continue
                    self._writer.enqueue(
                        action="turn",
                        agent_identity=self._agent_identity,
                        target="conversations",
                        content=content[:8000],
                        extra={
                            "role": role,
                            "session_id": self._session_id or "default",
                            "embed": self._should_embed_turn(role, content, policy),
                            "parent_session_id": self._parent_session_id,
                        },
                        metadata={},
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("pgvector on_session_end failed (ignored): %s", exc)

    # -- Built-in memory mirror (THE main integration point) ----------------

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in `memory` tool writes to Postgres (non-blocking).

        Built-in tool fires this on every add/replace/remove. We enqueue
        the write; the background thread drains, embeds, and INSERTs.
        Returns instantly so the agent loop never blocks on the embed
        endpoint or the DB.
        """
        if not self._healthy or not self._writer:
            return
        if target not in ("memory", "user"):
            logger.debug("pgvector ignoring unsupported target: %r", target)
            return
        if action not in ("add", "replace", "remove"):
            logger.debug("pgvector ignoring unknown action: %r", action)
            return

        meta = dict(metadata or {})
        meta.setdefault("session_id", self._session_id)
        if self._delegation_enabled and self._parent_session_id:
            meta.setdefault("parent_session_id", self._parent_session_id)
        if self._raw_identity != self._agent_identity:
            meta.setdefault("raw_identity", self._raw_identity)
        old_text = meta.get("old_text") or meta.get("replaces")

        self._writer.enqueue(
            action=action,
            agent_identity=self._agent_identity,
            target=target,
            content=content,
            extra={"old_text": str(old_text)} if old_text else {},
            metadata=meta,
        )

    def _worker(self, item: "_PendingWrite") -> None:
        """Drain-thread worker: embed + DB write for a single queued item.

        Must NOT raise — the AsyncWriter logs + survives if we do, but
        we still want failures to degrade gracefully (drop the write,
        keep the queue moving).
        """
        if not self._store:
            return
        try:
            if item.action == "add":
                vec = self._maybe_embed(item.content)
                self._store.add(
                    agent_identity=item.agent_identity,
                    target=item.target,
                    content=item.content,
                    embedding=vec,
                    metadata=item.metadata,
                )
            elif item.action == "replace":
                old_text = item.extra.get("old_text")
                vec = self._maybe_embed(item.content)
                if old_text:
                    n = self._store.replace(
                        agent_identity=item.agent_identity,
                        target=item.target,
                        old_text=old_text,
                        new_content=item.content,
                        new_embedding=vec,
                    )
                    if n == 0:
                        # Nothing matched — degrade to add (built-in wrote
                        # the new entry to disk; mirror it).
                        self._store.add(
                            agent_identity=item.agent_identity,
                            target=item.target,
                            content=item.content,
                            embedding=vec,
                            metadata=item.metadata,
                        )
                else:
                    # No old_text in metadata → can't locate prior row;
                    # add the new content so we don't lose it.
                    self._store.add(
                        agent_identity=item.agent_identity,
                        target=item.target,
                        content=item.content,
                        embedding=vec,
                        metadata=item.metadata,
                    )
            elif item.action == "remove":
                self._store.remove(
                    agent_identity=item.agent_identity,
                    target=item.target,
                    old_text=item.content,
                )
            elif item.action == "turn":
                role = item.extra.get("role") or "user"
                sid = item.extra.get("session_id") or "default"
                vec = self._maybe_embed(item.content) if item.extra.get("embed", True) else None
                self._store.append_turn(
                    session_id=sid,
                    agent_identity=item.agent_identity,
                    role=role,
                    content=item.content,
                    embedding=vec,
                    metadata=item.metadata,
                    parent_session_id=item.extra.get("parent_session_id"),
                )
            elif item.action == "register_agent":
                self._store.register_agent(
                    agent_identity=item.agent_identity,
                    kind=item.extra.get("kind", "theme"),
                )
            elif item.action == "edge":
                self._store.record_delegation(
                    parent_identity=item.extra.get("parent_identity"),
                    child_identity=item.extra.get("child_identity"),
                    parent_session_id=item.extra.get("parent_session_id"),
                    child_session_id=item.extra.get("child_session_id"),
                    kind=item.extra.get("kind", "delegated"),
                    attrs=item.extra.get("attrs") or {},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "pgvector worker (%s/%s/%s) failed: %s",
                item.action,
                item.agent_identity,
                item.target,
                str(exc)[:200],
            )

    # -- Bulk sync (v0.1.1) --------------------------------------------------

    def _bulk_sync_from_disk(self, hermes_home: Optional[str]) -> None:
        """Import MEMORY.md + USER.md entries from disk into memory_entries.

        Called by initialize(). Runs synchronously (not via async writer)
        so the table is warm before the first turn's prefetch. Cheap on
        re-init: an existence pre-check skips already-imported entries
        without re-embedding.
        """
        if not self._store:
            return
        if not hermes_home:
            # Fall back to hermes_constants if the runtime didn't pass it.
            try:
                from hermes_constants import get_hermes_home
                hermes_home = str(get_hermes_home())
            except Exception:  # noqa: BLE001
                return

        memories_dir = Path(hermes_home) / "memories"
        embed_fn = self._make_embed_fn()

        for target, fname in (("memory", "MEMORY.md"), ("user", "USER.md")):
            try:
                result = self._store.bulk_upsert_md(
                    agent_identity=self._agent_identity,
                    target=target,
                    file_path=memories_dir / fname,
                    embed_fn=embed_fn,
                )
                if result.get("inserted"):
                    logger.info(
                        "pgvector bulk-sync %s: parsed=%d inserted=%d skipped=%d",
                        fname,
                        result.get("parsed", 0),
                        result.get("inserted", 0),
                        result.get("skipped", 0),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("pgvector bulk-sync %s failed: %s", fname, exc)

    def _make_embed_fn(self):
        """Return a closure over the configured embed endpoint, or None."""
        if not self._config.get("embed_on_write", True):
            return None
        base_url = self._config["embed_url"]
        model = self._config["embed_model"]
        def _fn(text: str):
            return embed(text, base_url=base_url, model=model)
        return _fn

    # -- Tool surface --------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_MEMORY_SCHEMA, RECALL_CONVERSATION_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "recall_conversation":
            return self._handle_recall_conversation(args)
        if tool_name != "recall_memory":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._healthy or not self._store:
            return json.dumps({"results": [], "count": 0, "error": "pgvector unavailable"})

        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required arg: query")

        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        # Scope resolution: 'current' → my agent_identity; 'all' → no filter;
        # anything else → treat as explicit theme name.
        scope = (args.get("scope") or self._config.get("scope_default") or "current").strip()
        if scope == "current":
            agent_filter: Optional[str] = self._agent_identity
        elif scope == "all":
            agent_filter = None
        else:
            agent_filter = scope

        # Target resolution: 'memory'/'user'/'both'.
        target_arg = (args.get("target") or "both").strip()
        target_filter: Optional[str] = None if target_arg == "both" else target_arg
        if target_filter not in (None, "memory", "user"):
            return tool_error(f"Invalid target: {target_arg!r}")

        try:
            vec = embed(
                query,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            return json.dumps({"results": [], "count": 0, "error": f"embed: {exc}"})

        try:
            rows = self._store.search(
                query_embedding=vec,
                agent_identity=agent_filter,
                target=target_filter,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"results": [], "count": 0, "error": f"db: {exc}"})

        results = []
        for r in rows:
            ts = r.get("updated_at") or r.get("created_at")
            results.append(
                {
                    "id": r.get("id"),
                    "agent_identity": r.get("agent_identity"),
                    "target": r.get("target"),
                    "ts": ts.isoformat() if ts else None,
                    "score": round(float(r.get("score") or 0.0), 4),
                    "content": (r.get("content") or "")[:2000],
                }
            )
        return json.dumps({"results": results, "count": len(results)})

    def _handle_recall_conversation(self, args: Dict[str, Any]) -> str:
        """Tool handler for recall_conversation over the conversations table."""
        if not self._healthy or not self._store:
            return json.dumps({"results": [], "count": 0, "error": "pgvector unavailable"})

        query = (args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required arg: query")
        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        scope = (args.get("scope") or "current").strip()
        agent_filter: Optional[str] = None
        session_filter: Optional[str] = None
        if scope == "current":
            agent_filter = self._agent_identity
        elif scope == "session":
            session_filter = self._session_id or None
        elif scope == "all":
            pass  # no filters
        else:
            agent_filter = scope  # treat as a specific theme name

        try:
            vec = embed(
                query,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
            )
        except EmbeddingError as exc:
            return json.dumps({"results": [], "count": 0, "error": f"embed: {exc}"})

        try:
            rows = self._store.search_turns(
                query_embedding=vec,
                agent_identity=agent_filter,
                session_id=session_filter,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"results": [], "count": 0, "error": f"db: {exc}"})

        results = []
        for r in rows:
            ts = r.get("ts")
            results.append(
                {
                    "id": r.get("id"),
                    "agent_identity": r.get("agent_identity"),
                    "session_id": r.get("session_id"),
                    "role": r.get("role"),
                    "ts": ts.isoformat() if ts else None,
                    "score": round(float(r.get("score") or 0.0), 4),
                    "content": (r.get("content") or "")[:2000],
                }
            )
        return json.dumps({"results": results, "count": len(results)})

    # -- Setup hooks ---------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "dsn",
                "description": "Postgres DSN (psycopg connection string)",
                "default": DEFAULTS["dsn"],
                "required": True,
            },
            {
                "key": "embed_url",
                "description": "Embedding endpoint base URL (OpenAI-compatible or Ollama native)",
                "default": DEFAULTS["embed_url"],
                "required": True,
            },
            {
                "key": "embed_model",
                "description": "Embedding model name (must return 768-dim vectors)",
                "default": DEFAULTS["embed_model"],
            },
            {
                "key": "prefetch_limit",
                "description": "Max ambient recall results injected per turn",
                "default": str(DEFAULTS["prefetch_limit"]),
            },
            {
                "key": "min_similarity",
                "description": "Cosine similarity cutoff for ambient prefetch (0.0–1.0)",
                "default": str(DEFAULTS["min_similarity"]),
            },
            {
                "key": "embed_on_write",
                "description": "Compute embedding on each write; turn off for text-only mode",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "scope_default",
                "description": "Default scope for recall_memory when caller omits it",
                "default": DEFAULTS["scope_default"],
                "choices": ["current", "all"],
            },
            {
                "key": "write_queue_maxsize",
                "description": "Bounded async-writer queue size; full = oldest writes drop with a warning",
                "default": str(DEFAULTS["write_queue_maxsize"]),
            },
            {
                "key": "bulk_sync_on_init",
                "description": "Import MEMORY.md / USER.md content from disk on agent init (v0.1.1)",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "sync_turns",
                "description": "Capture every substantive (user, assistant) turn pair into the conversations table",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "turn_min_chars",
                "description": "Turns shorter than this (after strip) are treated as boilerplate and skipped",
                "default": str(DEFAULTS["turn_min_chars"]),
            },
            {
                "key": "allowed_themes",
                "description": "v0.4 identity governance: optional allow-list of theme names. Empty/unset = allow any. When set, an unknown X-Hermes-Session-Key falls back to 'default' with a one-time warning (the whatsapp-dm / _bench / default sinks are always permitted).",
                "default": "",
            },
            {
                "key": "bench_mode",
                "description": "v0.4: how to handle benchmark identities (skill-bench, *-bench): 'bucket' isolates them to the '_bench' theme; 'reject' drops them to 'default'.",
                "default": DEFAULTS["bench_mode"],
                "choices": ["bucket", "reject"],
            },
            {
                "key": "conversation_embed_policy",
                "description": "v0.4: which captured turns get an embedding. 'all' (recall-safe; every turn that passed the noise filter) | 'substantive_only' (user turns + assistant turns >=120 chars; caps HNSW growth) | 'none' (text-only, conversation recall disabled).",
                "default": DEFAULTS["conversation_embed_policy"],
                "choices": ["all", "substantive_only", "none"],
            },
            {
                "key": "ttl_days",
                "description": "v0.4: advisory retention window for conversations (memory_entries are NEVER pruned). 0 = off. Pruning only ever happens when an operator runs `python -m pgvector prune` — this value is the default for that command, never an automatic background delete.",
                "default": str(DEFAULTS["ttl_days"]),
            },
            {
                "key": "embed_write_retries",
                "description": "v0.4: bounded embed retries on the background writer path ONLY (the hot path — prefetch/recall/sync — always uses a single attempt). Durable recovery of missed embeddings is the `python -m pgvector backfill` sweep, not inline retries.",
                "default": str(DEFAULTS["embed_write_retries"]),
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing: Dict[str, Any] = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as fh:
                    existing = yaml.safe_load(fh) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["pgvector"] = values
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.dump(existing, fh, default_flow_style=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector save_config failed: %s", exc)

    # -- Helpers -------------------------------------------------------------

    def _maybe_embed(self, content: str) -> Optional[List[float]]:
        if not self._config.get("embed_on_write", True):
            return None
        try:
            # This runs ONLY in the background AsyncWriter drain thread, so a
            # bounded retry here is safe (it never blocks the agent loop). The
            # hot-path callers (prefetch / recall tools / sync_turn) call embed()
            # with the default retries=0.
            return embed(
                content,
                base_url=self._config["embed_url"],
                model=self._config["embed_model"],
                retries=int(self._config.get("embed_write_retries", 2)),
                backoff=float(self._config.get("embed_write_backoff", 0.1)),
            )
        except EmbeddingError as exc:
            if not self._embed_warned:
                logger.warning("pgvector embed failed (degrading to text-only): %s", exc)
                self._embed_warned = True
            return None


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the pgvector memory provider with the plugin system."""
    provider = PgvectorMemoryProvider(config=_load_plugin_config())
    ctx.register_memory_provider(provider)

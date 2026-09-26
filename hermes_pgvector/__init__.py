"""pgvector — Postgres + pgvector memory provider for hermes-agent.

Mirrors hermes-agent's built-in `memory` tool entries (MEMORY.md / USER.md
in tools/memory_tool.py) into a single Postgres table, adds embeddings
(768-dim by default; see embed_dim) for semantic recall, and scopes by
`agent_identity` so each
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
        embed_url:  "http://localhost:11434"
        embed_model: "nomic-embed-text"
        embed_dim: 768             # must match the model AND the vector(N) columns
        embed_api_key_env: ""      # NAME of an env var holding a bearer token
        embed_protocol: "auto"     # 'auto' | 'openai' | 'ollama'
        prefetch_limit: 5
        min_similarity: 0.30
        embed_on_write: true
        scope_default: "current"   # 'current' | 'all'
        hybrid_search: true        # fuse vector + full-text (RRF) in recall tools

Tools exposed: `recall_memory` and `recall_conversation` (two explicit search
tools). All built-in memory writes (add/replace/remove) are mirrored
automatically via the on_memory_write hook — no agent-facing change.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider, spawn_context_thread
    from tools.registry import tool_error
    from hermes_cli.config import cfg_get
except ImportError:  # pragma: no cover
    # Standalone context (the `hermes-pgvector` CLI / unit tests) — hermes-agent
    # is not importable. Provide minimal fallbacks so the package still imports;
    # the provider class is never instantiated outside the agent, only the store/
    # embed/identity helpers + the maintenance CLI are used here.
    MemoryProvider = object  # type: ignore[assignment,misc]

    def spawn_context_thread(  # type: ignore[misc]
        target, *, name: str, daemon: bool = True, args: tuple = (), kwargs=None,
    ) -> threading.Thread:
        # Standalone fallback: no profile-isolation contextvars to bind (that
        # machinery lives in agent.memory_provider). Same shape as upstream --
        # an UNSTARTED daemon thread the caller starts itself.
        return threading.Thread(target=target, args=args, kwargs=kwargs or {}, name=name, daemon=daemon)

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

from .embed import EmbeddingError
# Public re-export only (`from hermes_pgvector import embed` keeps working).
# Code in this file must NEVER call the bare name `embed`: hermes-agent's
# directory loader (plugins/plugin_loader.py:load_plugin_module) runs
# `setattr(package, "embed", <the embed SUBMODULE>)` after executing this file,
# which rebinds that global to a module object -- every `embed(...)` here then
# raised `TypeError: 'module' object is not callable`. Call sites go through
# _embed_with_config(), which uses the private alias below; the loader only
# rebinds names that match a sibling file, so the alias is never clobbered.
from .embed import embed  # noqa: F401
from .embed import embed as _embed_text
from .identity import (BENCH_BUCKET, DM_BUCKET, GROUP_BUCKET, classify_kind,
                       normalize_identity)
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


# Credential-looking fragments that must never round-trip into a tool response
# (psycopg/libpq errors can echo conninfo verbatim; tool errors flow back to
# the model and can be persisted durably by conversation capture).
_CRED_RE = re.compile(r"(password|passfile|sslkey|sslpassword)=\S+", re.IGNORECASE)


def _safe_err(exc: BaseException) -> str:
    """Exception text safe to return to the model: truncated + creds redacted."""
    return _CRED_RE.sub(r"\1=[redacted]", str(exc)[:300])


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
    # L2 (v0.6.0): was a private LAN address (192.168.100.50) -- wrong default
    # for anyone outside that one deployment, and it leaked that deployment's
    # topology into every fresh install. Upgrade note: set this explicitly
    # before upgrading if you relied on the old default.
    "embed_url": "http://localhost:11434",
    "embed_model": "nomic-embed-text",
    # v0.5.3 -- the embedding contract is configuration, not code. Defaults
    # reproduce the pre-0.5.3 behaviour exactly: 768 dims, no Authorization
    # header, OpenAI-compatible path with the Ollama-native fallback.
    # embed_dim must equal the model's output AND the vector(N) columns;
    # changing it on an existing database needs a column migration + re-embed
    # (see README, "Changing the embedding dimension").
    "embed_dim": 768,
    # NAME of an environment variable holding a bearer token (e.g.
    # "OPENROUTER_API_KEY"), never the token itself. Read at call time.
    "embed_api_key_env": None,
    "embed_protocol": "auto",  # auto | openai | ollama
    "prefetch_limit": 5,
    "min_similarity": 0.30,
    "embed_on_write": True,
    "scope_default": "current",
    # v0.4.1 — hybrid recall: fuse the HNSW vector ranking with a Postgres
    # full-text ranking via RRF in the recall_memory / recall_conversation
    # tools. Recovers exact-lexical hits cosine smooths away + text-only rows
    # with NULL embeddings. Degrades to pure vector on any error, and to
    # full-text-only when the query fails to embed. Needs migration 003 for the
    # GIN index (works without it, just slower — seq-scan FTS on a small table).
    "hybrid_search": True,
    "write_queue_maxsize": 256,
    # v0.1.1 — bulk sync MEMORY.md / USER.md on init
    "bulk_sync_on_init": True,
    # v0.2 — conversation turn capture
    "sync_turns": True,
    "turn_min_chars": 40,  # turns shorter than this (after strip) are boilerplate noise
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
    # v0.5.0 — embed timeouts, split by call context. Previously `timeout` was
    # never plumbed from config at all: every caller took embed()'s hardcoded
    # 10s. On a deployment whose endpoint answers in 6-17s (measured on the
    # home k8s nomic-embed-text) that means a large share of writes time out,
    # fail soft, and land with a NULL embedding -- unsearchable until the
    # nightly backfill sweep. Retries could not rescue it, because every
    # attempt was capped BELOW the latency the endpoint actually needs.
    #
    # Hot path stays short on purpose: prefetch and the recall tools run on the
    # agent thread, and a slow query there degrades to full-text-only recall,
    # which is a good outcome. Waiting longer would be the worse one.
    #
    # v0.6.0 (H4): lowered from 10.0. The host's external-provider prefetch
    # budget is a hard 8s (agent.memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S) --
    # under embed_protocol='auto' a single embed() call can try BOTH the
    # OpenAI-compat and Ollama-native paths, and 10s per call already left no
    # room for that fallback (let alone the DB search after it) inside the
    # host's budget. See prefetch_budget below, which is what prefetch()'s
    # synchronous fallback path actually enforces.
    "embed_timeout": 5.0,
    # Writer drain only. Nothing is waiting on it, and the cost of giving up is
    # a permanently unsearchable row, so it gets real headroom.
    "embed_write_timeout": 30.0,
    # v0.6.0 (H4) -- hard wall-clock budget for prefetch()'s SYNCHRONOUS
    # fallback path (embed + search), used only when queue_prefetch() has not
    # already cached a formatted block for this session. Passed to embed() as
    # its timeout, which -- under embed_protocol='auto' -- is now ONE shared
    # deadline across both the OpenAI-compat and Ollama-native attempts (see
    # embed.py), not a fresh full timeout for each. Must stay below the host's
    # 8s external-prefetch budget or the host logs a timeout warning and skips
    # this provider on later turns until the stuck call returns.
    "prefetch_budget": 5.0,
    # v0.6.0 (M4) -- which agent_context values from initialize()'s kwargs are
    # allowed to WRITE. Recall (prefetch, recall tools, system_prompt_block) is
    # never gated by this -- only mutation paths. "primary" is normal
    # interactive/API traffic; "cron" is real scheduled work; "subagent" output
    # reaches the parent via on_delegation, not by writing directly under its
    # own context; "flush" is a background teardown pass. Accepts a comma
    # string (save_config()) or a YAML list; compared lowercased/stripped.
    "write_contexts": "primary,cron",
    # v0.6.0 (M3) -- timeout passed to the writer's shutdown() drain. Kept
    # short: while draining, _maybe_embed() short-circuits to None (no embed
    # endpoint call), so the drain is DB-only and fast -- rows land text-only
    # and the nightly backfill sweep heals them later.
    "shutdown_drain_timeout": 10.0,
}


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce a config value to a real bool.

    Config reaches us with three different types: DEFAULTS (real bools), a
    hand-edited config.yaml (YAML bools), and save_config(), which persists
    the config schema's declared values -- the STRINGS "true"/"false". A plain
    truthiness test silently inverts the string form (bool("false") is True),
    so every boolean toggle reads through here.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int) -> int:
    """Coerce a config value to int, falling back to the default.

    Same hazard as _as_bool: config values arrive as strings from
    save_config(), and an empty or malformed entry must not raise on a
    write path (invariant #4).
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    """Coerce a config value to float, falling back to the default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_theme_list(value: Any) -> Optional[List[str]]:
    """Coerce allowed_themes into a list of theme names.

    The config schema declares this key as a scalar string; the README
    documents it as a YAML list. A bare string must never reach
    normalize_identity(), which iterates its argument -- a string iterates
    CHARACTER BY CHARACTER, so every real theme fails the membership test and
    the allow-list silently routes the entire fleet to 'default'.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()] or None
    try:
        return list(value) or None
    except TypeError:
        return None


def _as_write_contexts(value: Any) -> List[str]:
    """Coerce write_contexts into a lowercased/stripped list of context names.

    Same string-vs-list hazard as _as_theme_list (config arrives as a
    save_config() comma string OR a hand-edited config.yaml YAML list), but
    unlike allowed_themes an empty/missing/malformed value is never "off" --
    write_contexts must always resolve to a concrete list, so this falls back
    to DEFAULTS["write_contexts"] instead of returning None.
    """
    default = DEFAULTS["write_contexts"]
    if value is None or (isinstance(value, str) and not value.strip()):
        value = default
    if isinstance(value, str):
        items = [x.strip().lower() for x in value.split(",") if x.strip()]
    else:
        try:
            items = [str(x).strip().lower() for x in value if str(x).strip()]
        except TypeError:
            items = []
    return items or [x.strip().lower() for x in default.split(",") if x.strip()]


def _as_alias_map(value: Any) -> Dict[str, str]:
    """Coerce identity_aliases into a real {raw: canonical} dict.

    Same string-vs-container hazard as _as_theme_list/_as_write_contexts, but
    for a MAPPING: get_config_schema() types are text|integer|number|boolean
    -- there is no mapping type -- so identity_aliases is declared as `text`
    in the documented "raw=canonical,raw2=canonical2" form. DEFAULTS ships a
    real dict and a hand-edited config.yaml may still use the YAML mapping
    form the README documents, but a string written back by save_config()
    must never reach normalize_identity()'s `aliases` argument as-is: it
    iterates its argument as a container (`raw in aliases`), and a str
    iterates CHARACTER BY CHARACTER, so every alias would silently stop
    matching.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        out: Dict[str, str] = {}
        for pair in value.split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            raw, _, canonical = pair.partition("=")
            raw = raw.strip()
            canonical = canonical.strip()
            if raw and canonical:
                out[raw] = canonical
        return out
    return {}


def _embed_dim(config: Dict[str, Any]) -> int:
    """embed_dim as a positive int; missing, malformed or <= 0 -> the default.

    A bad value is not guessed at: the endpoint's real dimension then fails
    the check with "expected 768 dims (embed_dim), got N", which names the key
    to fix.
    """
    dim = _as_int(config.get("embed_dim"), DEFAULTS["embed_dim"])
    return dim if dim > 0 else int(DEFAULTS["embed_dim"])


def _embed_with_config(
    text: str,
    config: Dict[str, Any],
    *,
    timeout: float,
    retries: int = 0,
    backoff: float = 0.1,
    max_total: Optional[float] = None,
) -> List[float]:
    """Embed `text` using the endpoint settings in `config`.

    The ONE place this package requests an embedding: prefetch, both recall
    tools, the init-time bulk import, the writer drain and the operator CLI
    all come through here, so embed_url / embed_model / embed_dim /
    embed_api_key_env / embed_protocol apply identically on every path.
    Callers keep owning the timing policy -- the agent-thread paths pass
    embed_timeout with a single attempt; the writer drain passes
    embed_write_timeout with bounded retries (invariant #2).

    Uses `_embed_text`, not the global `embed`; see the import note above.
    """
    key_env = config.get("embed_api_key_env")
    return _embed_text(
        text,
        base_url=config.get("embed_url", DEFAULTS["embed_url"]),
        model=config.get("embed_model", DEFAULTS["embed_model"]),
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        max_total=max_total,
        dim=_embed_dim(config),
        api_key_env=str(key_env).strip() if key_env else None,
        protocol=config.get("embed_protocol", DEFAULTS["embed_protocol"]),
    )


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

    # Upper bound on the turn-fingerprint set. Fingerprints deliberately
    # survive session switches (see on_session_switch), so this is what keeps
    # a long-lived provider from growing without limit. ~60 fingerprints per
    # session means this holds roughly 150 sessions before a reset.
    _FINGERPRINT_CAP = 10000

    # H4 (v0.6.0) -- upper bound on the queue_prefetch() cache, keyed by
    # session_id. Bounds a long-lived provider instance the same way
    # _FINGERPRINT_CAP does; a provider only ever needs a handful of
    # concurrent sessions cached at once.
    _PREFETCH_CACHE_CAP = 32

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
        self._db_warned: bool = False
        # v0.6.0 (M4) -- agent_context gate. True until the first initialize()
        # decides otherwise, so a provider used without initialize() (unit
        # tests instantiate it directly) keeps writing.
        self._agent_context: str = "primary"
        self._writes_enabled: bool = True
        # v0.6.0 (M3) -- set by shutdown() before draining the writer, cleared
        # again once a fresh writer is built for a new session. While set,
        # _maybe_embed() skips the embed endpoint so the drain is DB-only.
        self._draining: bool = False
        # Fingerprints of turns already enqueued by sync_turn this session, so
        # on_session_end can act as a backstop without double-writing rows the
        # per-turn path already captured (conversations has no unique key).
        self._turn_fingerprints: set = set()
        # H4 (v0.6.0) -- queue_prefetch()'s background cache. One worker
        # thread at most, guarded by _prefetch_lock; a newer queue_prefetch()
        # call while one is in flight replaces _prefetch_pending rather than
        # starting a second thread. _prefetch_generation is bumped by
        # initialize()/shutdown() so a worker still running from a PRIOR
        # session can't write a stale cache entry after the cache has been
        # dropped for the new one.
        self._prefetch_cache: "OrderedDict[str, str]" = OrderedDict()
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending: Optional[tuple] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_generation: int = 0
        # M6 (v0.6.0) -- system_prompt_block() is STATIC per the upstream
        # contract; computed once (initialize(), or lazily on first call for
        # callers that skip initialize()) and cached here so later calls never
        # re-query the DB. None = "not computed yet"; a computed value is
        # never None (an empty string is a valid cached result).
        self._system_prompt_block_cache: Optional[str] = None

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
        # Per-session log-signal reset (v0.4.2): the provider instance is
        # reused across sessions, so without this a single embed failure in
        # any past session silences the one-time warning for the process
        # lifetime — a later session with a broken endpoint gets zero signal.
        self._embed_warned = False
        self._db_warned = False
        # _turn_fingerprints is deliberately NOT reset here either -- same
        # reasoning as on_session_switch: the dedup guard must survive a
        # re-initialize on a reused instance. The cap bounds it instead.
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
            allowed_themes=_as_theme_list(self._config.get("allowed_themes")),
            aliases=_as_alias_map(self._config.get("identity_aliases")),
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

        # v0.6.0 (M4) -- agent_context write gate. Upstream values: primary |
        # subagent | cron | flush (agent/agent_init.py). Recall (prefetch,
        # recall tools, system_prompt_block) is never gated by this -- only
        # mutation paths (on_memory_write, sync_turn, on_session_end,
        # on_delegation, plus the bulk import + register_agent enqueue below).
        raw_context = kwargs.get("agent_context")
        self._agent_context = str(raw_context).strip().lower() if raw_context else "primary"
        write_contexts = _as_write_contexts(self._config.get("write_contexts"))
        self._writes_enabled = self._agent_context in write_contexts
        logger.info(
            "pgvector: agent_context=%r write_contexts=%r -> writes %s",
            self._agent_context, write_contexts,
            "enabled" if self._writes_enabled else "disabled",
        )

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
        try:
            _queue_max = int(self._config.get("write_queue_maxsize", 256))
        except (TypeError, ValueError):
            _queue_max = int(DEFAULTS["write_queue_maxsize"])
        self._writer = AsyncWriter(self._worker, maxsize=_queue_max)
        # v0.6.0 (M3): a fresh writer means a fresh drain -- clear the flag a
        # prior shutdown() (e.g. the re-initialization guard above, on a
        # reused provider instance) may have left set, or every write this
        # session would silently skip embedding.
        self._draining = False

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
            elif self._writes_enabled:
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
        # plugin sees pre-plugin entries and entries ADDED by direct file
        # edits, not just the new writes captured via on_memory_write.
        # NOTE: this path is insert-only. It never reconciles removals or
        # rewrites -- editing an existing line in MEMORY.md leaves the stale
        # row in place alongside the new one, and recall (cosine / RRF, no
        # recency tiebreak) can surface both. `hermes-pgvector cleanup` is
        # the only remedy.
        if (
            self._healthy
            and self._writes_enabled
            and _as_bool(self._config.get("bulk_sync_on_init"), True)
        ):
            self._bulk_sync_from_disk(kwargs.get("hermes_home"))

        # M6 (v0.6.0): compute the STATIC system-prompt block ONCE here, after
        # the bulk import above so a freshly-imported store's counts are
        # reflected -- system_prompt_block() then just returns this cached
        # value and never queries the DB again for the rest of the session.
        self._system_prompt_block_cache = self._compute_system_prompt_block()

    def shutdown(self) -> None:
        # H4 (v0.6.0): drop the queue_prefetch() cache and bump the
        # generation counter FIRST, so a worker thread from a session that is
        # ending (still running because it is a daemon thread we never join)
        # cannot write a stale entry into the NEXT session's cache after this
        # returns -- see _prefetch_worker_loop's generation check.
        with self._prefetch_lock:
            self._prefetch_cache.clear()
            self._prefetch_pending = None
            self._prefetch_generation += 1
        # Drain the in-flight writes first so we don't drop work...
        if self._writer:
            # v0.6.0 (M3): set BEFORE draining. _maybe_embed() checks this and
            # returns None without calling the embed endpoint, so the drain is
            # DB-only and fast -- rows land text-only; the nightly backfill
            # sweep heals them. Without this, one write mid-drain can spend up
            # to embed_write_timeout * 2 blocking the drain thread, and
            # shutdown_drain_timeout below abandons every write still queued
            # behind it.
            self._draining = True
            timeout = _as_float(
                self._config.get("shutdown_drain_timeout"),
                DEFAULTS["shutdown_drain_timeout"],
            )
            self._writer.shutdown(timeout=timeout)
            self._writer = None
        # ...then close the pool the writer was draining into.
        if self._store:
            self._store.close()
            self._store = None
        self._healthy = False

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        # Per-session log-signal reset, same as initialize(): the provider
        # instance is reused across sessions, so a single embed/DB failure in
        # an earlier session would otherwise silence the one-shot warnings for
        # the rest of the process.
        self._embed_warned = False
        self._db_warned = False
        # NOT cleared here. on_session_end runs before on_session_switch only
        # on the commit_session_boundary_async path; conversation_compression.py
        # calls on_session_switch DIRECTLY with no on_session_end, so clearing
        # would let the turn double-write reappear on every context compression
        # (which fires on any long session). Fingerprints therefore live for the
        # provider's lifetime, bounded below.
        if len(self._turn_fingerprints) > self._FINGERPRINT_CAP:
            # Bound the memory: a very long-lived process may re-duplicate one
            # turn after a reset, which is far cheaper than unbounded growth.
            logger.debug(
                "pgvector turn-fingerprint set exceeded %d; resetting",
                self._FINGERPRINT_CAP,
            )
            self._turn_fingerprints = set()
        # NOTE: deliberately does NOT touch _parent_session_id. The host's
        # on_session_switch(parent_session_id=...) carries the PREVIOUS session
        # in this agent's own lineage (/new passes old_session_id, /undo passes
        # ""), NOT a delegation parent. _parent_session_id feeds
        # conversations.parent_session_id, which migration 002 defines as
        # delegation traceback -- assigning lineage here would stamp a
        # fabricated delegation edge on every write after a rotation, and the
        # /undo path would erase a real subagent's parent.

    # -- System prompt + ambient recall --------------------------------------

    def system_prompt_block(self) -> str:
        # M6 (v0.6.0): computed ONCE (initialize() primes this as its last
        # step); this lazy branch only covers a caller that skips
        # initialize() entirely (unit tests; a defensive fallback). Either
        # way, at most one DB round trip per cached value -- never one per
        # call.
        if self._system_prompt_block_cache is None:
            self._system_prompt_block_cache = self._compute_system_prompt_block()
        return self._system_prompt_block_cache

    def _compute_system_prompt_block(self) -> str:
        """The actual count()/estimate_count() work behind system_prompt_block().

        M6 (v0.6.0): count_all used to be an UNSCOPED self._store.count() --
        a full COUNT(*) over memory_entries on every session init. It is now
        estimate_count("memory_entries"), which reads pg_class.reltuples
        (O(1)) instead. That helper returns None (never 0) whenever the
        estimate is unknown -- a genuinely empty table and a stale
        pre-ANALYZE zero are indistinguishable from reltuples alone -- so
        None must never be rendered as "0 total" (that would assert
        something false: XCUT-3b). A failed SCOPED count is likewise not the
        same fact as "the store has zero rows"; that path returns ""
        instead of falling through to an "Empty store" claim.
        """
        if not self._healthy or not self._store:
            return ""
        try:
            count_scoped = self._store.count(agent_identity=self._agent_identity)
        except Exception as exc:  # noqa: BLE001
            logger.debug("pgvector system_prompt_block count failed: %s", exc)
            return ""
        try:
            count_all = self._store.estimate_count("memory_entries")
        except Exception as exc:  # noqa: BLE001 -- best-effort global figure only
            logger.debug("pgvector system_prompt_block estimate_count failed: %s", exc)
            count_all = None

        if count_scoped == 0 and count_all is None:
            # Nothing to report for THIS theme, and no reliable global figure
            # either -- NOT the same claim as "Empty store" (other themes may
            # already hold rows; estimate_count() returning None here is
            # exactly the "can't tell" case, not "zero").
            return (
                "# pgvector memory\n"
                "Active. No entries yet for this theme. Use the built-in "
                "`memory` tool to save durable notes — entries are mirrored "
                "to Postgres with embeddings for semantic recall across "
                "sessions."
            )
        total_desc = (
            f"{count_all} total across all themes" if count_all is not None
            else "an unknown number of entries total across all themes"
        )
        return (
            "# pgvector memory\n"
            f"Active. {count_scoped} entries for '{self._agent_identity}', "
            f"{total_desc}. "
            "Use `recall_memory(query, scope='all'|'<theme>')` for deeper / "
            "cross-theme recall beyond what's in the built-in memory block."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        try:
            return self._prefetch_impl(query, session_id=session_id)
        except Exception as exc:  # noqa: BLE001 -- invariant #4: never raise into the agent loop
            logger.debug("pgvector prefetch failed (ignored): %s", exc)
            return ""

    def _prefetch_impl(self, query: str, *, session_id: str = "") -> str:
        if not self._healthy or not self._store:
            return ""
        # H4 (v0.6.0): a prior queue_prefetch(query, session_id=...) may have
        # already computed and cached this turn's block. Consume it (once)
        # instead of paying for a synchronous embed + search here -- that is
        # the whole point of queue_prefetch existing.
        cached = self._pop_cached_prefetch(session_id)
        if cached is not None:
            return cached
        if not query:
            return ""
        # H4: hard wall-clock budget for the SYNCHRONOUS fallback (used only
        # when nothing was queued ahead of time). Must stay below the host's
        # 8s external-prefetch timeout (agent.memory_manager) -- passed to
        # embed() as its timeout, which is now ONE shared deadline across
        # embed_protocol='auto''s two HTTP attempts, not a fresh timeout for
        # each (see embed.py).
        budget = _as_float(self._config.get("prefetch_budget"), DEFAULTS["prefetch_budget"])
        deadline = time.monotonic() + budget
        return self._compute_prefetch_block(query, timeout=budget, deadline=deadline)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for `session_id`; the next prefetch()
        call for that session consumes the cached block (H4, v0.6.0).

        At most one worker thread runs at a time. A queue_prefetch() call
        while one is already in flight does not spawn a second thread --
        it replaces `_prefetch_pending`, so the worker picks up the NEWEST
        request once it finishes the one it's on. Never raises (invariant
        #4): scheduling failures are logged and swallowed; the worker loop
        below has its own independent safety net so a bad embed/search call
        cannot crash the (daemon) thread either.
        """
        try:
            if not self._healthy or not self._store or not query:
                return
            with self._prefetch_lock:
                self._prefetch_pending = (query, session_id, self._prefetch_generation)
                if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                    return  # a worker is already in flight; it will pick up this request next
                thread = spawn_context_thread(
                    target=self._prefetch_worker_loop, name="pgvector-queue-prefetch",
                )
                self._prefetch_thread = thread
                thread.start()
        except Exception as exc:  # noqa: BLE001 -- invariant #4
            logger.debug("pgvector queue_prefetch failed to schedule (ignored): %s", exc)

    def _prefetch_worker_loop(self) -> None:
        """Background worker (daemon thread): drain `_prefetch_pending` until
        empty, one request at a time. Never raises -- a broken embed/search
        call must not kill this thread (it would silently stop refreshing
        the cache for the rest of the process) or, worse, escape to the
        host's executor (which is not this thread, but the discipline is the
        same as every other fail-soft path in this file)."""
        while True:
            with self._prefetch_lock:
                pending = self._prefetch_pending
                self._prefetch_pending = None
            if pending is None:
                return
            query, session_id, generation = pending
            block = ""
            try:
                block = self._compute_prefetch_block(query, timeout=self._embed_timeout())
            except Exception as exc:  # noqa: BLE001 -- must not block shutdown or crash
                logger.debug("pgvector queue_prefetch worker failed (ignored): %s", exc)
            self._cache_prefetch(session_id, block, generation=generation)

    def _compute_prefetch_block(
        self, query: str, *, timeout: float, deadline: Optional[float] = None,
    ) -> str:
        """Shared embed + search + format step behind prefetch()'s
        synchronous fallback and queue_prefetch()'s background worker.

        `deadline` (a time.monotonic() timestamp), when given, enforces
        prefetch_budget as a hard wall-clock bound on the WHOLE step, not
        just the embed call: if the embed already ate the budget, the DB
        search is skipped rather than run past the deadline anyway.
        queue_prefetch()'s background worker passes no deadline -- it is not
        blocking anything, so it lets the search run once the embed
        succeeds.
        """
        store = self._store
        if store is None:
            return ""
        try:
            vec = _embed_with_config(query, self._config, timeout=timeout)
        except EmbeddingError as exc:
            logger.debug("pgvector prefetch embed failed: %s", exc)
            return ""
        if deadline is not None and time.monotonic() >= deadline:
            logger.debug("pgvector prefetch: embed consumed the budget; skipping search")
            return ""

        # Ambient prefetch is scoped to the current agent_identity by
        # default — keeps marketing turns from polluting trading recall.
        try:
            rows = store.search(
                query_embedding=vec,
                agent_identity=self._agent_identity,
                limit=_as_int(self._config.get("prefetch_limit"),
                              DEFAULTS["prefetch_limit"]),
                min_similarity=_as_float(self._config.get("min_similarity"),
                                         DEFAULTS["min_similarity"]),
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

    def _cache_prefetch(self, session_id: str, block: str, *, generation: int) -> None:
        """Store `block` for `session_id`, unless a newer initialize()/
        shutdown() has since bumped the generation counter (H4) -- a worker
        still finishing up from a session that has already ended must not
        write into the NEXT session's cache."""
        key = session_id or ""
        with self._prefetch_lock:
            if generation != self._prefetch_generation:
                return
            self._prefetch_cache[key] = block
            self._prefetch_cache.move_to_end(key)
            while len(self._prefetch_cache) > self._PREFETCH_CACHE_CAP:
                self._prefetch_cache.popitem(last=False)

    def _pop_cached_prefetch(self, session_id: str) -> Optional[str]:
        """Consume (pop) the cached block for `session_id`, or None if
        nothing is cached. Popping means each queued result is used at most
        once, matching upstream's "prefetch() consumes it next turn"."""
        key = session_id or ""
        with self._prefetch_lock:
            return self._prefetch_cache.pop(key, None)

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
        if not self._writes_enabled:  # v0.6.0 (M4)
            return
        if not _as_bool(self._config.get("sync_turns"), True):
            return

        sid = session_id or self._session_id or "default"
        try:
            min_chars = int(self._config.get("turn_min_chars", 40))
        except (TypeError, ValueError):
            min_chars = int(DEFAULTS["turn_min_chars"])
        policy = self._config.get("conversation_embed_policy", "all")
        psid = self._parent_session_id if self._delegation_enabled else None

        # Fail-soft (invariant #4): this is called on the agent's turn path,
        # so nothing here may raise into the agent loop.
        try:
            for role, content in (("user", user_content), ("assistant", assistant_content)):
                if not content:
                    continue
                if self._is_noise(content, min_chars=min_chars):
                    continue
                meta = {"session_id": sid}
                if self._raw_identity != self._agent_identity:
                    meta["raw_identity"] = self._raw_identity
                accepted = self._writer.enqueue(
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
                # Fingerprint ONLY on accept. enqueue() returns False when the
                # queue is full and the write is dropped; recording it anyway
                # would make on_session_end skip a turn that never landed,
                # turning a recoverable drop into permanent data loss.
                if accepted:
                    self._remember_turn(self._turn_fingerprint(role, content))
        except Exception as exc:  # noqa: BLE001
            logger.debug("pgvector sync_turn failed (ignored): %s", exc)

    @staticmethod
    def _strip_skill_scaffolding(content: str) -> str:
        """Normalize a user turn the way the host does before handing it to us.

        The host calls sync_turn() with `_strip_skill_scaffolding(user_content)`
        (agent/memory_manager.py:389 ->
        agent.skill_commands.extract_user_instruction_from_skill_message) but
        passes on_session_end() the RAW transcript. So a /skill turn reaches the
        two hooks as two different strings, hashes to two different
        fingerprints, and defeats the dedup -- it gets written twice, which is
        the exact bug the fingerprints exist to prevent.

        Guarded import, matching how this module already reaches for
        `agent.memory_provider` / `hermes_constants`: outside hermes-agent the
        content is simply used as-is.
        """
        try:
            from agent.skill_commands import (
                extract_user_instruction_from_skill_message as _strip,
            )
            return _strip(content) or content
        except Exception:  # noqa: BLE001 -- host internals are optional
            return content

    def _remember_turn(self, fingerprint: str) -> None:
        """Record a captured turn, enforcing the cap at the point of growth.

        The cap used to be checked only in on_session_switch(). On the gateway
        every session calls initialize() rather than on_session_switch(), and
        initialize() deliberately does not reset the set (clearing it would
        re-open the double-write on the compression path), so the set could
        grow without bound there. Checking here covers every path that can
        add to it.
        """
        if len(self._turn_fingerprints) >= self._FINGERPRINT_CAP:
            logger.debug(
                "pgvector turn-fingerprint set hit %d; resetting",
                self._FINGERPRINT_CAP,
            )
            self._turn_fingerprints = set()
        self._turn_fingerprints.add(fingerprint)

    @staticmethod
    def _turn_fingerprint(role: str, content: str) -> str:
        """Stable short digest of one captured turn.

        on_session_end() replays the whole message list, which overlaps the
        turns sync_turn() already enqueued during the session. `conversations`
        has no unique constraint (unlike memory_entries), so without this the
        same turn lands twice -- doubling the table, the embedding cost, and
        the rows recall_conversation returns.
        """
        import hashlib
        digest = hashlib.sha1(f"{role}:{content}".encode("utf-8", "replace"))
        return digest.hexdigest()[:16]

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
        if not self._writes_enabled:  # v0.6.0 (M4)
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
                        # v0.6.0 (M5): conversations.parent_session_id is the
                        # session that DELEGATED this row (migration 002's
                        # definition), not the child's -- the child session id
                        # stays in metadata.child_session_id (below) and in
                        # memory_agent_edges (the "edge" enqueue above). This
                        # used to write child_session_id here, the reverse.
                        "parent_session_id": self._parent_session_id,
                    },
                    metadata={"kind": "delegation", "child_session_id": child_session_id},
                )
        except Exception as exc:  # noqa: BLE001 — never escape into the agent loop
            logger.debug("pgvector on_delegation failed (ignored): %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Best-effort: capture session turns + bump last_seen. Fail-soft,
        non-blocking. No LLM summary (invariant #2).

        v0.6.0 (L9): the turn backstop below runs whenever sync_turns is on
        and the provider is healthy with a writer -- it no longer needs
        migration 002. Only the register_agent enqueue stays gated on
        self._delegation_enabled (memory_agents/last_seen tracking is a 002
        object). parent_session_id on the backstop's own writes is gated the
        same way sync_turn() gates it: self._parent_session_id only when 002
        is applied, else None -- writing it unconditionally would try to
        INSERT into a column that does not exist on a pre-002 schema.
        """
        if not self._healthy or not self._writer:
            return
        if not self._writes_enabled:  # v0.6.0 (M4)
            return
        try:
            if self._delegation_enabled:
                self._writer.enqueue(
                    action="register_agent",
                    agent_identity=self._agent_identity,
                    target="memory_agents",
                    content="",
                    extra={"kind": classify_kind(self._agent_identity)},
                    metadata={},
                )
            # Capture substantive user/assistant turns for conversation recall.
            # Backstop only: sync_turn() already captured this session's turns
            # per-exchange, and the host calls BOTH hooks (and calls this one
            # again on session rotation), so anything already fingerprinted is
            # skipped -- `conversations` has no unique constraint to catch it.
            # Honors the same sync_turns toggle sync_turn() does; without this
            # an operator who disabled turn capture still got turns persisted.
            if messages and _as_bool(self._config.get("sync_turns"), True):
                policy = self._config.get("conversation_embed_policy", "all")
                psid = self._parent_session_id if self._delegation_enabled else None
                try:
                    min_chars = int(self._config.get("turn_min_chars", 40))
                except (TypeError, ValueError):
                    min_chars = int(DEFAULTS["turn_min_chars"])
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
                    if self._is_noise(content, min_chars=min_chars):
                        continue
                    # Fingerprint the NORMALIZED form for user turns, so a
                    # /skill turn matches what sync_turn() was handed.
                    fp_content = (
                        self._strip_skill_scaffolding(content)
                        if role == "user" else content
                    )
                    fp = self._turn_fingerprint(role, fp_content)
                    if fp in self._turn_fingerprints:
                        continue  # already enqueued by sync_turn this session
                    accepted = self._writer.enqueue(
                        action="turn",
                        agent_identity=self._agent_identity,
                        target="conversations",
                        content=content[:8000],
                        extra={
                            "role": role,
                            "session_id": self._session_id or "default",
                            "embed": self._should_embed_turn(role, content, policy),
                            "parent_session_id": psid,
                        },
                        metadata={},
                    )
                    # Same contract as sync_turn: fingerprint ONLY on accept.
                    # enqueue() returns False on a full queue -- and a full
                    # queue is most likely exactly here, since this replays a
                    # whole transcript at once through a 256-slot queue. Marking
                    # a dropped turn as captured would make the NEXT
                    # on_session_end (session rotation replays the same list)
                    # skip it, turning a recoverable drop into permanent loss.
                    if accepted:
                        self._remember_turn(fp)
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
        if not self._writes_enabled:  # v0.6.0 (M4)
            return
        if target not in ("memory", "user"):
            logger.debug("pgvector ignoring unsupported target: %r", target)
            return
        if action not in ("add", "replace", "remove"):
            logger.debug("pgvector ignoring unknown action: %r", action)
            return
        # An add/replace with no content produces a row that can NEVER be
        # embedded: embed() raises EmbeddingError("empty input") on empty or
        # whitespace text. Such a row is retried by every nightly backfill
        # forever, always fails, and permanently prevents the NULL-embedding
        # count from reaching zero -- destroying the one signal an operator
        # watches. It also carries no information worth mirroring. `remove`
        # is exempt: it legitimately arrives with empty content and targets
        # the row via old_text.
        if action in ("add", "replace") and not (content or "").strip():
            # Not routine. on_memory_write only fires for writes the built-in
            # tool COMMITTED, and it rejects empty add/replace content -- so
            # arriving here means real content landed on disk and reached us
            # as ''. Skipping keeps an un-embeddable row out of the table, but
            # the mirror is now missing an entry the agent believes it saved,
            # which is worth more than a debug line.
            logger.warning(
                "pgvector: %r for target=%r arrived with empty content; "
                "skipping (the built-in store has it, the mirror will not)",
                action, target,
            )
            return

        meta = dict(metadata or {})
        meta.setdefault("session_id", self._session_id)
        if self._delegation_enabled and self._parent_session_id:
            meta.setdefault("parent_session_id", self._parent_session_id)
        if self._raw_identity != self._agent_identity:
            meta.setdefault("raw_identity", self._raw_identity)
        old_text = meta.get("old_text") or meta.get("replaces")
        # H2 (v0.6.0): metadata["previous_content"] is the EXACT prior entry
        # the built-in store edited under its own lock
        # (memory_manager.notify_memory_tool_write / MemoryProvider.on_memory_write's
        # docstring) -- not a caller hint like old_text, which "is not
        # authoritative identity" per that same docstring. It has to reach the
        # worker so replace()/remove() can match content = %s exactly instead
        # of old_text's substring LIKE (which picks the lowest-id match and
        # can hit an unrelated stale row). It must never be PERSISTED: it is
        # the full text of a row about to be overwritten/deleted, not
        # provenance for the new row -- so it is popped here regardless of
        # whether it turns out to be usable below.
        previous_content = meta.pop("previous_content", None)
        extra: Dict[str, Any] = {}
        if old_text:
            extra["old_text"] = str(old_text)
        if isinstance(previous_content, str) and previous_content.strip():
            extra["previous_content"] = previous_content

        self._writer.enqueue(
            action=action,
            agent_identity=self._agent_identity,
            target=target,
            content=content,
            extra=extra,
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
                # H2 (v0.6.0): previous_content, when present, is the exact
                # prior entry (from upstream's notify_memory_tool_write) --
                # match it precisely instead of old_text's substring LIKE,
                # which can hit a stale lower-id row that merely CONTAINS the
                # pattern rather than the row the built-in store actually
                # edited.
                previous_content = item.extra.get("previous_content")
                vec = self._maybe_embed(item.content)
                if previous_content:
                    n = self._store.replace(
                        agent_identity=item.agent_identity,
                        target=item.target,
                        new_content=item.content,
                        new_embedding=vec,
                        exact_content=previous_content,
                    )
                    if n == 0:
                        # Nothing matched exactly — degrade to add, same as
                        # the old_text path below (built-in wrote the new
                        # entry to disk; mirror it so we don't lose it).
                        self._store.add(
                            agent_identity=item.agent_identity,
                            target=item.target,
                            content=item.content,
                            embedding=vec,
                            metadata=item.metadata,
                        )
                elif old_text:
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
                    # No old_text/previous_content in metadata → can't locate
                    # prior row; add the new content so we don't lose it.
                    self._store.add(
                        agent_identity=item.agent_identity,
                        target=item.target,
                        content=item.content,
                        embedding=vec,
                        metadata=item.metadata,
                    )
            elif item.action == "remove":
                # H2 (v0.6.0): previous_content, when present, is the exact
                # prior entry -- match it precisely and never fall back to
                # old_text/LIKE if it matches nothing (a caller that has the
                # exact text and still gets no match has the wrong row
                # identity, not an ambiguous pattern to widen).
                previous_content = item.extra.get("previous_content")
                if previous_content:
                    n = self._store.remove(
                        agent_identity=item.agent_identity,
                        target=item.target,
                        exact_content=previous_content,
                    )
                    if n == 0:
                        logger.debug(
                            "pgvector remove: previous_content matched no row "
                            "for %s/%s (not falling back to old_text/LIKE)",
                            item.agent_identity, item.target,
                        )
                else:
                    # The removal target lives in extra["old_text"], NOT in
                    # content. The built-in tool's remove op takes old_text and
                    # leaves content empty (tools/memory_tool.py: remove ->
                    # store.remove(target, old_text)), and the host forwards
                    # old_text via METADATA (memory_manager.notify_memory_tool_write).
                    # Reading item.content here meant remove() was called with
                    # "", which becomes `content LIKE '%%'` -- matching every
                    # row and deleting the entire mirror for that
                    # (agent_identity, target).
                    old_text = item.extra.get("old_text") or item.content
                    if not (old_text or "").strip():
                        logger.warning(
                            "pgvector refusing remove with no old_text for %s/%s "
                            "(would match every row)",
                            item.agent_identity, item.target,
                        )
                    else:
                        self._store.remove(
                            agent_identity=item.agent_identity,
                            target=item.target,
                            old_text=old_text,
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
            # One-shot WARNING (mirrors the _embed_warned tripwire): _healthy is
            # evaluated once at initialize() and never re-probed, so a Postgres
            # restart after a healthy init silently discards every durable write
            # for the rest of the session. At debug level that is invisible at
            # the host's default log level -- total write loss, zero signal.
            if not self._db_warned:
                self._db_warned = True
                logger.warning(
                    "pgvector worker (%s/%s/%s) failed: %s -- further worker "
                    "failures this session are logged at debug level",
                    item.action,
                    item.agent_identity,
                    item.target,
                    str(exc)[:200],
                )
            else:
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
        if not _as_bool(self._config.get("embed_on_write"), True):
            return None
        config = self._config
        timeout = self._embed_timeout()
        def _fn(text: str):
            return _embed_with_config(text, config, timeout=timeout)
        return _fn

    # -- Tool surface --------------------------------------------------------

    def _restricted_identities(self) -> List[str]:
        """Sink themes this agent must not read out of.

        identity.py buckets direct-message traffic into `whatsapp-dm`,
        multi-party group/channel/thread traffic into `external-group`, and
        benchmark traffic into `_bench`. That bucketing is WRITE-side only: it
        strips PII from the *identity*, but the message bodies still land in
        `content`. Without a read-side gate, any theme could pull DM content
        (and discarded bench fixtures) into its context via scope='all' or by
        naming the bucket directly -- and, with turn capture on, the reply
        quoting it would be written back under the *reading* theme,
        permanently re-attributing DM data into a production theme.

        An agent that IS the bucket keeps full access to its own rows, so
        DM-scoped recall still works for the DM agent itself.
        """
        return [
            b for b in (DM_BUCKET, GROUP_BUCKET, BENCH_BUCKET)
            if b != self._agent_identity
        ]

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_MEMORY_SCHEMA, RECALL_CONVERSATION_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "recall_conversation":
            return self._handle_recall_conversation(args)
        if tool_name != "recall_memory":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._healthy or not self._store:
            return json.dumps({"results": [], "count": 0, "error": "pgvector unavailable"})

        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required arg: query")

        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        # Scope resolution: 'current' → my agent_identity; 'all' → no filter;
        # anything else → treat as explicit theme name.
        scope = str(args.get("scope") or self._config.get("scope_default") or "current").strip()
        restricted = self._restricted_identities()
        exclude: Optional[List[str]] = None
        if scope == "current":
            agent_filter: Optional[str] = self._agent_identity
        elif scope == "all":
            agent_filter = None
            # Cross-theme recall stays opt-in and broad, but never reaches the
            # PII/bench sinks -- see _restricted_identities().
            exclude = restricted or None
        elif scope == "session":
            # Only recall_conversation supports 'session'. Falling through
            # would silently filter on a literal 'session' theme and return
            # zero rows — indistinguishable from "nothing found" (v0.4.2).
            return tool_error(
                "scope='session' is only valid for recall_conversation; "
                "use 'current', 'all', or a theme name here."
            )
        elif scope in restricted:
            return tool_error(
                f"scope={scope!r} is a restricted sink (direct-message / bench "
                "traffic) and is not readable from this theme."
            )
        else:
            agent_filter = scope

        # Target resolution: 'memory'/'user'/'both'.
        target_arg = str(args.get("target") or "both").strip()
        target_filter: Optional[str] = None if target_arg == "both" else target_arg
        if target_filter not in (None, "memory", "user"):
            return tool_error(f"Invalid target: {target_arg!r}")

        hybrid = _as_bool(self._config.get("hybrid_search"), True)
        try:
            vec = _embed_with_config(query, self._config, timeout=self._embed_timeout())
        except EmbeddingError as exc:
            if not hybrid:
                return json.dumps({"results": [], "count": 0, "error": f"embed: {_safe_err(exc)}"})
            # Hybrid on: the query text still drives a full-text search without a
            # vector, so degrade to lexical recall instead of erroring.
            logger.debug("recall_memory embed failed; full-text-only: %s", exc)
            vec = None

        try:
            if hybrid:
                rows = self._store.hybrid_search(
                    query_text=query,
                    query_embedding=vec,
                    agent_identity=agent_filter,
                    target=target_filter,
                    limit=limit,
                    exclude_identities=exclude,
                )
            else:
                rows = self._store.search(
                    query_embedding=vec,
                    agent_identity=agent_filter,
                    target=target_filter,
                    limit=limit,
                    exclude_identities=exclude,
                )
        except Exception as exc:  # noqa: BLE001
            # Fail-soft: a hybrid hiccup with a usable vector falls back to the
            # proven pure-vector path rather than returning nothing.
            if hybrid and vec is not None:
                try:
                    rows = self._store.search(
                        query_embedding=vec,
                        agent_identity=agent_filter,
                        target=target_filter,
                        limit=limit,
                        exclude_identities=exclude,
                    )
                except Exception as exc2:  # noqa: BLE001
                    return json.dumps({"results": [], "count": 0, "error": f"db: {_safe_err(exc2)}"})
            else:
                return json.dumps({"results": [], "count": 0, "error": f"db: {_safe_err(exc)}"})

        results = []
        for r in rows:
            ts = r.get("updated_at") or r.get("created_at")
            score = r.get("score")
            entry = {
                "id": r.get("id"),
                "agent_identity": r.get("agent_identity"),
                "target": r.get("target"),
                "ts": ts.isoformat() if ts else None,
                # None stays None (v0.4.2): a full-text-only hybrid hit has no
                # cosine score; flattening it to 0.0 made a strong lexical
                # match look identical to an orthogonal vector match.
                "score": round(float(score), 4) if score is not None else None,
                "content": (r.get("content") or "")[:2000],
            }
            if r.get("rrf_score") is not None:
                entry["rrf_score"] = round(float(r["rrf_score"]), 6)
            results.append(entry)
        return json.dumps({"results": results, "count": len(results)})

    def _handle_recall_conversation(self, args: Dict[str, Any]) -> str:
        """Tool handler for recall_conversation over the conversations table."""
        if not self._healthy or not self._store:
            return json.dumps({"results": [], "count": 0, "error": "pgvector unavailable"})

        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("Missing required arg: query")
        try:
            limit = max(1, min(int(args.get("limit", 5)), 20))
        except (TypeError, ValueError):
            limit = 5

        scope = str(args.get("scope") or "current").strip()
        agent_filter: Optional[str] = None
        session_filter: Optional[str] = None
        restricted = self._restricted_identities()
        exclude: Optional[List[str]] = None
        if scope == "current":
            agent_filter = self._agent_identity
        elif scope == "session":
            session_filter = self._session_id or None
            # Close the gate's last branch: with no session id (never produced
            # by the current host, but the surrounding code already treats an
            # empty id as possible) this would otherwise be a completely
            # unfiltered cross-theme sweep INCLUDING both sinks.
            if session_filter is None:
                exclude = restricted or None
        elif scope == "all":
            # No agent filter, but never the PII/bench sinks.
            exclude = restricted or None
        elif scope in restricted:
            return tool_error(
                f"scope={scope!r} is a restricted sink (direct-message / bench "
                "traffic) and is not readable from this theme."
            )
        else:
            agent_filter = scope  # treat as a specific theme name

        hybrid = _as_bool(self._config.get("hybrid_search"), True)
        try:
            vec = _embed_with_config(query, self._config, timeout=self._embed_timeout())
        except EmbeddingError as exc:
            if not hybrid:
                return json.dumps({"results": [], "count": 0, "error": f"embed: {_safe_err(exc)}"})
            logger.debug("recall_conversation embed failed; full-text-only: %s", exc)
            vec = None

        try:
            if hybrid:
                rows = self._store.hybrid_search_turns(
                    query_text=query,
                    query_embedding=vec,
                    agent_identity=agent_filter,
                    session_id=session_filter,
                    limit=limit,
                    exclude_identities=exclude,
                )
            else:
                rows = self._store.search_turns(
                    query_embedding=vec,
                    agent_identity=agent_filter,
                    session_id=session_filter,
                    limit=limit,
                    exclude_identities=exclude,
                )
        except Exception as exc:  # noqa: BLE001
            if hybrid and vec is not None:
                try:
                    rows = self._store.search_turns(
                        query_embedding=vec,
                        agent_identity=agent_filter,
                        session_id=session_filter,
                        limit=limit,
                        exclude_identities=exclude,
                    )
                except Exception as exc2:  # noqa: BLE001
                    return json.dumps({"results": [], "count": 0, "error": f"db: {_safe_err(exc2)}"})
            else:
                return json.dumps({"results": [], "count": 0, "error": f"db: {_safe_err(exc)}"})

        results = []
        for r in rows:
            ts = r.get("ts")
            score = r.get("score")
            entry = {
                "id": r.get("id"),
                "agent_identity": r.get("agent_identity"),
                "session_id": r.get("session_id"),
                "role": r.get("role"),
                "ts": ts.isoformat() if ts else None,
                "score": round(float(score), 4) if score is not None else None,
                "content": (r.get("content") or "")[:2000],
            }
            if r.get("rrf_score") is not None:
                entry["rrf_score"] = round(float(r["rrf_score"]), 6)
            results.append(entry)
        return json.dumps({"results": results, "count": len(results)})

    # -- Setup hooks ---------------------------------------------------------

    def identity_signature(self) -> Dict[str, Any]:
        """Governance config that must bust a cached gateway agent when it
        changes. Upstream calls this on an UNINITIALIZED instance on every
        inbound message (agent.memory_provider.MemoryProvider.identity_signature),
        so it reads self._config only -- no I/O, no self._store, cheap and
        read-only. allowed_themes/identity_aliases/bench_mode decide which
        theme a raw identity normalizes to (identity.normalize_identity);
        write_contexts decides whether a context writes at all -- a cached
        agent built under the OLD values would keep routing/writing wrong
        until the process restarts, without this.
        """
        return {
            "pgvector.allowed_themes": _as_theme_list(self._config.get("allowed_themes")),
            "pgvector.identity_aliases": _as_alias_map(self._config.get("identity_aliases")),
            "pgvector.bench_mode": self._config.get("bench_mode", DEFAULTS["bench_mode"]),
            "pgvector.write_contexts": _as_write_contexts(self._config.get("write_contexts")),
        }

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "dsn",
                "description": "Postgres DSN (psycopg connection string)",
                "type": "text",
                "default": DEFAULTS["dsn"],
                "required": True,
            },
            {
                "key": "embed_url",
                "description": "Embedding endpoint base URL (OpenAI-compatible or Ollama native)",
                "type": "text",
                "default": DEFAULTS["embed_url"],
                "required": True,
            },
            {
                "key": "embed_model",
                "description": "Embedding model name (must return embed_dim-length vectors; 768 by default)",
                "type": "text",
                "default": DEFAULTS["embed_model"],
            },
            {
                "key": "embed_dim",
                "description": "Vector length the embed model returns. Must also match the database's vector(N) columns (768 as created by migration 001). Changing it on an existing database requires migrating those columns and re-embedding every row -- see README, 'Changing the embedding dimension'.",
                "type": "integer",
                "default": DEFAULTS["embed_dim"],
                "minimum": 1,
            },
            {
                "key": "embed_api_key_env",
                "description": "NAME of an environment variable holding a bearer token for the embed endpoint (e.g. OPENROUTER_API_KEY) -- never the token itself. Read at call time; when the variable is set and non-empty the plugin sends 'Authorization: Bearer <value>'. Empty = no Authorization header.",
                "type": "text",
                "default": "",
            },
            {
                "key": "embed_protocol",
                "description": "'auto' tries the OpenAI-compatible /v1/embeddings path, then falls back to Ollama-native /api/embed. 'openai' uses /v1/embeddings only, so auth and unknown-model errors surface as-is (use it for OpenRouter / OpenAI). 'ollama' uses /api/embed only. Unknown values fall back to 'auto' with a warning.",
                "type": "text",
                "default": DEFAULTS["embed_protocol"],
                "choices": ["auto", "openai", "ollama"],
            },
            {
                "key": "prefetch_limit",
                "description": "Max ambient recall results injected per turn",
                "type": "integer",
                "default": DEFAULTS["prefetch_limit"],
                "minimum": 1,
                "maximum": 50,
            },
            {
                "key": "min_similarity",
                "description": "Cosine similarity cutoff for ambient prefetch",
                "type": "number",
                "default": DEFAULTS["min_similarity"],
                "minimum": 0.0,
                "maximum": 1.0,
            },
            {
                "key": "embed_on_write",
                "description": "Compute embedding on each write; turn off for text-only mode",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "scope_default",
                "description": "Default scope for recall_memory when caller omits it",
                "type": "text",
                "default": DEFAULTS["scope_default"],
                "choices": ["current", "all"],
            },
            {
                "key": "hybrid_search",
                "description": "Fuse the HNSW vector ranking with a Postgres full-text ranking (Reciprocal Rank Fusion) in recall_memory / recall_conversation. Recovers exact-lexical hits cosine smooths away and text-only rows with NULL embeddings; degrades to pure vector on error and to full-text-only when the query fails to embed. Apply migration 003 for the GIN index (works without it, just slower).",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "write_queue_maxsize",
                "description": "Bounded async-writer queue size; full = newest writes drop with a warning",
                "type": "integer",
                "default": DEFAULTS["write_queue_maxsize"],
                "minimum": 1,
            },
            {
                "key": "bulk_sync_on_init",
                "description": "Import MEMORY.md / USER.md content from disk on agent init",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "sync_turns",
                "description": "Capture every substantive (user, assistant) turn pair into the conversations table",
                "type": "boolean",
                "default": True,
            },
            {
                "key": "turn_min_chars",
                "description": "Turns shorter than this (after strip) are treated as boilerplate and skipped",
                "type": "integer",
                "default": DEFAULTS["turn_min_chars"],
                "minimum": 0,
            },
            {
                "key": "allowed_themes",
                "description": "Identity governance: optional allow-list of theme names, as a comma-separated string (e.g. 'marketing,sales'). Empty/unset = allow any. When set, an unknown X-Hermes-Session-Key falls back to 'default' with a one-time warning (the whatsapp-dm / _bench / default sinks are always permitted).",
                "type": "text",
                "default": "",
            },
            {
                "key": "identity_aliases",
                "description": "Raw-identity remaps applied before normalization, as a comma-separated 'raw=canonical' string (e.g. 'mkt=marketing,sales-bot=sales'). A hand-edited config.yaml may instead use a YAML mapping -- both forms are accepted.",
                "type": "text",
                "default": "",
            },
            {
                "key": "bench_mode",
                "description": "How to handle benchmark identities (skill-bench, *-bench): 'bucket' isolates them to the '_bench' theme; 'reject' drops them to 'default'.",
                "type": "text",
                "default": DEFAULTS["bench_mode"],
                "choices": ["bucket", "reject"],
            },
            {
                "key": "conversation_embed_policy",
                "description": "Which captured turns get an embedding. 'all' (recall-safe; every turn that passed the noise filter) | 'substantive_only' (user turns + assistant turns >=120 chars; caps HNSW growth) | 'none' (text-only, conversation recall disabled).",
                "type": "text",
                "default": DEFAULTS["conversation_embed_policy"],
                "choices": ["all", "substantive_only", "none"],
            },
            {
                "key": "ttl_days",
                "description": "Advisory retention window for conversations (memory_entries are NEVER pruned). 0 = off. Pruning only ever happens when an operator runs `hermes-pgvector prune` — this value is the default for that command, never an automatic background delete.",
                "type": "integer",
                "default": DEFAULTS["ttl_days"],
                "minimum": 0,
            },
            {
                "key": "embed_timeout",
                "description": "Seconds to wait for an embedding on the AGENT thread (prefetch, recall_memory, recall_conversation) and during the init-time bulk import. Kept short on purpose: a timeout here degrades recall to full-text-only, which beats making the agent wait. Raise it only if recall quality matters more than latency on your endpoint.",
                "type": "number",
                "default": DEFAULTS["embed_timeout"],
                "minimum": 0.1,
            },
            {
                "key": "embed_write_timeout",
                "description": "Seconds to wait for an embedding on the BACKGROUND writer path. Nothing waits on this, and giving up costs a permanently unsearchable row (recoverable only by `hermes-pgvector backfill`), so it is far more generous than embed_timeout. Raise it if your endpoint is slow: writes that time out land with a NULL embedding.",
                "type": "number",
                "default": DEFAULTS["embed_write_timeout"],
                "minimum": 0.1,
            },
            {
                "key": "embed_write_retries",
                "description": "Bounded embed retries on the background writer path ONLY (the hot path — prefetch/recall/sync — always uses a single attempt). Durable recovery of missed embeddings is the `hermes-pgvector backfill` sweep, not inline retries.",
                "type": "integer",
                "default": DEFAULTS["embed_write_retries"],
                "minimum": 0,
            },
            {
                "key": "embed_write_backoff",
                "description": "Base seconds for the exponential backoff between embed_write_retries attempts on the background writer path (backoff * 2**attempt_index).",
                "type": "number",
                "default": DEFAULTS["embed_write_backoff"],
                "minimum": 0.0,
            },
            {
                "key": "write_contexts",
                "description": "Comma-separated agent_context values (from initialize()'s kwargs) allowed to WRITE, e.g. 'primary,cron'. Contexts not listed skip on_memory_write / sync_turn / on_session_end / on_delegation (recall still works). A missing agent_context kwarg is treated as 'primary'.",
                "type": "text",
                "default": DEFAULTS["write_contexts"],
            },
            {
                "key": "shutdown_drain_timeout",
                "description": "Seconds shutdown() waits for the writer queue to drain. While draining, embeds are skipped (rows land text-only; `hermes-pgvector backfill` heals them later) so the drain stays DB-only and fast.",
                "type": "number",
                "default": DEFAULTS["shutdown_drain_timeout"],
                "minimum": 0.0,
            },
            {
                "key": "prefetch_budget",
                "description": "Hard wall-clock budget (seconds) for prefetch()'s synchronous fallback path (embed + search), used when queue_prefetch() has not already cached a block for the session. Must stay below the host's external-provider prefetch timeout (8s), or the host logs a timeout warning and skips this provider on later turns until the stuck call returns.",
                "type": "number",
                "default": DEFAULTS["prefetch_budget"],
                "minimum": 0.1,
                "maximum": 7.5,
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
            # Merge, don't replace: `values` only carries keys declared by
            # get_config_schema(), so a wholesale assignment silently deletes
            # live hand-edited keys that are read at runtime but not declared
            # (identity_aliases, embed_write_backoff).
            current = existing["plugins"].get("pgvector") or {}
            existing["plugins"]["pgvector"] = {**current, **values}
            with open(config_path, "w", encoding="utf-8") as fh:
                yaml.dump(existing, fh, default_flow_style=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pgvector save_config failed: %s", exc)

    # -- Helpers -------------------------------------------------------------

    def _embed_timeout(self) -> float:
        """Timeout for embeds on the agent thread (prefetch, recall tools) and
        on the init-time bulk import.

        Deliberately NOT the writer's timeout. A slow embed here degrades
        recall to full-text-only, which is a good outcome; blocking the agent
        longer is a worse one. The init-time bulk import shares it because it
        runs before the first turn and would otherwise stall session start.
        """
        return _as_float(self._config.get("embed_timeout"), DEFAULTS["embed_timeout"])

    def _maybe_embed(self, content: str) -> Optional[List[float]]:
        if not _as_bool(self._config.get("embed_on_write"), True):
            return None
        if self._draining:
            # v0.6.0 (M3): shutdown() is draining the writer queue -- skip the
            # embed endpoint entirely so the drain is DB-only and fast. Rows
            # land text-only (embedding=NULL); `hermes-pgvector backfill`
            # heals them later. Deliberately checked here (not just in
            # shutdown()) so every _worker call site that embeds is covered
            # by one guard.
            return None
        _write_timeout = _as_float(self._config.get("embed_write_timeout"),
                                   DEFAULTS["embed_write_timeout"])
        try:
            # This runs ONLY in the background AsyncWriter drain thread, so a
            # bounded retry here is safe (it never blocks the agent loop). The
            # hot-path callers (prefetch / recall tools / sync_turn) call
            # _embed_with_config() with the default retries=0.
            return _embed_with_config(
                content,
                self._config,
                timeout=_write_timeout,
                # Bound the WORST case, not just each attempt: retries x the
                # two protocol paths would otherwise multiply a slow endpoint
                # into minutes of drain-thread block, filling the queue.
                max_total=_write_timeout * 2,
                retries=_as_int(self._config.get("embed_write_retries"),
                                DEFAULTS["embed_write_retries"]),
                backoff=_as_float(self._config.get("embed_write_backoff"),
                                  DEFAULTS["embed_write_backoff"]),
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

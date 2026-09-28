# hermes-memory-pgvector

**Postgres + pgvector memory provider for [hermes-agent](https://github.com/NousResearch/hermes-agent).** A shared memory substrate for a fleet of cooperating hermes-agent minions — built on Postgres and a single embedding endpoint you probably already run, with no LLM in the memory hot path.

```text
each minion → X-Hermes-Session-Key: <theme>
            → hermes-agent gateway
            → pgvector plugin
                ├── memory_entries  (mirrors built-in MEMORY.md / USER.md per theme)
                └── conversations   (every substantive turn, semantically searchable)
```

## Why it exists

Existing memory providers each solve a piece of the problem; the gap for **fleet deployments** is wide:

- **Built-in `memory` tool** persists to per-host `MEMORY.md` / `USER.md`. Two minions on the same host stomp on each other; minions on different hosts have no shared substrate.
- **Honcho** offers cross-session user modelling but requires a full external service, an LLM in the memory hot path for its deriver + dialectic loops, and its own ontology layered on top of the built-in tool. In high-concurrency fleet use it produces retry storms, embedding-endpoint queue backups, and gateway↔Honcho circular dependencies.
- **Holographic** is a fine in-process fact store but uses SQLite — a poor fit for many minions writing concurrently from many hosts.
- **Other providers** (Mem0, Hindsight, OpenViking, ByteRover, RetainDB, Supermemory) all either require a paid cloud, require LLM mediation for memory ops, or both.

What was missing: a **storage layer** that gives the built-in `memory` model durable, multi-tenant, semantically-searchable backing, with no LLM in the hot path, scoped cleanly per-minion so a marketing agent's notes don't pollute a trading agent's recall. That's what this plugin provides.

## Design philosophy

1. **Storage layer, not a memory model.** The agent keeps using `memory(action='add', target='memory'|'user', …)`. We mirror those writes via `on_memory_write`. No new ontology for the agent to learn.
2. **No LLM in the memory hot path.** Embeddings are vector math, not LLM calls. There is no deriver, no dialectic, no dream cycle.
3. **Per-agent themes by default, cross-theme recall on explicit demand.** Every row carries `agent_identity`. Recall is scoped to the current theme unless the agent asks for `scope='all'`.
4. **Fail-soft everywhere.** Embed endpoint down → degrade to text-only writes. Async writer queue full → drop with a one-time warning. DB down → log + skip. No exception escapes into the agent loop.
5. **Admin/runtime separation.** DDL runs once with superuser via `hermes-pgvector migrate`; the runtime role gets DML only on the migrated schema.

## What it does

| Hook / surface | Behavior |
|---|---|
| `initialize()` | Verifies schema, opens a `psycopg_pool.ConnectionPool`, bulk-imports existing `MEMORY.md` + `USER.md` content. |
| `on_memory_write(action, target, content, meta)` | Mirrors built-in `memory` writes into `memory_entries` (add / replace / remove). |
| `sync_turn(user, assistant, session_id)` / `on_session_end(messages)` | Captures every substantive chat turn into `conversations`; `on_session_end` is a dedup'd backstop. |
| `on_delegation(task, result, …)` | Records parent→child delegation provenance (`memory_agent_edges`) and stores the exchange as a recallable turn. |
| `prefetch(query)` / `queue_prefetch(query, session_id=...)` | Ambient recall: top-K similar `memory_entries` for the current theme, injected into the system prompt. `queue_prefetch` pre-computes it off the agent thread. |
| `recall_memory(query, scope, target, limit)` tool | Explicit cross-theme search of durable memory entries. |
| `recall_conversation(query, scope, limit)` tool | Explicit search over past chat turns. `scope ∈ {current, session, all, <theme>}`. |

Operator maintenance CLI: `hermes-pgvector migrate · stats · backfill · prune · cleanup · remap` (destructive commands default to dry-run) — see [docs/operations.md](docs/operations.md).

## Install

### Option 1: pip + discovery shim (recommended)

```bash
pip install hermes-memory-pgvector
hermes-pgvector install          # writes $HERMES_HOME/plugins/pgvector/ (older hosts only)

# Apply ALL migrations (schema + attribution + FTS + runtime grants + md5 unique index)
hermes-pgvector migrate --admin-dsn \
    "dbname=<your-memory-db> user=postgres host=/var/run/postgresql"
# add --runtime-role NAME if your runtime role isn't `hermes`

hermes config set memory.provider pgvector
sudo systemctl restart hermes.service
hermes memory status              # expect: Provider: pgvector; Status: available
```

**Why the shim?** hermes-agent resolves a provider from bundled dirs, then `$HERMES_HOME/plugins/<name>/`, then the `hermes_agent.memory_providers` pip entry-point group (declared by this package since v0.5.0). On a host that reads the entry point, `pip install` alone is enough and `install` is a no-op you can skip; the shim remains for older hosts that only scan directories. A shim *directory* always wins over the entry point when one is present.

### Option 2: clone + run the installer script (from source)

```bash
git clone https://github.com/andreab67/hermes-memory-pgvector.git
cd hermes-memory-pgvector
./scripts/install.sh
```

Installs the pinned Python deps, copies `hermes_pgvector/` into `$HERMES_HOME/plugins/pgvector/`, and prints the migrate/activate commands.

### Option 3: manual

```bash
# Python deps
pip install 'psycopg[binary]>=3.3.6,<4' 'psycopg-pool>=3.3.3,<4' 'PyYAML>=6.0,<7'

# Plugin module
mkdir -p ~/.hermes/plugins
cp -r hermes_pgvector ~/.hermes/plugins/pgvector
```

Then apply migrations and activate as in Option 1 above (`hermes-pgvector migrate --admin-dsn ...`, or `psql -f` each `hermes_pgvector/migrations/*.sql` in order — see [docs/operations.md](docs/operations.md) for the full admin walkthrough and [docs/upgrading.md](docs/upgrading.md) if you're coming from an older version).

## Configuration

Lives in `$HERMES_HOME/config.yaml` under `plugins.pgvector`, every key optional:

```yaml
plugins:
  pgvector:
    dsn: "dbname=hermes_memory user=hermes host=/var/run/postgresql"
    embed_url: "http://localhost:11434"
    embed_model: "nomic-embed-text"
    allowed_themes: "marketing,sales,trading"   # empty/unset = allow any theme
```

Full reference for every key (type, default, since-version, effect): [docs/configuration.md](docs/configuration.md). Scaling guidance (HNSW tuning, pool sizing, embed timeouts): [docs/scaling.md](docs/scaling.md).

## Multi-agent / per-minion themes

Each systemd-run minion sets one header on its OpenAI client; everything else flows automatically:

```python
client = AsyncOpenAI(
    base_url="http://127.0.0.1:8642/v1",
    api_key=API_KEY,
    default_headers={"X-Hermes-Session-Key": "marketing"},   # ← theme
)
```

The gateway plumbs `X-Hermes-Session-Key` through as `gateway_session_key=…` in `MemoryProvider.initialize` kwargs, taking priority over the profile default so unprofiled API traffic doesn't collapse every minion into one shared `default` scope.

Convention: lowercase, dash-separated, stable (identities are case-folded automatically — `Marketing` and `marketing` are the same theme). Governed sinks that always exist regardless of your themes: `whatsapp-dm` (collapsed DM/session keys), `external-group` (collapsed group/channel/thread keys), `_bench` (benchmark traffic), `default` (last resort). Set `plugins.pgvector.allowed_themes` to your product/worker list to enforce an allow-list — an unknown or typo'd header then falls back to `default` (one-time warning) instead of silently minting a new theme.

## Compatibility policy

This plugin follows semantic versioning on its **public surface**, stable as
of `1.0.0`:

- `plugins.pgvector.*` config keys (see [docs/configuration.md](docs/configuration.md))
- Tool names and parameters (`recall_memory`, `recall_conversation`)
- CLI commands, flags, and exit codes (`hermes-pgvector ...`)
- Database table/column names
- The `pgvector` provider name and the package's pip entry point

Within the `1.x` series this surface is guaranteed stable: it may grow (new
config keys, new optional params), but nothing already listed is renamed or
removed, and no default changes behaviour, without a `2.0` release.

`MemoryStore` and every other Python class/module are **internal** — not covered by semver, safe to change between minor releases, and excluded from the `1.x` guarantee above. A migration file, once released, is never edited in place; schema changes land as a new numbered migration.

## Support matrix

Tested in CI on every push/PR:

| Component | Versions |
|---|---|
| Python | 3.11, 3.12, 3.13 |
| PostgreSQL | 16, 17, 18 |
| pgvector extension | `>= 0.5.0` required; CI uses the version bundled in the `pgvector/pgvector:pg1x` images |
| hermes-agent | Conformance-tested against the pinned ref in [`conformance/HERMES_AGENT_REF`](conformance/HERMES_AGENT_REF), plus a non-blocking weekly drift check against upstream `main` |

## Docs

- [docs/configuration.md](docs/configuration.md) — every config key
- [docs/operations.md](docs/operations.md) — migrate, backfill, prune, cleanup, remap, stats, alerting
- [docs/scaling.md](docs/scaling.md) — HNSW tuning, pool sizing, embed latency budgets
- [docs/troubleshooting.md](docs/troubleshooting.md) — common failures and fixes
- [docs/upgrading.md](docs/upgrading.md) — version-to-version upgrade procedures
- [docs/release/RELEASING.md](docs/release/RELEASING.md) — maintainer release checklist
- [CHANGELOG.md](CHANGELOG.md) — every release, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format
- [ROADMAP.md](ROADMAP.md) — milestones, what shipped where, what's deliberately not planned
- [SECURITY.md](SECURITY.md) — supported versions, how to report a vulnerability

## Tests

```bash
pip install -e ".[test]"
pytest -q                                    # no DB/embed endpoint: everything skips gracefully

# Live mode (against a throwaway Postgres + your embed endpoint)
export PG_TEST_DSN='dbname=hermes_test user=postgres host=/var/run/postgresql'
export PG_TEST_EMBED_URL='http://your-embed-endpoint:11434'
pytest -q
```

## Rollback

```bash
hermes config set memory.provider none
sudo systemctl restart hermes.service

# Optional — drop every plugin-owned object (data loss, irreversible). Their
# sequences and indexes are dropped automatically with the tables; DROP VIEW
# must run before the tables it selects from.
sudo -u postgres psql -d <your-memory-db> -c "
DROP VIEW  IF EXISTS v_agent_memory;
DROP TABLE IF EXISTS memory_maintenance_log;
DROP TABLE IF EXISTS memory_agent_edges;
DROP TABLE IF EXISTS memory_agents;
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS memory_entries;
"

# Optional — remove the plugin files
rm -rf ~/.hermes/plugins/pgvector
```

## Why a standalone plugin (not an upstream PR)?

Per the hermes-agent [`CONTRIBUTING.md`](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md): the set of built-in memory providers is closed, and new backends are expected to ship as standalone plugins installed into `~/.hermes/plugins/` or via a pip entry point — exactly what this package does.

## Contributing

Bug reports + PRs welcome. Open an issue describing the failure mode + your environment (hermes-agent version, Postgres version, embed endpoint), or a PR with a focused change + test.

## License

[BSD 3-Clause](LICENSE) © 2026 Green Yoga Inc

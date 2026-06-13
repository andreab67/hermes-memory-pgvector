"""pgvector CLI — operator maintenance for the hermes-memory-pgvector plugin.

Runs standalone (no hermes-agent runtime needed), so it is safe in cron:

    python -m pgvector migrate  --admin-dsn "dbname=hermes_memory user=postgres host=/var/run/postgresql"
    python -m pgvector stats    [--dsn ...]
    python -m pgvector backfill [--dsn ...] [--embed-url ...] [--batch-size 100] [--dry-run]
    python -m pgvector prune    --days 90 [--dsn ...] [--execute]
    python -m pgvector cleanup  --identities "agent:main:whatsapp:dm:17192714834,skill-bench,skill-bench-ws" [--execute]
    python -m pgvector remap    --old hermes --new agent-hermes [--execute] [--force]

Destructive commands (prune/cleanup/remap) DEFAULT TO DRY-RUN. Pass --execute
to actually mutate. Every destructive run that mutates is recorded in
memory_maintenance_log.

Connection + embed settings resolve in this order: explicit CLI flag >
plugins.pgvector in --config <config.yaml> > built-in DEFAULTS.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import DEFAULTS
from .embed import embed
from .store import MemoryStore


def _load_config_file(path: Optional[str]) -> dict:
    if not path:
        return {}
    try:
        import yaml
        with open(path, encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh) or {}
        plugins = (data.get("plugins") or {})
        return plugins.get("pgvector") or {}
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not read config {path}: {exc}", file=sys.stderr)
        return {}


def _resolve(args, file_cfg: dict, key: str):
    """CLI flag > config-file value > DEFAULTS[key]."""
    val = getattr(args, key, None)
    if val is not None:
        return val
    if key in file_cfg and file_cfg[key] is not None:
        return file_cfg[key]
    return DEFAULTS.get(key)


def _make_store(args, file_cfg: dict) -> MemoryStore:
    return MemoryStore(_resolve(args, file_cfg, "dsn"))


def _make_embed_fn(args, file_cfg: dict):
    base_url = _resolve(args, file_cfg, "embed_url")
    model = _resolve(args, file_cfg, "embed_model")
    return lambda text: embed(text, base_url=base_url, model=model, retries=1)


# --- commands -------------------------------------------------------------

def cmd_migrate(args) -> int:
    store = MemoryStore("dbname=postgres")  # dsn unused; admin path opens its own conn
    applied = store.apply_all_migrations(admin_dsn=args.admin_dsn)
    print(f"applied migrations: {', '.join(applied) if applied else '(none found)'}")
    print(f"migration 002 present: {store.ensure_migration_002_applied()}")
    return 0


def cmd_stats(args) -> int:
    file_cfg = _load_config_file(args.config)
    store = _make_store(args, file_cfg)
    health = store.health()
    print(f"health: {json.dumps(health)}")
    mem = store.count()
    conv = store.count_turns()
    print(f"memory_entries: {mem} rows")
    print(f"conversations:  {conv} rows")
    # null-embedding counts (dry-run backfill returns remaining-null per table)
    nulls = store.backfill_null_embeddings(embed_fn=lambda t: [0.0] * 768, dry_run=True)
    for table, info in nulls.items():
        print(f"  {table}: {info['remaining']} null-embedding rows")
    if store.ensure_migration_002_applied():
        print("migration 002: applied — per-agent attribution:")
        for row in store.agent_attribution():
            print(
                f"  {row['agent_identity']:<36} kind={row.get('kind') or '-':<8} "
                f"memory={row.get('memory_rows', 0):<5} conv={row.get('conversation_rows', 0)}"
            )
    else:
        print("migration 002: NOT applied (agent attribution unavailable)")
    return 0


def cmd_backfill(args) -> int:
    file_cfg = _load_config_file(args.config)
    store = _make_store(args, file_cfg)
    tables = tuple(args.tables.split(",")) if args.tables else None
    embed_fn = _make_embed_fn(args, file_cfg)
    result = store.backfill_null_embeddings(
        embed_fn=embed_fn, tables=tables, batch_size=args.batch_size, dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        for table, info in result.items():
            store.log_maintenance(
                operation="backfill", target_table=table,
                affected_count=info.get("succeeded"), dry_run=False, details=info,
            )
    return 0


def cmd_prune(args) -> int:
    file_cfg = _load_config_file(args.config)
    store = _make_store(args, file_cfg)
    days = args.days if args.days is not None else int(_resolve(args, file_cfg, "ttl_days") or 0)
    dry_run = not args.execute  # destructive -> default dry-run, like cleanup/remap
    n = store.prune_conversations(older_than_days=days, dry_run=dry_run)
    verb = "would delete" if dry_run else "deleted"
    print(f"prune (>{days}d, dry_run={dry_run}): {verb} {n} conversation rows")
    if not dry_run and days > 0:
        store.log_maintenance(operation="prune", target_table="conversations",
                              affected_count=n, dry_run=False, details={"days": days})
    return 0


def cmd_cleanup(args) -> int:
    file_cfg = _load_config_file(args.config)
    store = _make_store(args, file_cfg)
    identities = [s.strip() for s in args.identities.split(",") if s.strip()]
    tables = tuple(args.tables.split(",")) if args.tables else None
    dry_run = not args.execute
    print(f"cleanup identities={identities} dry_run={dry_run}")
    # PII content scan first (a phone number may live in row CONTENT, not just identity)
    pii = store.scan_pii()
    print(f"PII content scan (\\d{{10,11}}): {json.dumps(pii)}")
    result = store.delete_by_identity(identities=identities, tables=tables, dry_run=dry_run)
    verb = "would delete" if dry_run else "deleted"
    print(f"{verb}: {json.dumps(result)}")
    if not dry_run:
        for table, cnt in result.items():
            store.log_maintenance(
                operation="cleanup_delete", target_table=table,
                identity_pattern=",".join(identities), affected_count=cnt,
                dry_run=False, details={"pii_scan": pii},
            )
    return 0


def cmd_remap(args) -> int:
    file_cfg = _load_config_file(args.config)
    store = _make_store(args, file_cfg)
    dry_run = not args.execute
    result = store.remap_identity(
        old_identity=args.old, new_identity=args.new, dry_run=dry_run, force=args.force,
    )
    print(json.dumps(result, indent=2))
    if not dry_run:
        store.log_maintenance(
            operation="remap_identity", identity_pattern=f"{args.old}->{args.new}",
            affected_count=result.get("memory_entries", {}).get("moved"),
            dropped_dupes=result.get("memory_entries", {}).get("dropped_duplicates"),
            dry_run=False, details=result,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pgvector", description="hermes-memory-pgvector maintenance CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--dsn", default=None, help="Postgres DSN (runtime hermes role)")
        sp.add_argument("--config", default=None, help="config.yaml to read plugins.pgvector from")

    m = sub.add_parser("migrate", help="apply all migrations as DB admin")
    m.add_argument("--admin-dsn", required=True, help="superuser/owner DSN (CREATE/GRANT)")
    m.set_defaults(func=cmd_migrate)

    s = sub.add_parser("stats", help="row counts, null-embedding counts, per-agent attribution")
    add_common(s)
    s.set_defaults(func=cmd_stats)

    b = sub.add_parser("backfill", help="re-embed rows with NULL embeddings")
    add_common(b)
    b.add_argument("--embed-url", default=None)
    b.add_argument("--embed-model", default=None)
    b.add_argument("--tables", default=None, help="comma list (default: memory_entries,conversations)")
    b.add_argument("--batch-size", type=int, default=100)
    b.add_argument("--dry-run", action="store_true", help="count nulls only, embed nothing")
    b.set_defaults(func=cmd_backfill)

    pr = sub.add_parser("prune", help="delete conversations older than N days (memory_entries never pruned)")
    add_common(pr)
    pr.add_argument("--days", type=int, default=None, help="age threshold; defaults to config ttl_days")
    pr.add_argument("--execute", action="store_true", help="actually delete (default is dry-run)")
    pr.set_defaults(func=cmd_prune)

    c = sub.add_parser("cleanup", help="delete all rows for given agent identities (DRY-RUN unless --execute)")
    add_common(c)
    c.add_argument("--identities", required=True, help="comma list of agent_identity values to purge")
    c.add_argument("--tables", default=None, help="comma list (default: memory_entries,conversations)")
    c.add_argument("--execute", action="store_true", help="actually delete (default is dry-run)")
    c.set_defaults(func=cmd_cleanup)

    rm = sub.add_parser("remap", help="merge one agent_identity into another (DRY-RUN unless --execute)")
    add_common(rm)
    rm.add_argument("--old", required=True)
    rm.add_argument("--new", required=True)
    rm.add_argument("--execute", action="store_true")
    rm.add_argument("--force", action="store_true", help="proceed even if >10 duplicates would be dropped")
    rm.set_defaults(func=cmd_remap)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

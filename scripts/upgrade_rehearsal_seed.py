#!/usr/bin/env python3
"""upgrade_rehearsal_seed.py -- seed a fresh database through the OLD
`hermes_pgvector.PgvectorMemoryProvider` for scripts/upgrade_rehearsal.sh
(Gate G1.6, docs/release/PLAN-1.0.md Sec 6 item 6).

Run this with the OLD (pre-upgrade, e.g. 0.5.5) venv's interpreter, against
a database that already has that version's migrations applied
(`hermes-pgvector migrate`). It drives the provider the way hermes-agent
would -- initialize() / on_memory_write() / sync_turn() / on_delegation() /
shutdown() -- standalone, with no hermes-agent installed: when
`agent.memory_provider` is not importable, `hermes_pgvector.MemoryProvider`
falls back to plain `object`, so `PgvectorMemoryProvider` is directly
instantiable outside the agent runtime.

Seeds three identities' worth of data, matching the upgrade scenario the
rehearsal exists to exercise:

  - "Marketing"  (mixed-case, RAW). Pre-0.6.0, `normalize_identity()` only
    strips whitespace -- it does not lowercase (that's an 0.6.0 change) --
    so this is exactly the kind of pre-existing mixed-case theme the
    upgraded `hermes-pgvector remap` step must be able to fix.
  - "marketing"  (already-lowercase) -- the eventual remap target; seeded
    with its own distinct content so remap has nothing to de-duplicate.
  - a second "marketing" provider instance pointed at an EMBED endpoint
    nothing listens on, so its writes land with embedding IS NULL
    (text-only) -- the rows `hermes-pgvector backfill` on the upgraded
    side must heal.

Every write goes through the public provider hooks (on_memory_write,
sync_turn, on_delegation); nothing here talks to the store or the database
directly. One of the memory adds is repeated verbatim (an exact duplicate)
to prove the pre-upgrade dedupe already worked, before the rehearsal
re-proves the same thing through the new code.

Env (all required):
  SEED_DSN               runtime DSN (user=hermes) for the seeded database
  SEED_EMBED_URL          a working embed endpoint (the fake embed server)
  SEED_NULL_EMBED_URL     an endpoint nothing listens on (e.g. http://127.0.0.1:1)
  SEED_HERMES_HOME_BASE   base tmp dir; one empty subdir per provider
                          instance is created under it, so bulk_sync_on_init
                          finds no MEMORY.md/USER.md and is a safe no-op

Exits 0 on success. All progress goes to stderr so a caller can reserve
stdout for its own output.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from hermes_pgvector import PgvectorMemoryProvider


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _make_provider(*, dsn: str, embed_url: str) -> PgvectorMemoryProvider:
    config = {
        "dsn": dsn,
        "embed_url": embed_url,
        "embed_dim": 768,
        "embed_protocol": "auto",
    }
    return PgvectorMemoryProvider(config)


def _seed_marketing_mixed_case(dsn: str, embed_url: str, hermes_home: Path) -> None:
    """"Marketing" -- mixed-case identity, working embed endpoint."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    provider = _make_provider(dsn=dsn, embed_url=embed_url)
    provider.initialize(
        "rehearsal-marketing-session",
        agent_identity="Marketing",
        hermes_home=str(hermes_home),
    )
    try:
        provider.on_memory_write(
            "add", "memory",
            "Q3 campaign budget increased to $50k for the enterprise segment kickoff.",
        )
        dup_content = (
            "Schedule social posts every Tuesday and Thursday for the product launch."
        )
        provider.on_memory_write("add", "memory", dup_content)
        provider.on_memory_write("add", "memory", dup_content)  # exact duplicate -> no-op
        provider.sync_turn(
            "What is the status of the Q3 marketing campaign rollout across channels?",
            "The Q3 campaign is on track; budget was increased to fifty thousand dollars.",
            session_id="rehearsal-marketing-session",
        )
        provider.on_delegation(
            "Draft a social calendar for next week across all channels.",
            "Delegation complete: a seven day social calendar was drafted and saved.",
            child_session_id="rehearsal-marketing-child-1",
        )
    finally:
        provider.shutdown()


def _seed_marketing_lowercase(dsn: str, embed_url: str, hermes_home: Path) -> None:
    """"marketing" -- already-lowercase identity, working embed endpoint."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    provider = _make_provider(dsn=dsn, embed_url=embed_url)
    provider.initialize(
        "rehearsal-marketing-lower-session",
        agent_identity="marketing",
        hermes_home=str(hermes_home),
    )
    try:
        provider.on_memory_write(
            "add", "memory",
            "Lower-case theme baseline entry about newsletter subscriber growth trends.",
        )
        provider.on_memory_write(
            "add", "memory",
            "Lower-case theme second entry: affiliate program performance summary for Q3.",
        )
        provider.sync_turn(
            "Can you summarize the lower theme's newsletter growth for the past quarter?",
            "Newsletter growth improved steadily and affiliate performance held stable too.",
            session_id="rehearsal-marketing-lower-session",
        )
    finally:
        provider.shutdown()


def _seed_marketing_null_embeddings(dsn: str, null_embed_url: str, hermes_home: Path) -> None:
    """"marketing" again, but the embed endpoint is unreachable -- writes land
    text-only (embedding IS NULL), which is what `backfill` must heal."""
    hermes_home.mkdir(parents=True, exist_ok=True)
    provider = _make_provider(dsn=dsn, embed_url=null_embed_url)
    provider.initialize(
        "rehearsal-marketing-null-session",
        agent_identity="marketing",
        hermes_home=str(hermes_home),
    )
    try:
        provider.on_memory_write(
            "add", "memory",
            "Null-embedding probe row: this text should land without a vector due to embed outage.",
        )
        provider.sync_turn(
            "This turn should also land with a null embedding because the embed "
            "endpoint is unreachable right now.",
            "Understood, this reply likewise lands without an embedding due to the outage.",
            session_id="rehearsal-marketing-null-session",
        )
    finally:
        provider.shutdown()


def main() -> int:
    dsn = os.environ["SEED_DSN"]
    embed_url = os.environ["SEED_EMBED_URL"]
    null_embed_url = os.environ["SEED_NULL_EMBED_URL"]
    base = Path(os.environ["SEED_HERMES_HOME_BASE"])

    _log("==> seeding identity 'Marketing' (mixed-case, working embed endpoint)")
    _seed_marketing_mixed_case(dsn, embed_url, base / "home-marketing")

    _log("==> seeding identity 'marketing' (lowercase, working embed endpoint)")
    _seed_marketing_lowercase(dsn, embed_url, base / "home-marketing-lower")

    _log(
        "==> seeding identity 'marketing' (lowercase, UNREACHABLE embed endpoint "
        "-> NULL embeddings)"
    )
    _seed_marketing_null_embeddings(dsn, null_embed_url, base / "home-marketing-null")

    _log("==> seeding complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

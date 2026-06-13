"""identity.py — pure agent-identity normalization for the pgvector plugin.

Resolves the RAW agent_identity (already chosen by the v0.3 priority chain in
``__init__.py:initialize``) into a canonical, governed theme. Pure functions
only — no I/O, no hermes-agent imports — so this module is unit-testable on its
own (see ``tests/test_identity.py``) without a running agent or database.

Why this exists (v0.4.0). Live data surfaced three failure modes the raw
priority chain let through:

  1. **PII + unbounded cardinality.** Raw direct-message session keys like
     ``agent:main:whatsapp:dm:17192714834`` became their own theme — a phone
     number stored as an ``agent_identity``, one bucket per contact.
  2. **Test pollution.** ``skill-bench`` / ``skill-bench-ws`` benchmark traffic
     landed in durable production memory.
  3. **Taxonomy drift.** Typo'd / unknown ``X-Hermes-Session-Key`` values
     silently created new themes instead of falling back to ``default``.

Placement rules (load-bearing — see the v0.4.0 red-team):

  * Normalize **once, at initialize() time, AFTER the priority chain**. Running
    it *before* the chain would lose a typo'd header before the allow-list can
    see it; running it *at read time* in ``search()`` would make historical
    rows (written under their raw identity) unrecallable.
  * Existing rows are **not** retroactively rewritten — they are historical
    events. Data cleanup is a separate, explicit operation.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional, Tuple

# Direct-message session keys carry a per-user id (often a phone number) as a
# segment. Collapse the whole family to ONE bucket so we neither get a theme
# per contact (cardinality) nor store the phone number as an agent_identity
# (PII). Matches e.g. 'agent:main:whatsapp:dm:17192714834',
# 'agent:x:telegram:dm:55512345', 'signal:dm:+1...'.
_DM_RE = re.compile(r"(?:^|:)dm:|:whatsapp:|:telegram:|:signal:", re.IGNORECASE)

# Benchmark / test-harness identities that must not pollute durable prod memory.
# 'skill-bench', 'skill-bench-ws', '<x>-bench', '<x>-bench-ws', 'bench'.
_BENCH_RE = re.compile(r"^(?:.*-)?bench(?:-ws)?$", re.IGNORECASE)

DM_BUCKET = "whatsapp-dm"
BENCH_BUCKET = "_bench"
DEFAULT_IDENTITY = "default"

# Governed sinks are always permitted, even under a strict allow-list — they are
# isolation buckets, not user themes, so isolation must keep working regardless.
_ALWAYS_ALLOWED = frozenset({DM_BUCKET, BENCH_BUCKET, DEFAULT_IDENTITY})


def normalize_identity(
    raw: Optional[str],
    *,
    allowed_themes: Optional[Iterable[str]] = None,
    aliases: Optional[Mapping[str, str]] = None,
    bench_mode: str = "bucket",
) -> Tuple[str, bool, str]:
    """Map a raw agent_identity to a canonical, governed theme.

    Returns ``(canonical, normalized, reason)``:

      * ``canonical``  — the theme to actually scope the write/recall by.
      * ``normalized`` — True if ``canonical`` differs from the raw input.
      * ``reason``     — short tag for logging: one of ``empty``, ``alias``,
        ``dm-bucket``, ``bench-bucket``, ``bench-reject``, ``not-in-allowlist``,
        ``unchanged``.

    Rule order (first terminal match wins; the allow-list is the final gate):

      0. empty / None                         -> 'default'      (empty)
      1. exact alias-map hit                   -> aliases[raw]   (alias)
      2. DM / direct-message key               -> 'whatsapp-dm'  (dm-bucket)
      3. *-bench / skill-bench(-ws) / bench    -> '_bench'       (bench-bucket)
                                                  or 'default'   (bench-reject)
      4. allow-list gate (only if allowed_themes given):
            canonical in allowed (∪ governed sinks) -> keep
            else                                     -> 'default' (not-in-allowlist)
      5. otherwise                             -> raw (trimmed)  (unchanged)

    ``bench_mode`` is ``'bucket'`` (isolate to ``_bench``, still searchable
    within that bucket) or ``'reject'`` (drop to ``default``).
    """
    if raw is None:
        return DEFAULT_IDENTITY, True, "empty"
    canonical = raw.strip()
    if not canonical:
        return DEFAULT_IDENTITY, True, "empty"

    # 1. explicit alias remap (operator-configured, e.g. {'agent-hermes': 'hermes'}).
    if aliases and canonical in aliases:
        canonical = (aliases[canonical] or "").strip() or DEFAULT_IDENTITY

    # 2. DM / direct-message session keys -> single bucket (PII + cardinality).
    if _DM_RE.search(canonical):
        return DM_BUCKET, (DM_BUCKET != raw), "dm-bucket"

    # 3. benchmark / test-harness traffic -> isolate or reject.
    if _BENCH_RE.match(canonical):
        if bench_mode == "reject":
            return DEFAULT_IDENTITY, (DEFAULT_IDENTITY != raw), "bench-reject"
        return BENCH_BUCKET, (BENCH_BUCKET != raw), "bench-bucket"

    # 4. allow-list gate (final). DM/bench buckets already returned above.
    if allowed_themes:
        allowed = {t.strip() for t in allowed_themes if t and t.strip()} | _ALWAYS_ALLOWED
        if canonical not in allowed:
            return DEFAULT_IDENTITY, (DEFAULT_IDENTITY != raw), "not-in-allowlist"

    # 5. governed-but-unchanged (or alias-applied) identity.
    reason = "alias" if (aliases and raw.strip() in aliases) else "unchanged"
    return canonical, (canonical != raw), reason


def classify_kind(agent_identity: str) -> str:
    """Best-effort classification of an identity for the memory_agents registry.

    Pure heuristic over the canonical identity — used only to tag rows in
    ``memory_agents.kind`` for human-readable attribution, never for scoping.
    """
    if agent_identity == DEFAULT_IDENTITY:
        return "default"
    if agent_identity == DM_BUCKET:
        return "dm"
    if agent_identity == BENCH_BUCKET:
        return "bench"
    if agent_identity.startswith("agent-"):
        return "worker"
    return "theme"

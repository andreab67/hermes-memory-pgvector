"""identity.py — pure agent-identity normalization for the pgvector plugin.

Resolves the RAW agent_identity (already chosen by the v0.3 priority chain in
``__init__.py:initialize``) into a canonical, governed theme. Pure functions
only — no I/O, no hermes-agent imports — so this module is unit-testable on its
own (see ``tests/test_identity.py``) without a running agent or database.

Why this exists (v0.4.0). Live data surfaced three failure modes the raw
priority chain let through:

  1. **PII + unbounded cardinality.** Raw direct-message session keys like
     ``agent:main:whatsapp:dm:15550100123`` became their own theme — a phone
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
# (PII). Matches e.g. 'agent:main:whatsapp:dm:15550100123',
# 'agent:x:telegram:dm:55512345', 'signal:dm:+1...', 'whatsapp:15550100124'.
#
# v0.4.2: a platform token alone no longer triggers — it must be followed by
# a 'dm:' segment or a phone/chat id. The old pattern's bare ':signal:'
# alternative swept ordinary colon-namespaced themes (e.g. a trading key like
# 'desk:signal:main') into the DM bucket, breaking theme isolation on nothing
# but the word "signal".
_DM_RE = re.compile(
    r"(?:^|:)dm:|(?:^|:)(?:whatsapp|telegram|signal):(?=dm:|\+?\d)",
    re.IGNORECASE,
)

# Platform tokens, for the UNPREFIXED key shapes only. The gateway always
# emits an "agent:<ns>:" prefix, but unprefixed keys demonstrably reach this
# module -- _DM_RE below deliberately matches 'signal:dm:+1...',
# 'whatsapp:15550100124' and 'telegram:+15551234', all pinned by tests. An
# enumeration is the only way to recognise those without also swallowing
# ordinary colon-namespaced themes like 'eng:channel:alerts'.
_PLATFORMS = (
    "local|telegram|discord|whatsapp_cloud|whatsapp|slack|signal|mattermost"
    "|matrix|homeassistant|email|sms|dingtalk|webhook|feishu|wecom_callback"
    "|wecom|weixin|qqbot|bluebubbles|msgraph_webhook|yuanbao|relay|api_server"
)

# Multi-party session keys (group / channel / thread). Host layout is
# agent:<ns>:<platform>:<chat_type>[:<chat_id>][:<thread_id>][:<user>]
# (gateway/session.py:build_session_key + _session_key_namespace). With the
# default group_sessions_per_user the TRAILING segment is the participant id --
# a phone number on WhatsApp/SMS/Signal -- so the same two failure modes the DM
# bucket exists for apply here: PII stored as an agent_identity, and one theme
# per (chat, participant) pair.
#
# TWO alternatives, deliberately, because either one alone leaks:
#
#   1. Prefixed, STRUCTURAL: agent:<ns>:<platform>:<chat_type>. Platform-
#      agnostic, so it covers all 26 shipped platforms and any a plugin adds
#      later. An earlier enumeration-only version silently missed every
#      platform it did not name (whatsapp_cloud, qqbot, bluebubbles, ...).
#   2. Unprefixed, ENUMERATED: <platform>:<chat_type>. An earlier
#      structural-only version dropped this shape entirely, so
#      'whatsapp:group:<chat>:<phone>' passed through untouched and the phone
#      number was stored verbatim -- the exact regression #1 was fixed for.
#      The enumeration is required here: a bare '<anything>:<chat_type>:'
#      would swallow ordinary themes like 'eng:channel:alerts' or
#      'ops:group:oncall'. That over-match is the trap the v0.4.2 note below
#      records for a bare ':signal:' alternative.
_GROUP_RE = re.compile(
    r"^agent:[^:]+:[^:]+:(?:group|channel|thread)(?::|$)"
    r"|(?:^|:)(?:" + _PLATFORMS + r"):(?:group|channel|thread)(?::|$)",
    re.IGNORECASE,
)

# Benchmark / test-harness identities that must not pollute durable prod memory.
# 'skill-bench', 'skill-bench-ws', '<x>-bench', '<x>-bench-ws', 'bench'.
_BENCH_RE = re.compile(r"^(?:.*-)?bench(?:-ws)?$", re.IGNORECASE)

DM_BUCKET = "whatsapp-dm"
# One bucket for ALL multi-party external traffic, mirroring DM_BUCKET: the
# point is isolation, not a per-chat taxonomy. Keeping a theme per group chat
# would trade the PII problem for the cardinality one.
GROUP_BUCKET = "external-group"
BENCH_BUCKET = "_bench"
DEFAULT_IDENTITY = "default"

# Governed sinks are always permitted, even under a strict allow-list — they are
# isolation buckets, not user themes, so isolation must keep working regardless.
_ALWAYS_ALLOWED = frozenset({DM_BUCKET, GROUP_BUCKET, BENCH_BUCKET, DEFAULT_IDENTITY})


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
        Always lowercase (v0.6.0 — M7: identities are case-folded, so
        ``Marketing`` and ``marketing`` resolve to the same theme).
      * ``normalized`` — True if ``canonical`` differs from the raw input,
        including a case-only change (e.g. ``Marketing`` -> ``marketing``).
      * ``reason``     — short tag for logging: one of ``empty``, ``alias``,
        ``dm-bucket``, ``group-bucket``, ``bench-bucket``, ``bench-reject``,
        ``not-in-allowlist``, ``lowercased``, ``unchanged``.

    Rule order (first terminal match wins; the allow-list is the final gate).
    Every comparison below (alias keys, allow-list entries) is done on the
    lowercased value; the caller still gets the untouched ``raw`` value back
    to record as ``raw_identity`` metadata — only the canonical bucket used
    for scoping is folded.

      0. empty / None                          -> 'default'      (empty)
      1. exact alias-map hit (lowercased)      -> aliases[raw], lowercased
                                                   (alias)
      2. DM / direct-message key               -> 'whatsapp-dm'  (dm-bucket)
      2b. group / channel / thread key         -> 'external-group'
                                                   (group-bucket)
      3. *-bench / skill-bench(-ws) / bench    -> '_bench'       (bench-bucket)
                                                  or 'default'   (bench-reject)
      4. allow-list gate (only if allowed_themes given, entries lowercased):
            canonical in allowed (∪ governed sinks) -> keep
            else                                     -> 'default' (not-in-allowlist)
      5. otherwise                             -> raw, stripped + lowercased
                                                   ('lowercased' if lowercasing
                                                   is the only change from the
                                                   stripped raw value, else
                                                   'unchanged')

    ``bench_mode`` is ``'bucket'`` (isolate to ``_bench``, still searchable
    within that bucket) or ``'reject'`` (drop to ``default``).
    """
    if raw is None:
        return DEFAULT_IDENTITY, True, "empty"
    stripped = raw.strip()
    if not stripped:
        return DEFAULT_IDENTITY, True, "empty"
    canonical = stripped.lower()

    # 1. explicit alias remap (operator-configured, e.g. {'agent-hermes': 'hermes'}).
    # Keys and the returned value are both case-folded, so an alias configured
    # as {'Agent-Hermes': 'Hermes'} matches 'agent-hermes' and always yields a
    # lowercase canonical identity.
    alias_hit = False
    if aliases:
        lowered_aliases = {str(k).lower(): v for k, v in aliases.items() if k}
        if canonical in lowered_aliases:
            canonical = ((lowered_aliases[canonical] or "").strip() or DEFAULT_IDENTITY).lower()
            alias_hit = True

    # 2. DM / direct-message session keys -> single bucket (PII + cardinality).
    if _DM_RE.search(canonical):
        return DM_BUCKET, (DM_BUCKET != raw), "dm-bucket"

    # 2b. group / channel / thread keys -> single bucket, same rationale: with
    # group_sessions_per_user (the host default) the trailing segment is the
    # participant id, which is a phone number on WhatsApp/SMS/Signal.
    if _GROUP_RE.search(canonical):
        return GROUP_BUCKET, (GROUP_BUCKET != raw), "group-bucket"

    # 3. benchmark / test-harness traffic -> isolate or reject.
    if _BENCH_RE.match(canonical):
        if bench_mode == "reject":
            return DEFAULT_IDENTITY, (DEFAULT_IDENTITY != raw), "bench-reject"
        return BENCH_BUCKET, (BENCH_BUCKET != raw), "bench-bucket"

    # 4. allow-list gate (final). DM/bench buckets already returned above.
    if allowed_themes:
        allowed = {t.strip().lower() for t in allowed_themes if t and t.strip()} | _ALWAYS_ALLOWED
        if canonical not in allowed:
            return DEFAULT_IDENTITY, (DEFAULT_IDENTITY != raw), "not-in-allowlist"

    # 5. governed-but-unchanged (or alias-applied, or case-only) identity.
    if alias_hit:
        reason = "alias"
    elif canonical != stripped:
        reason = "lowercased"
    else:
        reason = "unchanged"
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
    if agent_identity == GROUP_BUCKET:
        return "group"
    if agent_identity == BENCH_BUCKET:
        return "bench"
    if agent_identity.startswith("agent-"):
        return "worker"
    return "theme"

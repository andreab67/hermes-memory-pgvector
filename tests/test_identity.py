"""Unit tests for pgvector/identity.py — pure normalization, no DB or embed.

These run everywhere (no PG_TEST_DSN needed); identity.py has no I/O.
Import style matches test_smoke.py (repo root on sys.path, package imports —
v0.4.2: the old pgvector/-dir insert leaked bare module names onto sys.path
for the whole pytest process, masking broken imports in sibling test files).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes_pgvector.identity import (  # noqa: E402
    BENCH_BUCKET,
    DEFAULT_IDENTITY,
    DM_BUCKET,
    classify_kind,
    normalize_identity,
    GROUP_BUCKET,
)


# --- empties --------------------------------------------------------------

def test_none_and_empty_go_to_default():
    for raw in (None, "", "   "):
        canon, normalized, reason = normalize_identity(raw)
        assert canon == DEFAULT_IDENTITY
        assert normalized is True
        assert reason == "empty"


# --- plain themes pass through --------------------------------------------

def test_known_theme_unchanged():
    canon, normalized, reason = normalize_identity("marketing")
    assert canon == "marketing"
    assert normalized is False
    assert reason == "unchanged"


def test_whitespace_trimmed_counts_as_unchanged_value():
    canon, normalized, _ = normalize_identity("  marketing  ")
    assert canon == "marketing"
    assert normalized is True  # differs from the raw (untrimmed) input


# --- the PII fix: whatsapp / DM session keys ------------------------------

def test_whatsapp_dm_key_is_bucketed_and_strips_phone():
    canon, normalized, reason = normalize_identity("agent:main:whatsapp:dm:17192714834")
    assert canon == DM_BUCKET
    assert "17192714834" not in canon  # phone number is gone
    assert normalized is True
    assert reason == "dm-bucket"


def test_other_dm_platforms_bucket_too():
    for raw in ("agent:x:telegram:dm:55512345", "signal:dm:+1-719-555-0000"):
        canon, _, reason = normalize_identity(raw)
        assert canon == DM_BUCKET
        assert reason == "dm-bucket"


def test_dm_bucket_is_idempotent():
    canon, normalized, _ = normalize_identity(DM_BUCKET)
    assert canon == DM_BUCKET
    assert normalized is False  # already canonical, nothing changed


def test_platform_token_alone_is_not_a_dm_key():
    # v0.4.2: a bare ':signal:'/':whatsapp:' segment no longer sweeps ordinary
    # colon-namespaced themes into the DM bucket on nothing but the word.
    canon, _, reason = normalize_identity("desk:signal:main")
    assert canon == "desk:signal:main"
    assert reason == "unchanged"


def test_platform_followed_by_id_is_a_dm_key():
    for raw in ("whatsapp:17195550000", "agent:x:signal:12345", "telegram:+15551234"):
        canon, _, reason = normalize_identity(raw)
        assert canon == DM_BUCKET
        assert reason == "dm-bucket"


# --- bench isolation ------------------------------------------------------

def test_skill_bench_buckets_by_default():
    for raw in ("skill-bench", "skill-bench-ws", "load-bench", "bench"):
        canon, normalized, reason = normalize_identity(raw)
        assert canon == BENCH_BUCKET
        assert reason == "bench-bucket"


def test_bench_reject_mode_drops_to_default():
    canon, _, reason = normalize_identity("skill-bench", bench_mode="reject")
    assert canon == DEFAULT_IDENTITY
    assert reason == "bench-reject"


# --- allow-list gate ------------------------------------------------------

def test_unknown_identity_falls_back_to_default_under_allowlist():
    allowed = ["marketing", "sales", "morning-report"]
    canon, normalized, reason = normalize_identity("typo-theme", allowed_themes=allowed)
    assert canon == DEFAULT_IDENTITY
    assert normalized is True
    assert reason == "not-in-allowlist"


def test_allowed_identity_passes_under_allowlist():
    canon, normalized, reason = normalize_identity("marketing", allowed_themes=["marketing", "sales"])
    assert canon == "marketing"
    assert normalized is False
    assert reason == "unchanged"


def test_governed_sinks_always_allowed_even_under_strict_allowlist():
    allowed = ["marketing"]  # deliberately excludes the buckets
    assert normalize_identity(DM_BUCKET, allowed_themes=allowed)[0] == DM_BUCKET
    assert normalize_identity(BENCH_BUCKET, allowed_themes=allowed)[0] == BENCH_BUCKET
    assert normalize_identity(DEFAULT_IDENTITY, allowed_themes=allowed)[0] == DEFAULT_IDENTITY


# --- aliases --------------------------------------------------------------

def test_alias_remaps_before_allowlist():
    canon, normalized, reason = normalize_identity(
        "agent-hermes",
        aliases={"agent-hermes": "hermes"},
        allowed_themes=["hermes"],
    )
    assert canon == "hermes"
    assert normalized is True
    assert reason == "alias"


# --- kind classification --------------------------------------------------

def test_classify_kind():
    assert classify_kind("agent-sre") == "worker"
    assert classify_kind("marketing") == "theme"
    assert classify_kind(DM_BUCKET) == "dm"
    assert classify_kind(BENCH_BUCKET) == "bench"
    assert classify_kind(DEFAULT_IDENTITY) == "default"


# ---------------------------------------------------------------------------
# v0.5.0 -- multi-party (group / channel / thread) session keys.
#
# The host builds session keys as
#   <ns>:<platform>:<chat_type>[:<chat_id>][:<thread_id>][:<user>]
# (gateway/session.py:build_session_key). With group_sessions_per_user, which
# is the DEFAULT, the trailing segment is the participant id -- a phone number
# on WhatsApp/SMS/Signal. Before v0.5.0 these keys passed through untouched, so
# a phone number was stored verbatim as an agent_identity: exactly failure mode
# #1 that this module exists to prevent, just via a different chat_type. With
# an allow-list configured they instead collapsed to `default`, where every
# theme could read the external content.
# ---------------------------------------------------------------------------

GROUP_KEYS = [
    "agent:main:whatsapp:group:120363@g.us:17192714834",
    "agent:main:whatsapp:channel:120363@g.us:17192714834",
    "agent:main:telegram:group:-1001234567890:55512345",
    "agent:main:slack:thread:C0123:1699999999.123:U0456",
    "agent:main:discord:channel:987654321:112233",
    "agent:main:signal:group:abcd==:+15551234567",
]


def test_group_keys_collapse_to_the_group_bucket():
    for key in GROUP_KEYS:
        canonical, normalized, reason = normalize_identity(key)
        assert canonical == GROUP_BUCKET, key
        assert normalized is True
        assert reason == "group-bucket"


def test_group_bucketing_removes_the_participant_id():
    """The whole point: no phone number may survive into the identity."""
    canonical, _, _ = normalize_identity("agent:main:whatsapp:group:120363@g.us:17192714834")
    assert "17192714834" not in canonical
    assert "120363" not in canonical


def test_group_bucket_survives_a_strict_allow_list():
    """Isolation buckets are always permitted -- otherwise a configured
    allow-list would route group traffic to `default`, where every theme can
    read it, which is worse than bucketing it."""
    canonical, _, reason = normalize_identity(
        "agent:main:whatsapp:group:120363@g.us:17192714834",
        allowed_themes=["marketing", "sales"],
    )
    assert canonical == GROUP_BUCKET
    assert reason == "group-bucket"


def test_dm_keys_still_win_over_group_matching():
    canonical, _, reason = normalize_identity("agent:main:whatsapp:dm:17192714834")
    assert canonical == DM_BUCKET
    assert reason == "dm-bucket"


def test_group_pattern_does_not_sweep_ordinary_colon_themes():
    """Anchored on <platform>:<chat_type>:, not a bare chat-type token. A loose
    pattern would swallow ordinary namespaced themes on a common word -- the
    same trap the v0.4.2 note records for a bare ':signal:' alternative."""
    for benign in (
        "eng:channel:alerts",
        "desk:signal:main",
        "ops:group:oncall",
        "thread:pool:tuning",
        "marketing",
        "intraday-trading",
    ):
        canonical, normalized, _ = normalize_identity(benign)
        assert canonical == benign, benign
        assert normalized is False


def test_group_pattern_is_platform_agnostic():
    """Anchored structurally on agent:<ns>:<platform>:, NOT on an enumerated
    platform list. The host ships 26 platforms and plugins register more; an
    earlier enumeration silently missed whatsapp_cloud, bluebubbles, qqbot,
    msgraph_webhook and every other name it did not list."""
    platforms = [
        "local", "telegram", "discord", "whatsapp", "whatsapp_cloud", "slack",
        "signal", "mattermost", "matrix", "homeassistant", "email", "sms",
        "dingtalk", "webhook", "feishu", "wecom", "wecom_callback", "weixin",
        "qqbot", "bluebubbles", "msgraph_webhook", "yuanbao", "relay",
        "api_server", "plugin", "some_future_platform",
    ]
    for plat in platforms:
        key = f"agent:main:{plat}:group:120363@g.us:17192714834"
        canonical, _, reason = normalize_identity(key)
        assert canonical == GROUP_BUCKET, f"{plat} not bucketed"
        assert reason == "group-bucket"


def test_group_bucket_has_its_own_registry_kind():
    """classify_kind() tags memory_agents rows. The new PII sink must not be
    registered as an ordinary theme, the way the other sinks are not."""
    assert classify_kind(GROUP_BUCKET) == "group"
    assert classify_kind(DM_BUCKET) == "dm"
    assert classify_kind("marketing") == "theme"

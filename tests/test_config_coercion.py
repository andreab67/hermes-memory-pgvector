"""Unit tests for the config type-coercion helpers in pgvector/__init__.py.

Pure functions, no DB or embed endpoint — `_as_bool` / `_as_theme_list` only
touch plain Python values. Import style matches test_identity.py / test_smoke.py
(repo root on sys.path, package imports).

GROUP A regression coverage: config values that reach the plugin as the
STRINGS "true"/"false" (the config schema's declared type for boolean
toggles) or as a bare comma-joined string (the schema's declared type for
`allowed_themes`) used to be handled with plain truthiness / iteration, which
silently broke embed_on_write, sync_turns, hybrid_search, bulk_sync_on_init,
and the whole per-agent theme allow-list. See the docstrings on _as_bool and
_as_theme_list in pgvector/__init__.py for the full rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector import _as_bool, _as_theme_list  # noqa: E402
from pgvector.identity import normalize_identity  # noqa: E402


# ---------------------------------------------------------------------------
# _as_bool
# ---------------------------------------------------------------------------

def test_as_bool_string_false_is_false():
    # The defect: save_config() persists booleans as the strings "true"/
    # "false" (the config schema's declared type). bool("false") is True —
    # plain truthiness silently inverted the toggle. This is the exact
    # regression: a config on-disk value of "false" must coerce to False.
    assert _as_bool("false", True) is False


def test_as_bool_string_true_is_true():
    assert _as_bool("true", False) is True


def test_as_bool_case_insensitive_and_synonyms():
    assert _as_bool("FALSE", True) is False
    assert _as_bool("no", True) is False
    assert _as_bool("0", True) is False
    assert _as_bool("on", False) is True
    assert _as_bool("1", False) is True


def test_as_bool_real_bools_pass_through():
    assert _as_bool(True, False) is True
    assert _as_bool(False, True) is False


def test_as_bool_none_returns_default():
    assert _as_bool(None, True) is True
    assert _as_bool(None, False) is False


# ---------------------------------------------------------------------------
# _as_theme_list
# ---------------------------------------------------------------------------

def test_as_theme_list_splits_comma_string():
    # The defect: allowed_themes reaches normalize_identity() as a bare
    # string (config schema declares it as a scalar). A string handed to
    # normalize_identity()'s allow-list is iterated CHARACTER BY CHARACTER
    # ({'m', 'a', 'r', ...}), so no real theme name ever matches and the
    # allow-list silently routes every theme to 'default'.
    assert _as_theme_list("marketing,sales") == ["marketing", "sales"]


def test_as_theme_list_single_theme_string():
    assert _as_theme_list("marketing") == ["marketing"]


def test_as_theme_list_trims_whitespace_around_entries():
    assert _as_theme_list("marketing, sales ,  trading") == ["marketing", "sales", "trading"]


def test_as_theme_list_passes_real_list_through():
    assert _as_theme_list(["marketing", "sales"]) == ["marketing", "sales"]


def test_as_theme_list_empty_forms_map_to_none():
    assert _as_theme_list(None) is None
    assert _as_theme_list("") is None
    assert _as_theme_list([]) is None


# ---------------------------------------------------------------------------
# End-to-end: _as_theme_list feeding normalize_identity()
# ---------------------------------------------------------------------------

def test_theme_list_from_config_string_keeps_allowed_theme():
    # This is the exact bug end to end: a config-file `allowed_themes:
    # "marketing,sales"` scalar must resolve to the real theme, not collapse
    # to 'default' because the raw string was iterated character-by-character.
    allowed = _as_theme_list("marketing,sales")
    canon, normalized, reason = normalize_identity("marketing", allowed_themes=allowed)
    assert canon == "marketing"
    assert normalized is False
    assert reason == "unchanged"


def test_theme_list_from_config_string_rejects_unknown_theme():
    allowed = _as_theme_list("marketing,sales")
    canon, normalized, reason = normalize_identity("typo-theme", allowed_themes=allowed)
    assert canon == "default"
    assert normalized is True
    assert reason == "not-in-allowlist"

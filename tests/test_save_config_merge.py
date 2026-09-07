"""Unit test for PgvectorMemoryProvider.save_config()'s merge-not-replace
contract. Pure filesystem + PyYAML (a declared runtime dependency, see
pyproject.toml) -- no DB, no embed endpoint, no hermes-agent runtime.

Regression coverage: get_config_schema() only declares a subset of the keys
the plugin actually reads at runtime (identity_aliases and
embed_write_backoff are both live config, per pgvector/__init__.py, but
neither has a schema entry). The hermes-agent config UI's "save settings"
flow calls save_config(values, hermes_home) with `values` containing only
the SCHEMA-declared keys. A save_config() that assigned
`existing["plugins"]["pgvector"] = values` wholesale would silently delete
any hand-edited, undeclared key on the very next settings save -- exactly
the identity_aliases remap map operators hand-edit into config.yaml. The fix
merges: `{**current, **values}`, so undeclared keys survive and only the
keys actually passed get updated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pgvector import PgvectorMemoryProvider  # noqa: E402


def test_save_config_preserves_undeclared_keys_not_present_in_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "plugins": {
                    "pgvector": {
                        "dsn": "dbname=hermes_memory user=hermes host=/var/run/postgresql",
                        # Undeclared in get_config_schema() -- hand-edited by an
                        # operator, read at runtime (normalize_identity's
                        # `aliases` arg in initialize()), never exposed through
                        # the settings UI's schema-driven form.
                        "identity_aliases": {"agent-hermes": "hermes"},
                    }
                }
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )

    provider = PgvectorMemoryProvider()
    # Simulates the settings UI saving a schema-declared key WITHOUT
    # identity_aliases in the payload -- it was never in the schema, so it's
    # never in `values`.
    provider.save_config({"dsn": "dbname=new_db user=hermes host=/var/run/postgresql"}, str(tmp_path))

    with open(config_path, encoding="utf-8") as fh:
        saved = yaml.safe_load(fh)

    pgvector_cfg = saved["plugins"]["pgvector"]
    assert pgvector_cfg["dsn"] == "dbname=new_db user=hermes host=/var/run/postgresql", (
        "the key actually passed in `values` must be updated"
    )
    assert pgvector_cfg.get("identity_aliases") == {"agent-hermes": "hermes"}, (
        "save_config() must MERGE into the existing plugins.pgvector block, not "
        "replace it wholesale -- a pre-existing key omitted from `values` "
        "(identity_aliases here) must survive the save"
    )

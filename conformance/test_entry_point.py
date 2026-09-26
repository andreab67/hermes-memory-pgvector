"""WP-F conformance test 3 (docs/release/PLAN-1.0.md Sec 5 WP-F):

The `hermes_agent.memory_providers` entry point named `pgvector` loads
through upstream's plugins/memory loader and exposes `register`.

Needs `plugins.memory` importable, which (unlike agent.memory_provider /
agent.memory_manager) pulls in hermes-agent's config stack
(hermes_cli.config -> hermes_yaml -> ruamel.yaml). hermes-agent's own
pyproject.toml gates nearly all of its runtime dependencies behind
`python_version >= '3.14'` (its own comment: "we only support 3.14, but we
need to allow old hermes installs on <3.14 to get thru their update step");
on an older interpreter a plain `pip install -e` of it -- what
scripts/conformance.sh does -- installs those few extra bits only if
they happen to already be present. Skip cleanly rather than require them.
"""

from __future__ import annotations

import pytest

agent_memory_provider = pytest.importorskip("agent.memory_provider")
memory_plugins = pytest.importorskip("plugins.memory")

import hermes_pgvector  # noqa: E402

MemoryProvider = agent_memory_provider.MemoryProvider


def test_pgvector_entry_point_is_registered_under_the_upstream_group():
    """importlib.metadata must expose our own pip-installed entry point under
    plugins/memory's discovery group -- this is what production activation
    (`memory.provider: pgvector` in config.yaml) resolves by name."""
    entry_point = memory_plugins.find_provider_entry_point("pgvector")
    assert entry_point is not None, (
        "no 'pgvector' entry point found under "
        f"{memory_plugins.ENTRY_POINTS_GROUP!r} -- scripts/conformance.sh must "
        "have pip-installed this package (pyproject.toml's "
        '[project.entry-points."hermes_agent.memory_providers"] pgvector = '
        '"hermes_pgvector") into the same interpreter running this test'
    )
    assert "hermes_pgvector" in (entry_point.value or "")


def test_pgvector_entry_point_loads_a_module_exposing_register():
    """entry_point.load() must resolve to our module (or the attribute it
    names) and that target must expose `register`, the sole contract
    `_load_provider_from_entry_point` / `_load_provider_from_dir` require
    to turn an installed package into a MemoryProvider."""
    entry_point = memory_plugins.find_provider_entry_point("pgvector")
    assert entry_point is not None
    loaded = entry_point.load()
    assert loaded is hermes_pgvector or loaded is getattr(hermes_pgvector, "register", None)
    target = loaded if hasattr(loaded, "register") else None
    assert target is not None, "entry point target has no register(ctx) callable"
    assert callable(target.register)


def test_load_memory_provider_pgvector_returns_a_working_provider(tmp_path, monkeypatch):
    """The real, end-to-end path hermes-agent itself calls at startup to
    activate a configured external provider: plugins.memory.load_memory_provider(name).

    HERMES_HOME must be a real (writable) directory -- upstream's own test
    suite (tests/plugins/memory/test_discovery_sources.py,
    test_activation_is_not_gated_on_plugins_enabled) sets this up the same
    way; it is not otherwise gated on `plugins.enabled`.

    NOTE: find_provider_dir() resolves our entry point to the on-disk
    hermes_pgvector/ package directory (so a pip-installed provider still
    gets its config-panel/CLI directory lookups -- see that function's
    docstring), and the loader then re-imports __init__.py BY PATH under
    its own synthetic `_hermes_user_memory` namespace rather than reusing
    the already-imported `hermes_pgvector` module. The returned instance is
    therefore NOT `isinstance(..., hermes_pgvector.PgvectorMemoryProvider)`
    even though it is the same source file -- checked by class name instead
    of identity/isinstance against our own import of the module.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)

    provider = memory_plugins.load_memory_provider("pgvector", register_skills=False)

    assert provider is not None, (
        "plugins.memory.load_memory_provider('pgvector') returned None -- "
        "the entry point did not resolve to an instantiable MemoryProvider"
    )
    assert isinstance(provider, MemoryProvider)
    assert provider.name == "pgvector"
    assert type(provider).__name__ == "PgvectorMemoryProvider"
    assert provider.is_available()

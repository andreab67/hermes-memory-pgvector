"""Regression: hermes-agent's plugin loader clobbers the package's `embed` name.

No DB, no network -- urllib.request.urlopen is replaced with a stub.

What happens in production. hermes-agent loads a provider directory with
plugins/plugin_loader.py:load_plugin_module(), which executes every sibling
file as `<pkg>.<stem>`, executes `__init__.py`, and then runs

    for sub_name, sub_mod in loaded_submodules:
        setattr(mod, sub_name, sub_mod)

For this package that includes `setattr(pkg, "embed", <the embed SUBMODULE>)`.
A module's attributes ARE its globals, so every `embed(...)` call inside
`__init__.py` then resolved to a module object and raised
`TypeError: 'module' object is not callable`. That is not an EmbeddingError:
it escaped prefetch and the recall tools, and the writer drain dropped each
write outright instead of storing it text-only.
Production papered over it with an external text patch that rebinds
`mod.embed` in register(). v0.5.3 fixes it natively: call sites go through
`_embed_with_config()`, which uses the private alias `_embed_text`.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
import urllib.request
from importlib import import_module
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PKG_DIR = REPO / "hermes_pgvector"
sys.path.insert(0, str(REPO))

import hermes_pgvector as pgvector_pkg  # noqa: E402

embed_mod = import_module("hermes_pgvector.embed")


class _Resp:
    def __init__(self, dim: int):
        self._raw = json.dumps({"data": [{"embedding": [0.02] * dim}]}).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def http(monkeypatch):
    calls = []

    def _urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp(768)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return calls


class _Store:
    def search(self, **kw):
        return [{"score": 0.8, "target": "memory", "content": "clobber-proof recall"}]

    def hybrid_search(self, **kw):
        return self.search()

    def search_turns(self, **kw):
        return []

    def hybrid_search_turns(self, **kw):
        return []


def _healthy(provider_cls):
    p = provider_cls()
    p._healthy = True
    p._store = _Store()
    p._agent_identity = "marketing"
    p._raw_identity = "marketing"
    p._session_id = "sess-clobber"
    return p


def _exercise_every_embed_path(p, http_calls):
    out = p.prefetch("what was decided")
    assert "clobber-proof recall" in out, f"prefetch lost its recall: {out!r}"

    payload = json.loads(p.handle_tool_call("recall_memory", {"query": "q"}))
    assert "error" not in payload and payload["count"] == 1, payload
    payload = json.loads(p.handle_tool_call("recall_conversation", {"query": "q"}))
    assert "error" not in payload, payload

    assert len(p._make_embed_fn()("bulk-imported entry")) == 768
    assert len(p._maybe_embed("durable content from the writer drain")) == 768

    assert len(http_calls) == 5, "every path must have reached the (stubbed) endpoint"


def test_provider_embed_paths_survive_the_embed_attribute_clobber(monkeypatch, http):
    """Simulate exactly what the loader does to the package attribute."""
    monkeypatch.setattr(pgvector_pkg, "embed", embed_mod)
    assert isinstance(pgvector_pkg.embed, types.ModuleType)
    assert not callable(pgvector_pkg.embed), "the simulation must reproduce the clobber"

    _exercise_every_embed_path(_healthy(pgvector_pkg.PgvectorMemoryProvider), http)


def test_cli_embed_fn_survives_the_clobber(monkeypatch, http):
    monkeypatch.setattr(pgvector_pkg, "embed", embed_mod)
    cli = import_module("hermes_pgvector.__main__")
    fn = cli._make_embed_fn(cli.build_parser().parse_args(["backfill"]), {})
    assert len(fn("row being backfilled")) == 768


def _load_like_hermes_agent(module_name: str) -> types.ModuleType:
    """Mirror hermes-agent plugins/plugin_loader.py:load_plugin_module().

    Siblings first (as `<module_name>.<stem>`), then __init__.py, then bind
    each sibling back onto the package -- the step that clobbers `embed`.
    """
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(PKG_DIR / "__init__.py"),
        submodule_search_locations=[str(PKG_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    loaded = []
    for sub_file in sorted(PKG_DIR.glob("*.py")):
        full = f"{module_name}.{sub_file.stem}"
        if sub_file.name == "__init__.py" or full in sys.modules:
            continue
        sub_spec = importlib.util.spec_from_file_location(full, str(sub_file))
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[full] = sub_mod
        try:
            sub_spec.loader.exec_module(sub_mod)
        except Exception:  # noqa: BLE001 -- the real loader debug-logs and skips
            continue
        loaded.append((sub_file.stem, sub_mod))
    spec.loader.exec_module(mod)
    for sub_name, sub_mod in loaded:
        setattr(mod, sub_name, sub_mod)
    return mod


def test_provider_survives_a_faithful_reproduction_of_the_loader(http):
    name = "_hpgv_loader_clobber_test.pgvector"
    try:
        mod = _load_like_hermes_agent(name)
        assert isinstance(mod.embed, types.ModuleType), (
            "the reproduction must leave `embed` bound to the submodule, as production does"
        )
        _exercise_every_embed_path(_healthy(mod.PgvectorMemoryProvider), http)
    finally:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            sys.modules.pop(key, None)


def test_init_never_calls_the_bare_global_embed():
    """Static guard: a future `embed(...)` call in __init__.py would reopen
    the bug, and only the loader path (not a plain import) would show it."""
    tree = ast.parse((PKG_DIR / "__init__.py").read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "embed"
    ]
    assert offenders == [], f"__init__.py calls bare embed() at lines {offenders}"


def test_public_embed_name_is_still_importable():
    from hermes_pgvector import embed

    assert embed is embed_mod.embed

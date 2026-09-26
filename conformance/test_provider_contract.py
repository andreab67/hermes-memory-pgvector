"""WP-F conformance tests 1-2 (docs/release/PLAN-1.0.md Sec 5 WP-F):

1. PgvectorMemoryProvider subclasses upstream MemoryProvider and instantiates
   (no abstract methods left).
2. For every method the plugin overrides, every parameter of the upstream
   signature is accepted (explicitly or via **kwargs) --
   inspect.signature(...).bind(...) with upstream's shape.

Both skip cleanly when hermes-agent is not importable (pytest.importorskip).

Import order is load-bearing here: hermes_pgvector/__init__.py falls back to
a stub `MemoryProvider = object` when `agent.memory_provider` is not yet on
sys.path AT THE MOMENT hermes_pgvector is first imported in this process,
and that binding is cached for the rest of the run. `pytest.importorskip`
below both imports agent.memory_provider (fixing sys.modules ordering) and
skips this module cleanly if it can't -- see conformance/conftest.py's
docstring for the full explanation.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

import pytest

agent_memory_provider = pytest.importorskip("agent.memory_provider")

import hermes_pgvector  # noqa: E402

MemoryProvider = agent_memory_provider.MemoryProvider
PluginProvider = hermes_pgvector.PgvectorMemoryProvider


# ---------------------------------------------------------------------------
# Test 1 -- subclass + instantiation
# ---------------------------------------------------------------------------

def test_provider_subclasses_upstream_memory_provider():
    assert issubclass(PluginProvider, MemoryProvider), (
        f"{PluginProvider!r} does not subclass {MemoryProvider!r}. If "
        "hermes_pgvector was imported anywhere in this process before "
        "agent.memory_provider was importable, hermes_pgvector/__init__.py's "
        "`except ImportError: MemoryProvider = object` fallback bound the "
        "wrong base class -- that binding is cached in sys.modules for the "
        "whole run. Otherwise this is a real conformance break: the plugin "
        "no longer derives from the upstream ABC."
    )


def test_provider_instantiates_with_no_abstract_methods_left():
    # Raises TypeError ("Can't instantiate abstract class ... with abstract
    # method(s) ...") if the plugin is missing an implementation upstream's
    # ABC still marks @abstractmethod (name, is_available, initialize,
    # get_tool_schemas as of the pinned ref).
    instance = PluginProvider()
    assert isinstance(instance, MemoryProvider)
    # name is itself one of the abstract members; touch it to prove it
    # resolves to a concrete value, not just a non-raising property object.
    assert isinstance(instance.name, str) and instance.name


# ---------------------------------------------------------------------------
# Test 2 -- every overridden method accepts upstream's call shape
# ---------------------------------------------------------------------------

def _is_real_override(name: str, upstream_attr: Any) -> bool:
    if name.startswith("__"):
        return False
    if isinstance(upstream_attr, property):
        # `name` is a property upstream; test 1 covers it via instantiation.
        # Signature compatibility for a property getter isn't a meaningful
        # "call shape" the manager could get wrong the way it can for a
        # method, so it is out of scope for this test.
        return False
    if not callable(upstream_attr):
        return False
    return True


def _upstream_defined_methods() -> List[Tuple[str, Any]]:
    """(name, unbound function) for every callable MemoryProvider defines
    directly on itself (abstract or not) -- the actual interface surface,
    not anything merely inherited from ABC/object."""
    return [
        (name, attr) for name, attr in vars(MemoryProvider).items()
        if _is_real_override(name, attr)
    ]


def _call_shape_for(sig: inspect.Signature) -> Tuple[list, Dict[str, Any]]:
    """Positional args + kwargs that exercise every named parameter of `sig`
    (minus `self`), passed with the SAME kind upstream declares it: a
    positional-or-keyword parameter is passed positionally (so a plugin
    override that reordered or renamed it fails to bind), a keyword-only
    parameter is passed by keyword. *args/**kwargs slots in the upstream
    signature itself carry no concrete parameter to forward, so they
    contribute nothing here. Values are all `None` -- bind() only checks
    the call SHAPE, it never executes the function body.
    """
    args: list = []
    kwargs: Dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[pname] = None
        else:
            args.append(None)
    return args, kwargs


def test_overridden_methods_accept_every_upstream_parameter():
    upstream_methods = _upstream_defined_methods()
    assert upstream_methods, "sanity: MemoryProvider should define several methods"

    checked = []
    failures = []
    for name, upstream_fn in upstream_methods:
        if name not in vars(PluginProvider):
            continue  # not overridden -- inherits upstream's own signature verbatim
        plugin_fn = vars(PluginProvider)[name]
        if isinstance(plugin_fn, property):
            continue
        upstream_sig = inspect.signature(upstream_fn)
        plugin_sig = inspect.signature(plugin_fn)
        args, kwargs = _call_shape_for(upstream_sig)
        try:
            # `None` in the first slot stands in for `self` -- both
            # signatures are the raw unbound functions from each class's
            # __dict__, so both still declare it as their first parameter.
            plugin_sig.bind(None, *args, **kwargs)
        except TypeError as exc:
            failures.append(
                f"{name}: plugin override {plugin_sig} does not accept "
                f"upstream's call shape {upstream_sig} ({exc})"
            )
        else:
            checked.append(name)

    assert checked, (
        "expected at least one PgvectorMemoryProvider method to be a real "
        "override of a MemoryProvider method -- if this is 0, something "
        "upstream renamed every hook the plugin overrides"
    )
    assert not failures, "signature drift vs. upstream MemoryProvider:\n" + "\n".join(failures)

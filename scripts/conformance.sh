#!/usr/bin/env bash
# scripts/conformance.sh [REF] [-- pytest args...] -- upstream conformance
# harness (WP-F, docs/release/PLAN-1.0.md Sec 5).
#
# Clones hermes-agent at REF (default: the SHA in conformance/HERMES_AGENT_REF;
# a branch name such as `main` also works) into .cache/hermes-agent, builds
# .venv-conformance, installs hermes-agent into it editable -- falling back to
# a PYTHONPATH import of its source tree if that does not leave
# `agent.memory_provider` importable -- installs this package
# (`pip install -e .[test]`), then runs `pytest conformance -q` (any extra
# arguments are passed straight through to pytest).
#
# Usage:
#   scripts/conformance.sh                       # pinned ref
#   scripts/conformance.sh main                  # a branch name instead
#   scripts/conformance.sh -k test_entry_point   # pinned ref, pytest -k filter
#   scripts/conformance.sh d0288be5 -k test_entry_point
#
# Env:
#   PYTHON                  interpreter used to create .venv-conformance (default: python3)
#   HERMES_AGENT_REPO_URL   override the clone URL (default: upstream hermes-agent)
#
# Exit status is pytest's -- non-zero on any test failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFORMANCE_DIR="$REPO_ROOT/conformance"
CACHE_DIR="$REPO_ROOT/.cache"
HERMES_AGENT_DIR="$CACHE_DIR/hermes-agent"
VENV_DIR="$REPO_ROOT/.venv-conformance"
PYTHON="${PYTHON:-python3}"
HERMES_AGENT_REPO_URL="${HERMES_AGENT_REPO_URL:-https://github.com/NousResearch/hermes-agent.git}"

# First arg is REF unless it looks like a pytest flag (e.g. `-k`); everything
# else is forwarded to pytest as-is.
REF=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    REF="$1"
    shift
fi
if [ -z "$REF" ]; then
    REF="$(cat "$CONFORMANCE_DIR/HERMES_AGENT_REF")"
fi

echo "==> hermes-memory-pgvector conformance harness"
echo "    hermes-agent ref: $REF"
echo "    clone dir:        $HERMES_AGENT_DIR"
echo "    venv:             $VENV_DIR"
echo

# 1. Clone (or reuse + fetch/checkout) hermes-agent --------------------------
if [ -d "$HERMES_AGENT_DIR/.git" ]; then
    echo "==> Reusing existing clone; fetching $REF..."
    git -C "$HERMES_AGENT_DIR" fetch --quiet origin "$REF" || true
    if ! git -C "$HERMES_AGENT_DIR" checkout --quiet "$REF" 2>/dev/null; then
        git -C "$HERMES_AGENT_DIR" checkout --quiet FETCH_HEAD
    fi
else
    echo "==> Cloning hermes-agent from $HERMES_AGENT_REPO_URL..."
    mkdir -p "$CACHE_DIR"
    git clone --quiet "$HERMES_AGENT_REPO_URL" "$HERMES_AGENT_DIR"
    if ! git -C "$HERMES_AGENT_DIR" checkout --quiet "$REF" 2>/dev/null; then
        git -C "$HERMES_AGENT_DIR" fetch --quiet origin "$REF"
        git -C "$HERMES_AGENT_DIR" checkout --quiet FETCH_HEAD
    fi
fi
RESOLVED_SHA="$(git -C "$HERMES_AGENT_DIR" rev-parse HEAD)"
echo "    resolved to: $RESOLVED_SHA"
echo

# 2. Create the conformance venv ---------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating $VENV_DIR ($PYTHON)..."
    "$PYTHON" -m venv "$VENV_DIR"
fi
VENV_PY="$VENV_DIR/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip

# 3. Install hermes-agent: editable, verified by import; PYTHONPATH fallback -
INSTALL_MODE="pip-editable"
FALLBACK_PYTHONPATH=""
echo "==> Installing hermes-agent (editable)..."
if "$VENV_PY" -m pip install --quiet -e "$HERMES_AGENT_DIR" \
        && "$VENV_PY" -c "import agent.memory_provider" >/dev/null 2>&1; then
    echo "    OK: editable install, agent.memory_provider imports"
    # hermes-agent's own pyproject.toml gates nearly all of ITS runtime deps
    # behind `python_version >= '3.14'` (its own comment there: "we only
    # support 3.14, but we need to allow old hermes installs on <3.14 to get
    # thru their update step"), so on an older interpreter a plain
    # `pip install -e` leaves almost none of them installed -- normal
    # hermes-agent installs resolve the full set via `uv.lock`, not plain
    # pip. This suite only ever imports a couple of hermes-agent's own
    # lightweight internal modules: agent.memory_provider and
    # agent.memory_manager always import with zero extra third-party deps;
    # plugins.memory (needed only by the entry-point conformance test)
    # additionally needs hermes_cli.config -> hermes_yaml -> ruamel.yaml.
    # Patch that one known gap directly instead of trying to reproduce
    # hermes-agent's full dependency resolution with plain pip -- any test
    # that still can't import what it needs skips cleanly on its own
    # (pytest.importorskip) rather than failing the run.
    "$VENV_PY" -m pip install --quiet "ruamel.yaml" || true
else
    echo "    editable install did not leave agent.memory_provider importable" \
         "-- falling back to a PYTHONPATH import of its source tree" >&2
    INSTALL_MODE="pythonpath"
    "$VENV_PY" -m pip uninstall --quiet -y hermes-agent >/dev/null 2>&1 || true
    FALLBACK_PYTHONPATH="$HERMES_AGENT_DIR"
    if ! PYTHONPATH="$FALLBACK_PYTHONPATH" "$VENV_PY" -c "import agent.memory_provider" >/dev/null 2>&1; then
        echo "    warning: agent.memory_provider is not importable via" \
             "PYTHONPATH either -- conformance tests will skip themselves" >&2
    fi
fi
echo "    install mode: $INSTALL_MODE"
echo

# 4. Install this package -----------------------------------------------------
echo "==> Installing hermes-memory-pgvector (editable, [test] extras)..."
"$VENV_PY" -m pip install --quiet -e "${REPO_ROOT}[test]"
echo

# 5. Run the suite -------------------------------------------------------------
echo "==> pytest conformance -q $*"
PYTHONPATH="${FALLBACK_PYTHONPATH}${FALLBACK_PYTHONPATH:+:}${PYTHONPATH:-}" \
    "$VENV_PY" -m pytest "$CONFORMANCE_DIR" -q "$@"

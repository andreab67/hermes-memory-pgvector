#!/usr/bin/env bash
# install.sh — install hermes-memory-pgvector into $HERMES_HOME/plugins/pgvector
#
# Two phases:
#   1. Python dependencies via pip (psycopg, psycopg-pool, PyYAML).
#   2. Plugin module copy into $HERMES_HOME/plugins/pgvector/ so the
#      hermes-agent discovery system (plugins/memory/__init__.py) finds it.
#
# Usage:
#   ./scripts/install.sh                # uses $HERMES_HOME or ~/.hermes
#   HERMES_HOME=/opt/hermes/.hermes ./scripts/install.sh
#
# After install, apply the schema migration as DB superuser and activate:
#   sudo -u postgres psql -d <db> -f $HERMES_HOME/plugins/pgvector/migrations/001_schema.sql
#   hermes config set memory.provider pgvector
#   sudo systemctl restart hermes.service
#   hermes memory status   # expect: Provider: pgvector; Status: available

set -euo pipefail

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DIR="$HERMES_HOME/plugins/pgvector"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> hermes-memory-pgvector installer"
echo "    HERMES_HOME: $HERMES_HOME"
echo "    target:      $PLUGIN_DIR"
echo

# 1. Python dependencies
echo "==> Installing Python dependencies..."
PIP="${PIP:-pip}"
"$PIP" install \
    'psycopg[binary]>=3.3.4,<4' \
    'psycopg-pool>=3.3.1,<4' \
    'PyYAML>=6.0,<7'

# 2. Copy plugin module
echo
echo "==> Installing plugin module..."
mkdir -p "$HERMES_HOME/plugins"

if [[ -d "$PLUGIN_DIR" ]]; then
    BACKUP="${PLUGIN_DIR}.bak.$(date +%Y%m%d-%H%M%S)"
    echo "    existing install detected, backing up to $BACKUP"
    mv "$PLUGIN_DIR" "$BACKUP"
fi

cp -r "$REPO_ROOT/pgvector" "$PLUGIN_DIR"
echo "    copied $REPO_ROOT/pgvector → $PLUGIN_DIR"

# 3. Next steps
cat <<EOF

==> Plugin files installed.

NOTE: for production installs prefer the pip-native path (v0.4.2+):
    pip install hermes-memory-pgvector && hermes-pgvector install
This script remains the from-clone alternative.

Next steps (admin once):
  1. Apply ALL migrations (schema + attribution + FTS indexes + runtime
     grants) in one shot — from the repo root:
       python -m pgvector migrate --admin-dsn \\
           "dbname=<your-memory-db> user=postgres host=/var/run/postgresql"
     (or apply migrations/00*.sql in lexical order with psql -f; migration
      004 grants the runtime 'hermes' role DML on the 001 tables — no manual
      OWNER transfer needed anymore)

  2. Activate the provider:
       hermes config set memory.provider pgvector
       sudo systemctl restart hermes.service   # or however you run hermes

  3. Verify:
       hermes memory status
       # expect: Provider: pgvector; Status: available

See $REPO_ROOT/README.md for the full operator docs, config knobs, and
multi-agent setup (per-minion X-Hermes-Session-Key themes).
EOF

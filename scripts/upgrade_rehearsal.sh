#!/usr/bin/env bash
# scripts/upgrade_rehearsal.sh -- Gate G1.6 (docs/release/PLAN-1.0.md Sec 6
# item 6): rehearse the OLD_VERSION -> this-branch upgrade end to end
# against a real Postgres.
#
# Fresh database -> install OLD_VERSION (default 0.5.5) from PyPI in a
# throwaway venv -> `migrate` -> seed real data through THAT version's
# `PgvectorMemoryProvider` class (scripts/upgrade_rehearsal_seed.py; a
# mixed-case identity, a lowercase one, several memory adds including one
# exact duplicate, turns, a delegation, and NULL-embedding rows) -> install
# THIS branch (non-editable, from the repo) in a second venv -> `migrate
# --runtime-role hermes` -> verify: row counts unchanged, the old unique
# constraint is gone and the new md5 unique index is present, a duplicate
# add() is still a no-op through the new code, `backfill` exits 0 and a
# following dry-run reaches remaining: 0, `remap` of the mixed-case theme
# works, and a second `migrate --runtime-role` is a no-op.
#
# Every check prints PASS or FAIL and the run continues regardless, so one
# failure never hides the rest; the script exits non-zero iff any check
# FAILed.
#
# Usage:
#   scripts/upgrade_rehearsal.sh
#   KEEP=1 scripts/upgrade_rehearsal.sh              # keep venvs + workdir + the
#                                                     # rehearsal database for inspection
#   ADMIN_PORT=55433 scripts/upgrade_rehearsal.sh    # explicit port (skips the
#                                                     # harness port-file lookup)
#
# Env:
#   ADMIN_HOST, ADMIN_USER, ADMIN_PASSWORD   admin connection (default:
#                                             127.0.0.1 / postgres / postgres)
#   ADMIN_PORT       admin connection port. When unset, read from the shared
#                     test harness's port file (scripts/test-env.sh's own
#                     TEST_ENV_STATE_DIR convention):
#                       "${TEST_ENV_STATE_DIR:-$HOME/.hpg-test-env}/pg17.port"
#                     Run `scripts/test-env.sh up pg17` first if that file
#                     does not exist yet, or set ADMIN_PORT explicitly.
#   DB_NAME          database to drop + recreate (default: upgrade_rehearsal)
#   EMBED_URL        working embed endpoint (default: http://127.0.0.1:11999,
#                     the fake embed server scripts/test-env.sh starts)
#   NULL_EMBED_URL   an endpoint nothing listens on, for the NULL-embedding
#                     seed rows (default: http://127.0.0.1:1)
#   OLD_VERSION      the pre-upgrade PyPI version to rehearse from (default: 0.5.5)
#   RUNTIME_ROLE     runtime role passed to `migrate --runtime-role` (default: hermes;
#                     the harness's runtime role is hermes/hermes)
#   PYTHON           interpreter used to create both venvs (default: python3)
#   WORKDIR          scratch dir for venvs + seed tmp homes. Default: a fresh
#                     `mktemp -d` under ${TMPDIR:-/tmp} -- pip/venv are much
#                     faster (and avoid transient ENOENTs seen under WSL) on a
#                     native filesystem than under a Windows-mounted /mnt/c
#                     checkout. Set WORKDIR explicitly to use
#                     .cache/upgrade-rehearsal/ under the repo instead (already
#                     gitignored); an explicit WORKDIR is wiped at the start of
#                     every run. Removed at the end unless KEEP=1.
#   KEEP=1           keep WORKDIR and the rehearsal database after the run,
#                     for inspection
#
# Needs: Docker running the pg17 container from scripts/test-env.sh, network
# access to PyPI, and a python3 with the `venv` module.
#
# Exit status: 0 iff every check PASSes; non-zero otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ADMIN_HOST="${ADMIN_HOST:-127.0.0.1}"
ADMIN_USER="${ADMIN_USER:-postgres}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-postgres}"
DB_NAME="${DB_NAME:-upgrade_rehearsal}"
EMBED_URL="${EMBED_URL:-http://127.0.0.1:11999}"
NULL_EMBED_URL="${NULL_EMBED_URL:-http://127.0.0.1:1}"
OLD_VERSION="${OLD_VERSION:-0.5.5}"
RUNTIME_ROLE="${RUNTIME_ROLE:-hermes}"
PYTHON="${PYTHON:-python3}"
KEEP="${KEEP:-0}"

if [ -z "${ADMIN_PORT:-}" ]; then
  STATE_DIR="${TEST_ENV_STATE_DIR:-$HOME/.hpg-test-env}"
  PORT_FILE="$STATE_DIR/pg17.port"
  if [ ! -f "$PORT_FILE" ]; then
    echo "error: ADMIN_PORT is not set and no harness port file at $PORT_FILE" >&2
    echo "  run 'scripts/test-env.sh up pg17' first, or set ADMIN_PORT explicitly" >&2
    exit 2
  fi
  ADMIN_PORT="$(tr -d '[:space:]' < "$PORT_FILE")"
fi

if [ -n "${WORKDIR:-}" ]; then
  rm -rf "$WORKDIR"
  mkdir -p "$WORKDIR"
else
  WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/hpg-upgrade-rehearsal.XXXXXX")"
fi

# `python -m hermes_pgvector ...` and `python -` (stdin scripts) both put the
# CURRENT DIRECTORY first on sys.path -- if cwd were $REPO_ROOT, that would
# shadow BOTH venvs' installed packages with the repo's own hermes_pgvector/
# source tree (a real, easy-to-miss gotcha: every `-m` call below would then
# run this checkout's code instead of the pinned OLD_VERSION or the installed
# branch, silently turning the whole rehearsal into a no-op). WORKDIR never
# contains a `hermes_pgvector` directory, so running from there is immune.
cd "$WORKDIR"

# shellcheck disable=SC2317 # invoked indirectly via `trap ... EXIT` below
cleanup() {
  if [ "$KEEP" = "1" ]; then
    echo "KEEP=1 -- leaving $WORKDIR and database \"$DB_NAME\" in place"
    return 0
  fi
  PGPASSWORD="$ADMIN_PASSWORD" psql -h "$ADMIN_HOST" -p "$ADMIN_PORT" -U "$ADMIN_USER" \
    -d postgres -tAc "DROP DATABASE IF EXISTS \"$DB_NAME\";" >/dev/null 2>&1 || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "==> hermes-memory-pgvector upgrade rehearsal ($OLD_VERSION -> this branch)"
echo "    admin:   $ADMIN_USER@$ADMIN_HOST:$ADMIN_PORT/$DB_NAME"
echo "    embed:   $EMBED_URL (working) / $NULL_EMBED_URL (unreachable, for NULL-embedding rows)"
echo "    workdir: $WORKDIR"
echo

ADMIN_DSN="host=$ADMIN_HOST port=$ADMIN_PORT user=$ADMIN_USER password=$ADMIN_PASSWORD dbname=$DB_NAME"
RUNTIME_DSN="host=$ADMIN_HOST port=$ADMIN_PORT user=hermes password=hermes dbname=$DB_NAME"

psql_admin() { # $1 = statement, run against dbname=postgres (DROP/CREATE DATABASE)
  PGPASSWORD="$ADMIN_PASSWORD" psql -h "$ADMIN_HOST" -p "$ADMIN_PORT" -U "$ADMIN_USER" \
    -d postgres -v ON_ERROR_STOP=1 -tAc "$1"
}

psql_db() { # $1 = statement, run against $DB_NAME, prints the trimmed scalar result
  PGPASSWORD="$ADMIN_PASSWORD" psql -h "$ADMIN_HOST" -p "$ADMIN_PORT" -U "$ADMIN_USER" \
    -d "$DB_NAME" -v ON_ERROR_STOP=1 -tAc "$1"
}

row_counts() { # "memory_entries conversations memory_agents memory_agent_edges"
  local me conv agents edges
  me="$(psql_db "SELECT count(*) FROM memory_entries;")"
  conv="$(psql_db "SELECT count(*) FROM conversations;")"
  agents="$(psql_db "SELECT count(*) FROM memory_agents;")"
  edges="$(psql_db "SELECT count(*) FROM memory_agent_edges;")"
  echo "$me $conv $agents $edges"
}

echo "==> 1/7 fresh database: $DB_NAME"
psql_admin "DROP DATABASE IF EXISTS \"$DB_NAME\";" >/dev/null
psql_admin "CREATE DATABASE \"$DB_NAME\" OWNER $ADMIN_USER;" >/dev/null
echo

echo "==> 2/7 old side: venv + hermes-memory-pgvector==$OLD_VERSION"
OLD_VENV="$WORKDIR/venv-old"
"$PYTHON" -m venv "$OLD_VENV"
OLD_PY="$OLD_VENV/bin/python"
"$OLD_PY" -m pip install --quiet --upgrade pip
"$OLD_PY" -m pip install --quiet "hermes-memory-pgvector==$OLD_VERSION"
"$OLD_PY" -m pip show hermes-memory-pgvector | grep -E '^(Name|Version):' | sed 's/^/    /'
echo

echo "==> 3/7 old side: migrate (no --runtime-role in $OLD_VERSION)"
"$OLD_PY" -m hermes_pgvector migrate --admin-dsn "$ADMIN_DSN"
echo

echo "==> 4/7 old side: seed through the $OLD_VERSION provider class"
SEED_DSN="$RUNTIME_DSN" \
SEED_EMBED_URL="$EMBED_URL" \
SEED_NULL_EMBED_URL="$NULL_EMBED_URL" \
SEED_HERMES_HOME_BASE="$WORKDIR/hermes-homes" \
  "$OLD_PY" "$SCRIPT_DIR/upgrade_rehearsal_seed.py"
echo

COUNTS_AFTER_SEED="$(row_counts)"
echo "    row counts after seeding (memory_entries conversations memory_agents memory_agent_edges): $COUNTS_AFTER_SEED"
echo

echo "==> 5/7 new side: venv + this branch (non-editable install)"
NEW_VENV="$WORKDIR/venv-new"
"$PYTHON" -m venv "$NEW_VENV"
NEW_PY="$NEW_VENV/bin/python"
"$NEW_PY" -m pip install --quiet --upgrade pip
"$NEW_PY" -m pip install --quiet "$REPO_ROOT"
echo -n "    "; "$NEW_PY" -m hermes_pgvector --version
echo

echo "==> 6/7 new side: migrate --runtime-role $RUNTIME_ROLE"
"$NEW_PY" -m hermes_pgvector migrate --admin-dsn "$ADMIN_DSN" --runtime-role "$RUNTIME_ROLE"
echo

COUNTS_AFTER_MIGRATE="$(row_counts)"
echo "    row counts after migrate: $COUNTS_AFTER_MIGRATE"
echo

echo "==> 7/7 verification"
echo

# From here on every check is fail-soft: a FAILing check must not stop the
# rest of the rehearsal from running and reporting its own result.
set +e
FAILED=0

report() { # $1 = description, $2 = 0 (pass) or non-zero (fail)
  if [ "$2" -eq 0 ]; then
    echo "PASS: $1"
  else
    echo "FAIL: $1"
    FAILED=1
  fi
}

# --- 1. row counts unchanged across the upgrade -----------------------------
if [ "$COUNTS_AFTER_SEED" = "$COUNTS_AFTER_MIGRATE" ]; then
  report "row counts unchanged across migrate ($COUNTS_AFTER_MIGRATE)" 0
else
  report "row counts unchanged across migrate (before=[$COUNTS_AFTER_SEED] after=[$COUNTS_AFTER_MIGRATE])" 1
fi

# --- 2. memory_entries_unique gone, memory_entries_unique_md5 present -------
OLD_CONSTRAINT_COUNT="$(psql_db "SELECT count(*) FROM pg_constraint WHERE conname = 'memory_entries_unique';")"
NEW_INDEX_COUNT="$(psql_db "SELECT count(*) FROM pg_indexes WHERE indexname = 'memory_entries_unique_md5';")"
if [ "$OLD_CONSTRAINT_COUNT" = "0" ] && [ "$NEW_INDEX_COUNT" = "1" ]; then
  report "memory_entries_unique dropped, memory_entries_unique_md5 present" 0
else
  report "memory_entries_unique dropped, memory_entries_unique_md5 present (constraint_count=$OLD_CONSTRAINT_COUNT index_count=$NEW_INDEX_COUNT)" 1
fi

# --- 3. duplicate add() is still a no-op through the NEW code --------------
DUP_CHECK_OUT="$("$NEW_PY" - "$RUNTIME_DSN" <<'PYEOF'
import sys

from hermes_pgvector.store import MemoryStore

dsn = sys.argv[1]
store = MemoryStore(dsn)
try:
    before = store.count()
    result = store.add(
        agent_identity="Marketing",
        target="memory",
        content="Schedule social posts every Tuesday and Thursday for the product launch.",
    )
    after = store.count()
finally:
    store.close()
print(f"add() returned {result!r}; count before={before} after={after}")
sys.exit(0 if (result is None and after == before) else 1)
PYEOF
)"; DUP_CHECK_RC=$?
report "duplicate add() is a no-op through the new code ($DUP_CHECK_OUT)" "$DUP_CHECK_RC"

# --- 4. backfill exits 0 and a dry-run afterwards shows remaining: 0 -------
BACKFILL_OUT="$("$NEW_PY" -m hermes_pgvector backfill --dsn "$RUNTIME_DSN" --embed-url "$EMBED_URL")"; BACKFILL_RC=$?
echo "$BACKFILL_OUT"
if [ "$BACKFILL_RC" -eq 0 ]; then
  DRYRUN_OUT="$("$NEW_PY" -m hermes_pgvector backfill --dsn "$RUNTIME_DSN" --embed-url "$EMBED_URL" --dry-run)"
  echo "$DRYRUN_OUT"
  NONZERO_REMAINING="$("$PYTHON" -c '
import json
import sys

data = json.loads(sys.argv[1])
bad = [t for t, info in data.items() if (info.get("remaining") or 0) != 0]
print(",".join(bad))
' "$DRYRUN_OUT")"; PARSE_RC=$?
  if [ "$PARSE_RC" -ne 0 ]; then
    report "backfill exits 0 and dry-run reaches remaining: 0 for every table (could not parse dry-run JSON)" 1
  elif [ -z "$NONZERO_REMAINING" ]; then
    report "backfill exits 0 and dry-run reaches remaining: 0 for every table" 0
  else
    report "backfill exits 0 and dry-run reaches remaining: 0 for every table (nonzero in: $NONZERO_REMAINING)" 1
  fi
else
  echo "$BACKFILL_OUT" >&2
  report "backfill exits 0 (exit=$BACKFILL_RC)" 1
fi

# --- 5. remap of the mixed-case theme works ---------------------------------
REMAP_OUT="$("$NEW_PY" -m hermes_pgvector remap --dsn "$RUNTIME_DSN" --old Marketing --new marketing --execute)"; REMAP_RC=$?
echo "$REMAP_OUT"
OLD_IDENTITY_ME="$(psql_db "SELECT count(*) FROM memory_entries WHERE agent_identity = 'Marketing';")"
OLD_IDENTITY_CONV="$(psql_db "SELECT count(*) FROM conversations WHERE agent_identity = 'Marketing';")"
NEW_IDENTITY_ME="$(psql_db "SELECT count(*) FROM memory_entries WHERE agent_identity = 'marketing';")"
NEW_IDENTITY_CONV="$(psql_db "SELECT count(*) FROM conversations WHERE agent_identity = 'marketing';")"
if [ "$REMAP_RC" -eq 0 ] && [ "$OLD_IDENTITY_ME" = "0" ] && [ "$OLD_IDENTITY_CONV" = "0" ] \
    && [ "$NEW_IDENTITY_ME" != "0" ] && [ "$NEW_IDENTITY_CONV" != "0" ]; then
  report "remap --old Marketing --new marketing (old identity gone, new has memory_entries=$NEW_IDENTITY_ME conversations=$NEW_IDENTITY_CONV)" 0
else
  report "remap --old Marketing --new marketing (exit=$REMAP_RC old_memory_entries=$OLD_IDENTITY_ME old_conversations=$OLD_IDENTITY_CONV new_memory_entries=$NEW_IDENTITY_ME new_conversations=$NEW_IDENTITY_CONV)" 1
fi

# --- 6. a second migrate --runtime-role is a no-op -------------------------
COUNTS_BEFORE_SECOND_MIGRATE="$(row_counts)"
SECOND_MIGRATE_OUT="$("$NEW_PY" -m hermes_pgvector migrate --admin-dsn "$ADMIN_DSN" --runtime-role "$RUNTIME_ROLE")"; SECOND_MIGRATE_RC=$?
echo "$SECOND_MIGRATE_OUT"
COUNTS_AFTER_SECOND_MIGRATE="$(row_counts)"
if [ "$SECOND_MIGRATE_RC" -eq 0 ] && [ "$COUNTS_BEFORE_SECOND_MIGRATE" = "$COUNTS_AFTER_SECOND_MIGRATE" ]; then
  report "second migrate --runtime-role $RUNTIME_ROLE is a no-op (counts $COUNTS_AFTER_SECOND_MIGRATE)" 0
else
  report "second migrate --runtime-role $RUNTIME_ROLE is a no-op (exit=$SECOND_MIGRATE_RC before=[$COUNTS_BEFORE_SECOND_MIGRATE] after=[$COUNTS_AFTER_SECOND_MIGRATE])" 1
fi

echo
if [ "$FAILED" -eq 0 ]; then
  echo "==> upgrade rehearsal PASSED"
else
  echo "==> upgrade rehearsal FAILED"
fi
exit "$FAILED"

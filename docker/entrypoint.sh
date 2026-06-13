#!/bin/bash
# entrypoint.sh — dispatches on $CMD_PROFILE
#
# Profiles:
#   test  → wait for Postgres + apply migration + run pytest
#   mcp   → wait for Postgres + apply migration (idempotent) + start MCP server
#   shell → drop to bash for debugging
#
# NOTE: shebang is bash (not sh) so /dev/tcp/host/port connectivity checks work.
#       python:3.11-slim-bookworm ships with bash at /bin/bash.
#
# Forked from andreab67/hermes-hexus (BSD-3-Clause)

set -eu

PROFILE="${CMD_PROFILE:-shell}"
PG_TEST_HOST="${PG_TEST_HOST:-pg}"
PG_TEST_PORT="${PG_TEST_PORT:-5432}"
PG_TEST_DSN="${PG_TEST_DSN:-}"

log() {
    printf "[entrypoint] %s\n" "$*"
}

wait_for_tcp() {
    host="$1"
    port="$2"
    log "waiting for tcp://$host:$port ..."
    i=0
    while [ "$i" -lt 60 ]; do
        if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
            exec 3<&- 3>&-
            log "tcp://$host:$port reachable after ${i}s"
            return 0
        fi
        i=$((i + 1))
        sleep 1
    done
    log "ERROR: tcp://$host:$port not reachable after 60s"
    return 1
}

wait_for_schema() {
    dsn="$1"
    log "waiting for schema (memory_entries and conversations tables) ..."
    i=0
    while [ "$i" -lt 60 ]; do
        if psql "$dsn" -tAc "SELECT to_regclass('memory_entries')" 2>/dev/null | grep -q memory_entries; then
            if psql "$dsn" -tAc "SELECT to_regclass('conversations')" 2>/dev/null | grep -q conversations; then
                log "schema ready after ${i}s"
                return 0
            fi
        fi
        i=$((i + 1))
        sleep 1
    done
    log "ERROR: schema not ready after 60s"
    return 1
}

apply_migration() {
    dsn="$1"
    for migration in /app/hexus/migrations/*.sql; do
        log "applying migration from $migration ..."
        if psql "$dsn" -v ON_ERROR_STOP=1 -f "$migration" 2>&1; then
            log "migration applied"
        else
            log "ERROR: migration failed"
            return 1
        fi
    done
}

case "$PROFILE" in
    test)
        wait_for_tcp "$PG_TEST_HOST" "$PG_TEST_PORT"
        # Apply migrations (fully idempotent)
        apply_migration "$PG_TEST_DSN"
        log "running pytest"
        exec pytest tests/ -v --tb=short
        ;;
    mcp)
        # The mcp service is the long-lived MCP server (streamable-http
        # by default; flip HEXUS_TRANSPORT=stdio for an
        # editor-launched bridge). The entrypoint waits for the schema
        # to be present (idempotent — apply_migration_as_admin is the
        # upstream plugin's admin path, but for the mcp container we
        # use the same psql -f path the test profile uses, since we
        # have admin creds via PG_TEST_DSN anyway).
        wait_for_tcp "$PG_TEST_HOST" "${PG_MCP_PORT:-5432}"
        # Apply migrations (fully idempotent)
        apply_migration "${HEXUS_DSN:-$PG_TEST_DSN}"
        log "starting MCP server (transport=${HEXUS_TRANSPORT:-http})"
        exec hexus-mcp serve \
            --transport "${HEXUS_TRANSPORT:-http}" \
            --host "${MCP_HOST:-0.0.0.0}" \
            --port "${MCP_PORT:-8000}" \
            --dsn "${HEXUS_DSN:-$PG_TEST_DSN}" \
            --agent-identity "${HEXUS_AGENT_IDENTITY:-default}" \
            --log-level "${HEXUS_LOG_LEVEL:-INFO}"
        ;;
    shell|*)
        log "profile=shell — dropping to bash"
        exec bash
        ;;
esac

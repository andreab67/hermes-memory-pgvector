#!/usr/bin/env bash
# scripts/test-env.sh — local test harness for hermes-memory-pgvector (WP-0).
#
# Starts a throwaway `pgvector/pgvector` Postgres container plus a stdlib-only
# fake embedding server, creates a runtime role and a scratch database with
# migrations applied, and prints the env vars the test suite and the repro
# scripts read. Needs only Docker and a Python >= 3.11 on PATH.
#
# Usage:
#   scripts/test-env.sh up [pg16|pg17|pg18]     # start Postgres + fake embed server
#   scripts/test-env.sh db <name> [--runtime-role ROLE]
#                                                # (re)create + migrate a database
#   scripts/test-env.sh down                    # stop and remove everything
#
# Typical flow:
#   scripts/test-env.sh up pg17
#   eval "$(scripts/test-env.sh db hermes_test)"
#   $PYTHON -m pytest -q
#   scripts/test-env.sh down
#
# PYTHON defaults to `python3`; set it to a venv interpreter that has this
# package installed (editable or not) when running `db` or the suite itself.
#
# State (chosen ports, the embed server's pidfile/log) lives under a
# gitignored directory so `db` can find the port `up` picked; see STATE_DIR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# TEST_ENV_STATE_DIR lets several git worktrees share one running instance
# (the `db` command of a second checkout must find the port `up` chose).
STATE_DIR="${TEST_ENV_STATE_DIR:-$REPO_ROOT/.cache/test-env}"
PYTHON="${PYTHON:-python3}"

EMBED_HOST="127.0.0.1"
EMBED_PORT="11999"
EMBED_PIDFILE="$STATE_DIR/embed-server.pid"
EMBED_LOGFILE="$STATE_DIR/embed-server.log"

default_port_for() {
  case "$1" in
    pg16) echo 55416 ;;
    pg17) echo 55432 ;;
    pg18) echo 55418 ;;
    *) echo "unknown tag: $1" >&2; return 1 ;;
  esac
}

port_state_file() {
  echo "$STATE_DIR/$1.port"
}

port_in_use() {
  # True (0) if something is already listening on 127.0.0.1:$1.
  "$PYTHON" - "$1" <<'PYEOF' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.3)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
else:
    sys.exit(0)
finally:
    s.close()
PYEOF
}

find_free_port() {
  local port="$1"
  while port_in_use "$port"; do
    port=$((port + 1))
  done
  echo "$port"
}

container_running() {
  [ -n "$(docker ps -q -f "name=^${1}$")" ]
}

container_exists() {
  [ -n "$(docker ps -aq -f "name=^${1}$")" ]
}

wait_for_postgres() {
  local container="$1"
  local tries=60
  while [ "$tries" -gt 0 ]; do
    if docker exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
      return 0
    fi
    tries=$((tries - 1))
    sleep 1
  done
  echo "error: $container did not become ready within 60s" >&2
  return 1
}

ensure_hermes_role() {
  local container="$1"
  docker exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres -c "
    DO \$\$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hermes') THEN
        CREATE ROLE hermes LOGIN PASSWORD 'hermes';
      END IF;
    END
    \$\$;
  " >/dev/null
}

embed_server_running() {
  [ -f "$EMBED_PIDFILE" ] && kill -0 "$(cat "$EMBED_PIDFILE")" 2>/dev/null
}

start_embed_server() {
  mkdir -p "$STATE_DIR"
  if embed_server_running; then
    echo "fake embed server already running (pid $(cat "$EMBED_PIDFILE")) on http://$EMBED_HOST:$EMBED_PORT"
    return 0
  fi
  nohup "$PYTHON" "$REPO_ROOT/tests/fake_embed_server.py" \
    --host "$EMBED_HOST" --port "$EMBED_PORT" \
    >"$EMBED_LOGFILE" 2>&1 &
  local pid=$!
  echo "$pid" >"$EMBED_PIDFILE"
  # Give it a moment and confirm it actually started.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "error: fake embed server exited immediately; see $EMBED_LOGFILE" >&2
      cat "$EMBED_LOGFILE" >&2 || true
      return 1
    fi
    if port_in_use "$EMBED_PORT"; then
      echo "started fake embed server (pid $pid) on http://$EMBED_HOST:$EMBED_PORT"
      return 0
    fi
    sleep 0.3
  done
  echo "error: fake embed server did not open $EMBED_HOST:$EMBED_PORT in time" >&2
  return 1
}

stop_embed_server() {
  if [ -f "$EMBED_PIDFILE" ]; then
    local pid
    pid="$(cat "$EMBED_PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$EMBED_PIDFILE"
  fi
}

cmd_up() {
  local tag="${1:-pg17}"
  case "$tag" in
    pg16|pg17|pg18) ;;
    *) echo "usage: $0 up [pg16|pg17|pg18]" >&2; return 2 ;;
  esac

  mkdir -p "$STATE_DIR"
  local container="hpg-test-$tag"
  local port_file
  port_file="$(port_state_file "$tag")"

  if container_running "$container"; then
    echo "container $container already up on port $(cat "$port_file" 2>/dev/null || echo '?')"
  else
    if container_exists "$container"; then
      docker rm -f "$container" >/dev/null
    fi
    local default_port
    default_port="$(default_port_for "$tag")"
    local port
    port="$(find_free_port "$default_port")"
    if [ "$port" != "$default_port" ]; then
      echo "port $default_port taken; using $port instead"
    fi
    echo "$port" >"$port_file"

    docker run -d --name "$container" \
      -e POSTGRES_PASSWORD=postgres \
      -p "127.0.0.1:${port}:5432" \
      --tmpfs /var/lib/postgresql/data \
      "pgvector/pgvector:${tag}" >/dev/null

    wait_for_postgres "$container"
    echo "started $container on 127.0.0.1:$port"
  fi

  ensure_hermes_role "$container"
  start_embed_server
}

resolve_tag_and_port() {
  # Echoes "<tag> <port>" for the running test-env instance to use with `db`.
  # Prefers $TEST_ENV_TAG when set; otherwise picks the one state file if
  # there's exactly one, else prefers pg17, else errors listing the choices.
  mkdir -p "$STATE_DIR"
  if [ -n "${TEST_ENV_TAG:-}" ]; then
    local f
    f="$(port_state_file "$TEST_ENV_TAG")"
    if [ ! -f "$f" ]; then
      echo "error: TEST_ENV_TAG=$TEST_ENV_TAG but $f does not exist; run 'up $TEST_ENV_TAG' first" >&2
      return 2
    fi
    echo "$TEST_ENV_TAG $(cat "$f")"
    return 0
  fi

  local files=("$STATE_DIR"/pg1*.port)
  local existing=()
  for f in "${files[@]}"; do
    [ -f "$f" ] && existing+=("$f")
  done

  if [ "${#existing[@]}" -eq 0 ]; then
    echo "error: no test-env instance is up; run '$0 up [pg16|pg17|pg18]' first" >&2
    return 2
  elif [ "${#existing[@]}" -eq 1 ]; then
    local base
    base="$(basename "${existing[0]}" .port)"
    echo "$base $(cat "${existing[0]}")"
    return 0
  else
    local pref
    pref="$(port_state_file pg17)"
    if [ -f "$pref" ]; then
      echo "pg17 $(cat "$pref")"
      return 0
    fi
    echo "error: multiple test-env instances are up (${existing[*]}); set TEST_ENV_TAG=pg16|pg17|pg18" >&2
    return 2
  fi
}

cmd_db() {
  local name="${1:-}"
  if [ -z "$name" ]; then
    echo "usage: $0 db <name> [--runtime-role ROLE]" >&2
    return 2
  fi
  shift
  local runtime_role=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --runtime-role) runtime_role="${2:-}"; shift 2 ;;
      *) echo "unknown argument: $1" >&2; return 2 ;;
    esac
  done

  local tag_and_port
  tag_and_port="$(resolve_tag_and_port)"
  local tag="${tag_and_port% *}"
  local port="${tag_and_port#* }"
  local container="hpg-test-$tag"

  if ! container_running "$container"; then
    echo "error: $container is not running; run '$0 up $tag' first" >&2
    return 2
  fi

  docker exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres -c \
    "DROP DATABASE IF EXISTS \"$name\";" >/dev/null
  docker exec "$container" psql -v ON_ERROR_STOP=1 -U postgres -d postgres -c \
    "CREATE DATABASE \"$name\" OWNER postgres;" >/dev/null

  local admin_dsn="host=127.0.0.1 port=${port} user=postgres password=postgres dbname=${name}"
  local runtime_dsn="host=127.0.0.1 port=${port} user=hermes password=hermes dbname=${name}"

  local migrate_cmd=("$PYTHON" -m hermes_pgvector migrate --admin-dsn "$admin_dsn")
  if [ -n "$runtime_role" ]; then
    local help_text
    help_text="$("$PYTHON" -m hermes_pgvector migrate --help 2>&1 || true)"
    if printf '%s' "$help_text" | grep -q -- "--runtime-role"; then
      migrate_cmd+=(--runtime-role "$runtime_role")
    else
      echo "note: this CLI's 'migrate' has no --runtime-role yet; ignoring --runtime-role $runtime_role" >&2
    fi
  fi

  "${migrate_cmd[@]}" >&2

  echo "export PG_TEST_DSN='${runtime_dsn}'"
  echo "export PG_TEST_ADMIN_DSN='${admin_dsn}'"
  # docs/code-review/repro-2026-09-26/*.py and scripts/install.sh's example
  # both read ADMIN_DSN (not PG_TEST_ADMIN_DSN) for the same value.
  echo "export ADMIN_DSN='${admin_dsn}'"
  echo "export PG_TEST_EMBED_URL='http://${EMBED_HOST}:${EMBED_PORT}'"
}

cmd_down() {
  local containers
  containers="$(docker ps -aq -f "name=^hpg-test-pg1[678]$")"
  if [ -n "$containers" ]; then
    # shellcheck disable=SC2086
    docker rm -f $containers >/dev/null
  fi
  stop_embed_server
  rm -f "$STATE_DIR"/pg1*.port
  echo "test-env down"
}

main() {
  local cmd="${1:-}"
  [ $# -gt 0 ] && shift
  case "$cmd" in
    up) cmd_up "$@" ;;
    db) cmd_db "$@" ;;
    down) cmd_down "$@" ;;
    *)
      echo "usage: $0 {up [pg16|pg17|pg18]|db <name> [--runtime-role ROLE]|down}" >&2
      return 2
      ;;
  esac
}

main "$@"

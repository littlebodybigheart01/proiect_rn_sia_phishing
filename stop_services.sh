#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"

API_PID_FILE="$RUNTIME_DIR/api.pid"
UI_PID_FILE="$RUNTIME_DIR/streamlit.pid"

stop_by_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [ ! -f "$pid_file" ]; then
    echo "[INFO] $name: PID file lipsa."
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [ -z "${pid:-}" ]; then
    rm -f "$pid_file"
    echo "[INFO] $name: PID invalid, curatat."
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[INFO] Oprire $name (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "[WARN] $name nu s-a oprit, trimit SIGKILL..."
      kill -9 "$pid" 2>/dev/null || true
    fi
    echo "[OK] $name oprit."
  else
    echo "[INFO] $name nu mai ruleaza."
  fi

  rm -f "$pid_file"
}

stop_by_pid_file "API" "$API_PID_FILE"
stop_by_pid_file "Streamlit" "$UI_PID_FILE"

echo "[OK] Oprire finalizata."

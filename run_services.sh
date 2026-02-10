#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"
LOG_DIR="$ROOT_DIR/logs"

API_PID_FILE="$RUNTIME_DIR/api.pid"
UI_PID_FILE="$RUNTIME_DIR/streamlit.pid"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR"

if [ ! -d "$ROOT_DIR/.venv" ]; then
  echo "[ERROR] Nu exista .venv. Creeaza mediul virtual inainte sa rulezi scriptul."
  exit 1
fi

VENV_PY="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "[ERROR] Nu gasesc Python in .venv/bin/python"
  exit 1
fi

is_running() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

if is_running "$API_PID_FILE"; then
  echo "[INFO] API deja ruleaza (PID $(cat "$API_PID_FILE"))."
else
  echo "[INFO] Pornesc API pe http://127.0.0.1:8000 ..."
  nohup "$VENV_PY" -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 \
    > "$LOG_DIR/api.log" 2>&1 &
  echo $! > "$API_PID_FILE"
  sleep 1
fi

if is_running "$UI_PID_FILE"; then
  echo "[INFO] Streamlit deja ruleaza (PID $(cat "$UI_PID_FILE"))."
else
  echo "[INFO] Pornesc UI pe http://127.0.0.1:8501 ..."
  nohup "$VENV_PY" -m streamlit run app.py --server.port 8501 \
    > "$LOG_DIR/streamlit.log" 2>&1 &
  echo $! > "$UI_PID_FILE"
  sleep 1
fi

echo
echo "[OK] Servicii pornite."
echo "  UI  : http://127.0.0.1:8501"
echo "  API : http://127.0.0.1:8000/health"
echo
echo "Log-uri:"
echo "  $LOG_DIR/streamlit.log"
echo "  $LOG_DIR/api.log"
echo
echo "Oprire:"
echo "  bash stop_services.sh"

#!/bin/zsh
set -euo pipefail

LABEL="${DBX_PROXY_LABEL:-com.${USER}.dbx-proxy}"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
PORT="${DBX_PROXY_PORT:-19000}"
LOG_FILE="${DBX_PROXY_LOG_FILE:-/tmp/dbx-proxy.log}"

usage() {
  echo "Usage: $0 {start|stop|restart|status|log}"
}

load_service() {
  if [[ ! -f "$PLIST" ]]; then
    echo "Missing launchd plist: $PLIST"
    echo "Run ./install.sh first."
    exit 1
  fi

  launchctl bootstrap "gui/$UID" "$PLIST" 2>/dev/null || launchctl kickstart -k "gui/$UID/$LABEL"
}

unload_service() {
  launchctl bootout "gui/$UID" "$PLIST" 2>/dev/null || launchctl stop "$LABEL" 2>/dev/null || true
}

is_listening() {
  lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1
}

case "${1:-}" in
  start)
    load_service
    sleep 1
    if is_listening; then
      echo "dbx-proxy started on port $PORT"
    else
      echo "dbx-proxy did not start. Check: $LOG_FILE"
      exit 1
    fi
    ;;
  stop)
    unload_service
    echo "dbx-proxy stopped"
    ;;
  restart)
    unload_service
    sleep 1
    load_service
    sleep 1
    if is_listening; then
      echo "dbx-proxy restarted on port $PORT"
    else
      echo "dbx-proxy did not restart. Check: $LOG_FILE"
      exit 1
    fi
    ;;
  status)
    if is_listening; then
      echo "dbx-proxy is running on port $PORT"
      lsof -nP -i :"$PORT" -sTCP:LISTEN
    else
      echo "dbx-proxy is not running on port $PORT"
      exit 1
    fi
    ;;
  log)
    touch "$LOG_FILE"
    tail -f "$LOG_FILE"
    ;;
  *)
    usage
    exit 1
    ;;
esac

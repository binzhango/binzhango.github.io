#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PORT="${DBX_PROXY_PORT:-19000}"
LABEL="${DBX_PROXY_LABEL:-com.${USER}.dbx-proxy}"
PLIST_DEST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_FILE="${DBX_PROXY_LOG_FILE:-/tmp/dbx-proxy.log}"
ERR_FILE="${DBX_PROXY_ERR_FILE:-/tmp/dbx-proxy.err.log}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

usage() {
  cat <<USAGE
Usage:
  ./install.sh
  ./install.sh <workspace-hostname> <endpoint-name>

Environment:
  DBX_TOKEN               Databricks token. Required.
  DBX_PROXY_PORT          Local proxy port. Default: 19000.
  DBX_MODEL               Optional model id override. Defaults to endpoint name.
  DBX_INSECURE_SKIP_VERIFY Set to 1 only when your corporate TLS setup requires it.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${DBX_TOKEN:-}" ]]; then
  echo "DBX_TOKEN is not set."
  echo "Add this to ~/.zshrc, then reload your shell:"
  echo "  export DBX_TOKEN=dapi..."
  exit 1
fi

if [[ -n "${1:-}" && -n "${2:-}" ]]; then
  WORKSPACE_HOST="$1"
  ENDPOINT="$2"
else
  printf "Databricks workspace hostname: "
  read -r WORKSPACE_HOST
  printf "Databricks serving endpoint name: "
  read -r ENDPOINT
fi

if [[ -z "$WORKSPACE_HOST" || -z "$ENDPOINT" ]]; then
  echo "Workspace hostname and endpoint name are required."
  exit 1
fi

WORKSPACE_HOST="${WORKSPACE_HOST#https://}"
WORKSPACE_HOST="${WORKSPACE_HOST#http://}"
WORKSPACE_HOST="${WORKSPACE_HOST%/}"

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "python3 was not found in PATH."
  exit 1
fi

cat > "$SCRIPT_DIR/.env" <<ENV
DBX_WORKSPACE_HOST=$WORKSPACE_HOST
DBX_ENDPOINT=$ENDPOINT
DBX_MODEL=${DBX_MODEL:-$ENDPOINT}
DBX_PROXY_PORT=$PORT
ENV

chmod 600 "$SCRIPT_DIR/.env"
chmod +x "$SCRIPT_DIR/dbx-proxy.py" "$SCRIPT_DIR/dbx-proxy-wrapper.sh" "$SCRIPT_DIR/dbx-proxy-service.sh" "$SCRIPT_DIR/dbx-copilot-cli.sh"
mkdir -p "${HOME}/Library/LaunchAgents"

cat > "$PLIST_DEST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$SCRIPT_DIR/dbx-proxy-wrapper.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$LOG_FILE</string>
  <key>StandardErrorPath</key>
  <string>$ERR_FILE</string>
  <key>WorkingDirectory</key>
  <string>$SCRIPT_DIR</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$UID" "$PLIST_DEST" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$PLIST_DEST"
sleep 1

if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "dbx-proxy is running on http://localhost:$PORT"
  echo "Databricks endpoint: https://$WORKSPACE_HOST/serving-endpoints/$ENDPOINT/invocations"
  echo "Manage it with: $SCRIPT_DIR/dbx-proxy-service.sh {start|stop|restart|status|log}"
else
  echo "Service did not start. Check logs:"
  echo "  $LOG_FILE"
  echo "  $ERR_FILE"
  exit 1
fi

#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

PORT="${DBX_PROXY_PORT:-19000}"
MODEL="${DBX_MODEL:-${DBX_ENDPOINT:-}}"

if [[ -z "$MODEL" ]]; then
  echo "DBX_MODEL or DBX_ENDPOINT is required."
  echo "Run ./install.sh first, or export DBX_MODEL manually."
  return 1 2>/dev/null || exit 1
fi

if ! lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "dbx-proxy is not running on port $PORT."
  echo "Start it with: $SCRIPT_DIR/dbx-proxy-service.sh start"
  return 1 2>/dev/null || exit 1
fi

export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_BASE_URL="http://localhost:$PORT"
export COPILOT_PROVIDER_API_KEY=proxy
export COPILOT_MODEL="$MODEL"
export COPILOT_PROVIDER_MAX_PROMPT_TOKENS="${COPILOT_PROVIDER_MAX_PROMPT_TOKENS:-128000}"
export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS="${COPILOT_PROVIDER_MAX_OUTPUT_TOKENS:-16000}"

echo "Copilot CLI configured for $COPILOT_MODEL via $COPILOT_PROVIDER_BASE_URL"

if [[ "${(%):-%x}" == "$0" ]]; then
  if ! command -v copilot >/dev/null 2>&1; then
    echo "copilot CLI was not found in PATH."
    exit 1
  fi
  exec copilot "$@"
fi

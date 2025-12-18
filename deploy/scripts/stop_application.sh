#!/usr/bin/env sh
set -eu

APP_DIR="/opt/agent_p"

echo "stop_application: stopping ${APP_DIR}"
cd "${APP_DIR}" || exit 0

if command -v docker-compose >/dev/null 2>&1; then
  docker-compose down || true
elif docker compose version >/dev/null 2>&1; then
  docker compose down || true
else
  echo "stop_application: docker compose not available; nothing to stop"
fi



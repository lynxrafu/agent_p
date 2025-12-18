#!/usr/bin/env sh
set -eu

APP_DIR="/opt/agent_p"

echo "start_application: starting ${APP_DIR}"
cd "${APP_DIR}"

if command -v docker-compose >/dev/null 2>&1; then
  docker-compose up -d --build
elif docker compose version >/dev/null 2>&1; then
  docker compose up -d --build
else
  echo "ERROR: docker compose is required on the instance for this deployment strategy." >&2
  exit 1
fi



#!/usr/bin/env sh
set -eu

APP_DIR="/opt/agent_p"

echo "before_install: preparing ${APP_DIR}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is required on the instance for this deployment strategy." >&2
  exit 1
fi

mkdir -p "${APP_DIR}"

# Create a minimal .env if none exists yet (CodeDeploy copies env.example too).
if [ ! -f "${APP_DIR}/.env" ] && [ -f "${APP_DIR}/env.example" ]; then
  cp "${APP_DIR}/env.example" "${APP_DIR}/.env"
fi


